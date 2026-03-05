# import os, sys, torch, torchaudio

# os.environ["FAIRSEQ2_ASSET_DIR"] = (
#     "/Users/dattay/Documents/SpeechTranslation/checkpoints"
# )
# sys.path.insert(
#     0, "/Users/dattay/Documents/SpeechTranslation/src/seamless_communication/src"
# )

# from seamless_communication.inference import Translator

# AUDIO = "/Users/dattay/Documents/IShowSpeech/datasets/wavs/stream_uploaded_audio_2020-10-21_PfZxh4SNUOhN_qDqi06KafdgBPQugHoFYh3CvLDs1fGmmJeSPZZjTqtg2uPg5zpo-0624040-0624206.wav"
# SRC, TGT = "vie", "eng"

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# dtype = torch.float16 if device.type == "cuda" else torch.float32

# wav, sr = torchaudio.load(AUDIO)
# if sr != 16000:
#     wav = torchaudio.functional.resample(wav, sr, 16000)
# if wav.shape[0] > 1:
#     wav = wav.mean(0, keepdim=True)

# translator = Translator("seamlessM4T_v2_large", "vocoder_v2", device, dtype=dtype)

# text, _ = translator.predict(wav.to(device), "S2TT", TGT, src_lang=SRC)
# print(text[0])


import os, sys, json, torch, torchaudio
from pathlib import Path
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ThreadPoolExecutor

os.environ["FAIRSEQ2_ASSET_DIR"] = (
    "/Users/dattay/Documents/SpeechTranslation/checkpoints"
)
sys.path.insert(
    0, "/Users/dattay/Documents/SpeechTranslation/src/seamless_communication/src"
)

from seamless_communication.inference import Translator

# ── Config ────────────────────────────────────────────────────────────────────
JSONL_PATH = "datasets/valset_vietnamese.jsonl"  # ← đổi file tùy dataset
OUTPUT_PATH = "predictions.jsonl"
BATCH_SIZE = 8
NUM_WORKERS = 4
TARGET_SR = 16_000

LANG_MAP = {"English": "eng", "Vietnamese": "vie"}  # schema → seamless code


# ── Schema ────────────────────────────────────────────────────────────────────
@dataclass
class Sample:
    audio_filepath: str
    duration: float
    ori_text: str
    ori_lang: str  # "English" | "Vietnamese"
    tgt_text: str  # ground-truth (dùng cho eval)
    tgt_lang: str


# ── Dataset ───────────────────────────────────────────────────────────────────
class SpeechDataset(Dataset):
    def __init__(self, jsonl_path: str):
        self.samples: list[Sample] = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                self.samples.append(Sample(**d))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        wav, sr = torchaudio.load(s.audio_filepath)
        if sr != TARGET_SR:
            wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        return {
            "wav": wav.squeeze(0),  # (T,)
            "src_lang": LANG_MAP[s.ori_lang],
            "tgt_lang": LANG_MAP[s.tgt_lang],
            "reference": s.tgt_text,
            "ori_text": s.ori_text,
            "filepath": s.audio_filepath,
        }


def collate_fn(batch: list[dict]) -> dict:
    """Pad waveforms về cùng độ dài, giữ nguyên metadata."""
    max_len = max(b["wav"].shape[0] for b in batch)
    padded = torch.stack(
        [
            torch.nn.functional.pad(b["wav"], (0, max_len - b["wav"].shape[0]))
            for b in batch
        ]
    )
    return {
        "wav": padded,  # (B, T)
        "src_lang": [b["src_lang"] for b in batch],
        "tgt_lang": [b["tgt_lang"] for b in batch],
        "reference": [b["reference"] for b in batch],
        "ori_text": [b["ori_text"] for b in batch],
        "filepath": [b["filepath"] for b in batch],
    }


# ── Multi-GPU Translator ──────────────────────────────────────────────────────
class MultiGPUTranslator:
    def __init__(self, model_name: str, vocoder_name: str):
        n_gpus = torch.cuda.device_count()
        if n_gpus == 0:
            self.devices = [torch.device("cpu")]
            dtype = torch.float32
        else:
            self.devices = [torch.device(f"cuda:{i}") for i in range(n_gpus)]
            dtype = torch.float16

        with ThreadPoolExecutor(max_workers=len(self.devices)) as pool:
            self.translators = list(
                pool.map(
                    lambda d: Translator(model_name, vocoder_name, d, dtype=dtype),
                    self.devices,
                )
            )
        print(f"[init] {len(self.devices)} device(s): {self.devices}")

    def predict_batch(self, batch: dict, gpu_idx: int) -> list[dict]:
        translator = self.translators[gpu_idx]
        device = self.devices[gpu_idx]
        results = []

        for i, wav in enumerate(batch["wav"]):
            src_lang = batch["src_lang"][i]
            tgt_lang = batch["tgt_lang"][i]

            text, _ = translator.predict(
                wav.unsqueeze(0).to(device),
                "S2TT",
                tgt_lang,
                src_lang=src_lang,
            )
            results.append(
                {
                    "filepath": batch["filepath"][i],
                    "ori_text": batch["ori_text"][i],
                    "src_lang": src_lang,
                    "tgt_lang": tgt_lang,
                    "hypothesis": str(text[0]),
                    "reference": batch["reference"][i],
                }
            )
        return results


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    dataset = SpeechDataset(JSONL_PATH)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
    print(f"[dataset] {len(dataset)} samples | batch={BATCH_SIZE}")

    mt = MultiGPUTranslator("seamlessM4T_v2_large", "vocoder_v2")
    n = len(mt.devices)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as fout, ThreadPoolExecutor(
        max_workers=n
    ) as pool:

        futures = {
            pool.submit(mt.predict_batch, batch, idx % n): idx
            for idx, batch in enumerate(dataloader)
        }
        for future in futures:
            for row in future.result():
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(f"[{row['filepath'].split('/')[-1]}] {row['hypothesis']}")

    print(f"\n[done] saved → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
