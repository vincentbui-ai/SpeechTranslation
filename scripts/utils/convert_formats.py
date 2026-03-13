import glob
import itertools
import json
import os
import sys

import hydra
import torch
import torch.distributed as dist
import torchaudio
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm

# Add seamless_communication to path
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "seamless_communication", "src")
)

from seamless_communication.inference import Translator

TARGET_SR = 16_000
LANG_MAP = {
    "English":    "eng",
    "Vietnamese": "vie",
    "eng":        "eng",
    "vie":        "vie",
}


def load_dataset(path: str) -> list[dict]:
    if os.path.isdir(path):
        files = glob.glob(os.path.join(path, "**/*.jsonl"), recursive=True)
    else:
        files = [path]

    samples = []
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            for line in f:
                samples.append(json.loads(line))
    return samples


def load_audio(filepath: str) -> tuple[torch.Tensor, int]:
    filepath = filepath.replace("/raid/voice/khanhnd65/ultimate_testset/", "/data/")
    wav, sr = torchaudio.load(filepath)
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    return wav.squeeze(0), TARGET_SR


def pad_batch(wavs: list[torch.Tensor]) -> torch.Tensor:
    max_len = max(w.shape[0] for w in wavs)
    return torch.stack([
        torch.nn.functional.pad(w, (0, max_len - w.shape[0]))
        for w in wavs
    ])


@hydra.main(version_base=None, config_path="../config/test", config_name="s2tt")
def main(config: DictConfig) -> None:

    dist.init_process_group("nccl")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()

    device = rank % torch.cuda.device_count()
    dtype  = torch.float16

    translator = Translator(
        config.model_name,
        config.vocoder_name,
        torch.device(f"cuda:{device}"),
        dtype=dtype,
    )

    dataset = load_dataset(config.testset)

    indices = list(range(len(dataset)))
    indices = indices[rank::world_size]

    indices = [
        indices[i: i + config.batch_size]
        for i in range(0, len(indices), config.batch_size)
    ]
    indices = indices if rank != 0 else tqdm(indices, desc=f"[GPU {rank}] Inference", unit="batch")

    results = []
    for idx in indices:
        datas    = [dataset[i] for i in idx]
        paths    = [d["audio_filepath"] for d in datas]
        refs     = [d.get("tgt_text", "") for d in datas]
        ori_texts = [d.get("ori_text", "") for d in datas]
        src_langs = [LANG_MAP.get(d.get("ori_lang", "eng"), "eng") for d in datas]
        tgt_langs = [LANG_MAP.get(d.get("tgt_lang", "vie"), "vie") for d in datas]

        wavs = []
        valid_indices = []
        for i, path in enumerate(paths):
            try:
                wav, _ = load_audio(path)
                wavs.append(wav)
                valid_indices.append(i)
            except Exception:
                continue

        if not wavs:
            continue

        for i, wav in zip(valid_indices, wavs):
            try:
                wav_input = wav.unsqueeze(0).to(f"cuda:{device}", dtype=dtype)
                text, _ = translator.predict(
                    wav_input,
                    "S2TT",
                    tgt_langs[i],
                    src_lang=src_langs[i],
                )
                results.append((
                    paths[i],
                    ori_texts[i],
                    src_langs[i],
                    tgt_langs[i],
                    refs[i],
                    str(text[0]),
                ))
            except Exception:
                continue

    outputs = [None for _ in range(world_size)]
    dist.gather_object(results, outputs if rank == 0 else None)

    if rank == 0:
        outputs = list(itertools.chain(*outputs))

        os.makedirs(config.output_folder, exist_ok=True)
        output_filepath = os.path.join(config.output_folder, "results.csv")

        with open(output_filepath, "w", encoding="utf-8") as f:
            f.write("FILEPATH,ORI_TEXT,SRC_LANG,TGT_LANG,REFERENCE,HYPOTHESIS\n")
            for (filepath, ori_text, src_lang, tgt_lang, ref, hyp) in outputs:
                f.write(f"{filepath},{ori_text},{src_lang},{tgt_lang},{ref},{hyp}\n")

        print(f"\n[done] {len(outputs)} samples saved → {output_filepath}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
    