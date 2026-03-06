"""
train_spm.py
============
Train SentencePiece unigram models cho VI và EN, tạo fairseq dictionary,
và sinh tokenized TSV files cho multitask decoder heads.

Chạy LOCAL sau khi đã chạy prepare_data.py.

Pipeline này thực hiện đúng theo
  src/StreamSpeech/preprocess_scripts/prep_cvss_c_multitask_data.py
nhưng không phụ thuộc vào fairseq PYTHONPATH.

Input  (sinh ra bởi prepare_data.py):
    data/vi-en/train.tsv      — tgt_text = EN translation
    data/vi-en/dev.tsv
    data/vi-en/train_asr.tsv  — tgt_text = VI transcript
    data/vi-en/dev_asr.tsv

Output:
    configs/vi-en/tgt_unigram6000/
        spm_unigram_en.model     ← SPM model tiếng Anh
        spm_unigram_en.vocab     ← SPM vocab
        spm_unigram_en.txt       ← fairseq dict (để set trong config YAML)
        train.tsv                ← id + SPM-tokenized EN text  ← dùng bởi target_unigram & ctc_target_unigram
        dev.tsv

    configs/vi-en/src_unigram6000/
        spm_unigram_vi.model     ← SPM model tiếng Việt
        spm_unigram_vi.vocab
        spm_unigram_vi.txt       ← fairseq dict
        train.tsv                ← id + SPM-tokenized VI text  ← dùng bởi source_unigram (ASR)
        dev.tsv

Usage:
    python scripts/train_spm.py --data-dir data/vi-en --configs-dir configs/vi-en
"""

import argparse
import csv
import re
from pathlib import Path

try:
    import sentencepiece as spm
except ImportError:
    raise ImportError("Cài sentencepiece trước: pip install sentencepiece")


VOCAB_SIZE    = 6000
MODEL_TYPE    = "unigram"
# character_coverage=1.0 bắt buộc cho tiếng Việt (có dấu thanh điệu)
CHARACTER_COVERAGE = 1.0
SPLITS = ["train", "dev"]


def normalize_text(text: str) -> str:
    """
    Chuẩn hóa text trước khi train/apply SPM.
    Giống với prep_cvss_c_multitask_data.py gốc:
      - lowercase
      - xóa dấu câu (giữ ký tự word Unicode, giữ khoảng trắng)
    \w trong Python 3 match Unicode nên giữ được dấu tiếng Việt.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def read_tsv_column(tsv_path: Path, column: str) -> list[dict]:
    """Đọc toàn bộ TSV, trả về list of {'id': ..., 'text': ...}."""
    rows = []
    with open(tsv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append({"id": row["id"], "text": row.get(column, "")})
    return rows


def train_spm_model(texts: list[str], model_prefix: Path, lang_tag: str):
    """Train SentencePiece model từ list of texts."""
    model_prefix.parent.mkdir(parents=True, exist_ok=True)
    import os
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False,
                                     encoding="utf-8") as f:
        tmp_path = f.name
        for t in texts:
            f.write(normalize_text(t) + "\n")
    try:
        print(f"  Training SPM ({lang_tag}, vocab={VOCAB_SIZE}, {len(texts):,} sentences) ...")
        spm.SentencePieceTrainer.train(
            input=tmp_path,
            model_prefix=str(model_prefix),
            model_type=MODEL_TYPE,
            vocab_size=VOCAB_SIZE,
            character_coverage=CHARACTER_COVERAGE,
            pad_id=1, unk_id=3, bos_id=0, eos_id=2,
            input_sentence_size=5_000_000,
            shuffle_input_sentence=True,
        )
    finally:
        os.unlink(tmp_path)
    print(f"  Saved: {model_prefix}.model, {model_prefix}.vocab")


def spm_vocab_to_fairseq_dict(vocab_path: Path, dict_path: Path):
    """Chuyển .vocab → fairseq dict.txt (token <space> count)."""
    SPECIAL = {"<unk>", "<s>", "</s>", "<pad>"}
    with open(vocab_path, encoding="utf-8") as f_in, \
         open(dict_path,  "w", encoding="utf-8") as f_out:
        count = 0
        for line in f_in:
            token = line.strip().split("\t")[0]
            if token not in SPECIAL:
                f_out.write(f"{token} 1\n")
                count += 1
    print(f"  Fairseq dict ({count:,} tokens) -> {dict_path}")


def apply_spm_and_write_tsv(
    rows: list[dict],
    sp_model: spm.SentencePieceProcessor,
    out_tsv: Path,
):
    """
    Apply SPM tokenization lên list of rows, ghi TSV chỉ có 2 cột: id + tgt_text.
    Đây là format mà fairseq multitask config đọc qua `data:` field.
    """
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_tsv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["id", "tgt_text"])
        for row in rows:
            normalized = normalize_text(row["text"])
            tokenized  = " ".join(sp_model.encode(normalized, out_type=str))
            writer.writerow([row["id"], tokenized])
    print(f"  Tokenized TSV ({len(rows):,} rows) -> {out_tsv}")


def process_one_head(
    data_dir: Path,
    configs_subdir: Path,
    prefix_name: str,
    lang_tag: str,
    tsv_basename: str,
):
    """
    Xử lý một decoder head:
      1. Train SPM từ tập train
      2. Tạo fairseq dict
      3. Apply SPM lên train + dev → lưu TSV tokenized
    """
    print(f"\n{'='*60}")
    print(f"[{lang_tag}] → {configs_subdir.name}/")
    print(f"{'='*60}")

    # Đọc train text
    train_tsv = data_dir / f"{tsv_basename}.tsv"
    if not train_tsv.exists():
        print(f"  [ERROR] Not found: {train_tsv}")
        return

    train_rows = read_tsv_column(train_tsv, "tgt_text")
    train_texts = [r["text"] for r in train_rows]

    # 1. Train SPM
    model_prefix = configs_subdir / prefix_name
    train_spm_model(train_texts, model_prefix, lang_tag)

    # 2. Fairseq dict
    vocab_path = Path(str(model_prefix) + ".vocab")
    dict_path  = Path(str(model_prefix) + ".txt")
    spm_vocab_to_fairseq_dict(vocab_path, dict_path)

    # 3. Load SPM và tokenize tất cả splits
    sp = spm.SentencePieceProcessor(model_file=str(model_prefix) + ".model")

    for split in SPLITS:
        split_tsv = data_dir / f"{tsv_basename.replace('train', split)}.tsv"
        if split == "train":
            split_tsv = data_dir / f"{tsv_basename}.tsv"
        else:
            # dev split: thay "train" → "dev" trong basename
            split_tsv = data_dir / f"{tsv_basename.replace('train', split)}.tsv"

        if not split_tsv.exists():
            print(f"  [SKIP] {split_tsv} not found")
            continue

        rows = read_tsv_column(split_tsv, "tgt_text")
        out_tsv = configs_subdir / f"{split}.tsv"
        apply_spm_and_write_tsv(rows, sp, out_tsv)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default="data/vi-en",
        help="Thư mục chứa train.tsv, dev.tsv, train_asr.tsv, dev_asr.tsv"
    )
    parser.add_argument(
        "--configs-dir",
        default="configs/vi-en",
        help="Thư mục output cho configs (mặc định: configs/vi-en/)"
    )
    args = parser.parse_args()

    data_dir    = Path(args.data_dir)
    configs_dir = Path(args.configs_dir)

    # --- EN translation head (target_unigram + ctc_target_unigram) ---
    # Đọc từ train.tsv (tgt_text = EN translation)
    process_one_head(
        data_dir      = data_dir,
        configs_subdir= configs_dir / "tgt_unigram6000",
        prefix_name   = "spm_unigram_en",
        lang_tag      = "English (translation target)",
        tsv_basename  = "train",
    )

    # --- VI ASR head (source_unigram) ---
    # Đọc từ train_asr.tsv (tgt_text = VI transcript)
    process_one_head(
        data_dir      = data_dir,
        configs_subdir= configs_dir / "src_unigram6000",
        prefix_name   = "spm_unigram_vi",
        lang_tag      = "Vietnamese (ASR source)",
        tsv_basename  = "train_asr",
    )

    print(f"\n{'='*60}")
    print("[DONE] SentencePiece training complete.")
    print(f"  Models & dicts : {configs_dir}/{{src,tgt}}_unigram6000/")
    print(f"  Tokenized TSVs : {configs_dir}/{{src,tgt}}_unigram6000/{{train,dev}}.tsv")
    print(f"\nKiểm tra kết quả:")
    print(f"  head -3 {configs_dir}/tgt_unigram6000/train.tsv")
    print(f"  head -3 {configs_dir}/src_unigram6000/train.tsv")


if __name__ == "__main__":
    main()
