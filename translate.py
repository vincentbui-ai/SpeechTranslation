import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from src.llm import GeminiLLM

# Config
MODEL_NAME = "gemini-2.0-flash"
TEMPERATURE = 0
MAX_OUTPUT_TOKENS = 2048
TOP_P = 0.95
TOP_K = 20
BATCH_SIZE = 8
THREADS = 64


def process_batch(batch_data, model: GeminiLLM):
    """Translates a batch of records using ori_lang -> tgt_lang direction."""
    ori_lang = batch_data[0]["ori_lang"]
    tgt_lang = batch_data[0]["tgt_lang"]
    transcripts = [item["ori_text"] for item in batch_data]

    prompt = (
        f"Translate these {ori_lang} lines to {tgt_lang}. "
        f"Follow these transcription rules strictly in the output:\n"
        f"1. Numbers must be spelled out in words, not written as numerals "
        f"(e.g. '2017' -> 'two thousand and seventeen').\n"
        f"2. Acronyms must be written as they are normally written in {tgt_lang}, "
        f"following standard capitalization rules. Do NOT transcribe them phonetically "
        f"(e.g. 'F B I' -> 'FBI').\n"
        f"3. Use punctuation appropriate for written {tgt_lang}. "
        f"Capitalize the beginning of new sentences.\n"
        f"4. Currency symbols, percentages and other symbols must be spelled out "
        f"(e.g. '$10' -> 'ten dollars', '5%' -> 'five percent').\n"
        f"Return ONLY a JSON list of strings with the same number of elements as the input: "
        f"{json.dumps(transcripts, ensure_ascii=False)}"
    )

    try:
        response = model.generate_content(prompt)
        raw_content = response.text.strip()
        clean_json = raw_content.replace("```json", "").replace("```", "").strip()
        translated_list = json.loads(clean_json)

        for i, item in enumerate(batch_data):
            item["tgt_text"] = (
                translated_list[i] if i < len(translated_list) else "ERROR: Missing"
            )

    except Exception as e:
        for item in batch_data:
            item["tgt_text"] = f"BATCH_ERROR: {str(e)}"

    return batch_data


def main(input_file):
    model = GeminiLLM(
        model_name=MODEL_NAME,
        temperature=TEMPERATURE,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        top_p=TOP_P,
        top_k=TOP_K,
    )

    with open(input_file, "r", encoding="utf-8") as f:
        all_data = [json.loads(line) for line in f if line.strip()]

    # Skip already translated records
    pending = [item for item in all_data if not item.get("tgt_text")]
    already_done = len(all_data) - len(pending)
    if already_done:
        print(f"Skipping {already_done} already translated records.")

    batches = [pending[i : i + BATCH_SIZE] for i in range(0, len(pending), BATCH_SIZE)]

    with ThreadPoolExecutor(max_workers=THREADS) as executor:
        list(
            tqdm(
                executor.map(lambda batch: process_batch(batch, model), batches),
                total=len(batches),
                desc="Translating",
            )
        )

    # Save back to the same file
    with open(input_file, "w", encoding="utf-8") as f_out:
        for record in all_data:
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Done! Translated {len(pending)} records -> saved to {input_file}")


if __name__ == "__main__":
    FILES = ["datasets/trainset_english.jsonl"]
    for file in FILES:
        main(file)
