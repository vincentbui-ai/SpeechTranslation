"""Benchmark translation quality with WER and BLEU metrics."""

import json
import re
import string
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import hydra
import jiwer
import requests
import sacrebleu
from omegaconf import DictConfig
from tqdm import tqdm

# Lock for thread-safe operations
file_lock = threading.Lock()

# Retry settings
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds


def repair_json(text: str) -> str:
    """Attempt to repair common JSON errors from LLM responses."""
    # Extract JSON array from response (remove any text before [ or after ])
    start = text.find('[')
    end = text.rfind(']') + 1
    if start != -1 and end > start:
        text = text[start:end]
    
    # Remove trailing commas before ]
    text = re.sub(r',\s*]', ']', text)
    
    # Fix invalid escape sequences (e.g., \x, \c -> \\x, \\c)
    # Valid JSON escapes: \", \\, \/, \b, \f, \n, \r, \t, \uXXXX
    text = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', text)
    
    return text


@dataclass
class LLMConfig:
    name: str
    url: str
    model: str
    host: str
    auth_token: str | None = None
    max_tokens: int = 4096
    temperature: float = 0
    reasoning_effort: str = "low"
    timeout: int = 60
    stop_tokens: list[str] = field(default_factory=lambda: [
        "<|end_of_text|>", "<|eot_id|>", "<|endoftext|>", "<|im_end|>"
    ])

    @classmethod
    def from_dict(cls, cfg: DictConfig) -> "LLMConfig":
        return cls(
            name=cfg.name, url=cfg.url, model=cfg.model, host=cfg.host,
            auth_token=cfg.get("auth_token"), max_tokens=cfg.max_tokens,
            temperature=cfg.temperature, reasoning_effort=cfg.reasoning_effort,
            timeout=cfg.get("timeout", 60), stop_tokens=list(cfg.stop_tokens)
        )


class Translator:
    def __init__(self, source_lang: str, target_lang: str, llm_cfg: DictConfig):
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.config = LLMConfig.from_dict(llm_cfg)
        self.headers = {"Content-Type": "application/json", "Host": self.config.host}
        if self.config.auth_token:
            self.headers["AuthToken"] = self.config.auth_token

    def _request(self, messages: list[dict]) -> str:
        payload = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "reasoning_effort": self.config.reasoning_effort,
            "ignore_eos": False,
            "stop": self.config.stop_tokens,
            "stream": False,
        }
        response = requests.post(
            self.config.url, headers=self.headers, 
            data=json.dumps(payload), timeout=self.config.timeout
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def translate_batch(self, texts: list[str], max_retries: int = MAX_RETRIES) -> list[str]:
        system_prompt = f"""You are a professional translator. Translate sentences from {self.source_lang} to {self.target_lang}.

Input: JSON array of sentences (one per line)
Output: JSON array of translated sentences (one per line, same order, same length)

Format output as:
[
"translated sentence 1",
"translated sentence 2",
...
]

Only output the JSON array, nothing else."""

        # Format input with one sentence per line for clarity
        formatted_texts = json.dumps(texts, ensure_ascii=False, indent=0).replace('[', '[\n').replace(']', '\n]')
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": formatted_texts}
        ]
        
        last_error: Exception | None = None
        for attempt in range(max_retries):
            try:
                response = self._request(messages)
                
                try:
                    result = json.loads(response)
                except json.JSONDecodeError:
                    # Attempt to repair JSON and retry parsing
                    repaired = repair_json(response)
                    result = json.loads(repaired)
                
                # Validate result length matches input
                if isinstance(result, list) and len(result) == len(texts):
                    return result
                    
                # If length mismatch, pad or truncate
                if isinstance(result, list):
                    if len(result) < len(texts):
                        result.extend([""] * (len(texts) - len(result)))
                    else:
                        result = result[:len(texts)]
                    return result
                
                raise ValueError(f"Expected list, got {type(result)}")
                    
            except (json.JSONDecodeError, requests.RequestException, ValueError) as e:
                last_error = e
                if attempt < max_retries - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))  # Exponential backoff
                    continue
        
        if last_error:
            raise last_error
        raise RuntimeError("Translation failed with no error captured")


# ==================== Dataset Loading ====================

def load_newstest2019(data_dir: str) -> dict:
    """Load newstest2019 dataset with English-Vietnamese pairs."""
    data_path = Path(data_dir)
    
    with open(data_path / "newstest2019-src.eng.txt", "r", encoding="utf-8") as f:
        eng_lines = [line.strip() for line in f.readlines()]
    
    with open(data_path / "newstest2019-ref.vie.txt", "r", encoding="utf-8") as f:
        vie_lines = [line.strip() for line in f.readlines()]
    
    return {"name": "newstest2019", "eng": eng_lines, "vie": vie_lines}


def load_tatoeba(data_dir: str) -> dict:
    """Load Tatoeba dataset with English-Vietnamese pairs."""
    data_path = Path(data_dir) / "tatoeba-test.eng-vie.tsv"
    
    eng_lines, vie_lines = [], []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                eng_lines.append(parts[2])
                vie_lines.append(parts[3])
    
    return {"name": "tatoeba", "eng": eng_lines, "vie": vie_lines}


# ==================== Text Normalization ====================

def remove_punctuation(text: str) -> str:
    """Remove punctuation but keep Vietnamese diacritics."""
    # Remove common punctuation marks
    punctuation = string.punctuation + "–—''""…«»"
    return text.translate(str.maketrans("", "", punctuation))


def normalize_text(text: str | None, remove_punct: bool = False, lowercase: bool = False) -> str:
    """Normalize text for evaluation."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    result = text.strip()
    if remove_punct:
        result = remove_punctuation(result)
    if lowercase:
        result = result.lower()
    # Normalize whitespace
    result = re.sub(r'\s+', ' ', result).strip()
    return result


# ==================== Metrics ====================

def compute_wer(references: list[str], hypotheses: list[str]) -> float:
    """Compute Word Error Rate using jiwer."""
    wer = jiwer.wer(references, hypotheses)
    return wer * 100  # Return as percentage


def compute_bleu(references: list[str], hypotheses: list[str]) -> float:
    """Compute corpus-level BLEU score using sacrebleu."""
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    return bleu.score


# ==================== Translation Pipeline ====================

def translate_texts(
    texts: list[str], 
    translator: Translator, 
    batch_size: int, 
    concurrency: int
) -> tuple[list[str | None], set[int]]:
    """Translate texts using batched concurrent requests.
    
    Returns:
        Tuple of (translations list, set of failed indices)
        Failed translations are marked as None
    """
    results: list[str | None] = [None] * len(texts)
    failed_indices: set[int] = set()
    batches = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batches.append((i, batch_texts))
    
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {}
        for start_idx, batch_texts in batches:
            future = executor.submit(translator.translate_batch, batch_texts)
            futures[future] = (start_idx, len(batch_texts))
        
        for future in tqdm(as_completed(futures), total=len(batches), desc="Translating"):
            start_idx, batch_len = futures[future]
            try:
                translations = future.result()
                for j, trans in enumerate(translations):
                    results[start_idx + j] = trans
            except Exception as e:
                print(f"Error in batch starting at {start_idx}: {e}")
                # Mark all indices in this batch as failed
                for j in range(batch_len):
                    failed_indices.add(start_idx + j)
    
    return results, failed_indices


def evaluate_translations(
    references: list[str], 
    hypotheses: list[str | None],
    failed_indices: set[int],
    remove_punct: bool = False,
    lowercase: bool = False
) -> dict:
    """Evaluate translations with WER and BLEU, excluding failed translations."""
    # Filter out failed indices
    valid_refs = []
    valid_hyps = []
    for i, (ref, hyp) in enumerate(zip(references, hypotheses)):
        if i not in failed_indices and hyp is not None:
            valid_refs.append(ref)
            valid_hyps.append(hyp)
    
    if not valid_refs:
        return {"wer": 100.0, "bleu": 0.0, "valid_count": 0, "total_count": len(references)}
    
    # Normalize texts
    norm_refs = [normalize_text(r, remove_punct, lowercase) for r in tqdm(valid_refs, desc="Normalizing refs", leave=False)]
    norm_hyps = [normalize_text(h, remove_punct, lowercase) for h in tqdm(valid_hyps, desc="Normalizing hyps", leave=False)]
    
    # Compute WER using jiwer
    wer = compute_wer(norm_refs, norm_hyps)
    
    # Compute BLEU using sacrebleu
    bleu = compute_bleu(norm_refs, norm_hyps)
    
    return {
        "wer": wer, 
        "bleu": bleu, 
        "valid_count": len(valid_refs), 
        "total_count": len(references)
    }


# ==================== Main Benchmark ====================

def run_benchmark(cfg: DictConfig):
    """Run benchmark on all datasets and directions."""
    data_dir = cfg.get("data_dir", "datasets")
    
    # Load datasets
    print("Loading datasets...")
    datasets = [
        load_newstest2019(data_dir),
        load_tatoeba(data_dir),
    ]
    
    results = []
    all_translations = {}  # Store all translation results
    
    # Calculate total tasks for progress
    directions = ["eng2vie", "vie2eng"]
    total_tasks = len(datasets) * len(directions)
    
    with tqdm(total=total_tasks, desc="Overall progress") as pbar:
        for dataset in datasets:
            print(f"\n{'='*60}")
            print(f"Dataset: {dataset['name']} ({len(dataset['eng'])} samples)")
            print(f"{'='*60}")
            
            for direction in directions:
                if direction == "eng2vie":
                    source_texts = dataset["eng"]
                    reference_texts = dataset["vie"]
                    source_lang, target_lang = "English", "Vietnamese"
                else:
                    source_texts = dataset["vie"]
                    reference_texts = dataset["eng"]
                    source_lang, target_lang = "Vietnamese", "English"
                
                pbar.set_postfix_str(f"{dataset['name']} {direction}")
                print(f"\nDirection: {direction}")
                print("-" * 40)
                
                # Create translator
                translator = Translator(source_lang, target_lang, cfg.llm)
                
                # Translate
                hypotheses, failed_indices = translate_texts(
                    source_texts, translator, 
                    cfg.batch_size, cfg.concurrency
                )
                
                # Store translation results
                translation_key = f"{dataset['name']}_{direction}"
                all_translations[translation_key] = {
                    "source_lang": source_lang,
                    "target_lang": target_lang,
                    "failed_count": len(failed_indices),
                    "total_count": len(source_texts),
                    "samples": [
                        {
                            "index": i,
                            "source": src,
                            "reference": ref,
                            "hypothesis": hyp,
                            "failed": i in failed_indices
                        }
                        for i, (src, ref, hyp) in enumerate(zip(source_texts, reference_texts, hypotheses))
                    ]
                }
                
                print(f"  Failed: {len(failed_indices)}/{len(source_texts)} samples")
                
                # Evaluate with different normalization settings
                for norm_name, (remove_punct, lowercase) in [
                    ("original", (False, False)),
                    ("normalized", (True, True)),
                ]:
                    metrics = evaluate_translations(
                        reference_texts, hypotheses, 
                        failed_indices=failed_indices,
                        remove_punct=remove_punct, 
                        lowercase=lowercase
                    )
                    
                    result = {
                        "dataset": dataset["name"],
                        "direction": direction,
                        "normalization": norm_name,
                        "wer": round(metrics["wer"], 2),
                        "bleu": round(metrics["bleu"], 2),
                        "valid_count": metrics["valid_count"],
                        "total_count": metrics["total_count"],
                    }
                    results.append(result)
                    
                    print(f"  [{norm_name}] WER: {result['wer']:.2f}%, BLEU: {result['bleu']:.2f} ({result['valid_count']}/{result['total_count']} valid)")
                
                pbar.update(1)
    
    # Save benchmark results
    output_path = cfg.get("output_path", "benchmark_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Save translation outputs
    translations_path = cfg.get("translations_path", "translation_outputs.json")
    with open(translations_path, "w", encoding="utf-8") as f:
        json.dump(all_translations, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"Results saved to {output_path}")
    print(f"Translations saved to {translations_path}")
    
    # Print summary table
    print("\n" + "="*90)
    print("SUMMARY")
    print("="*90)
    print(f"{'Dataset':<15} {'Direction':<10} {'Norm':<12} {'WER':<10} {'BLEU':<10} {'Valid':<15}")
    print("-"*90)
    for r in results:
        valid_str = f"{r['valid_count']}/{r['total_count']}"
        print(f"{r['dataset']:<15} {r['direction']:<10} {r['normalization']:<12} {r['wer']:<10.2f} {r['bleu']:<10.2f} {valid_str:<15}")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    run_benchmark(cfg)


if __name__ == "__main__":
    main()
