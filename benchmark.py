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

Input: JSON array of sentences
Output: JSON array of translated sentences (same order, same length)

Only output the JSON array, nothing else."""

        user_prompt = json.dumps(texts, ensure_ascii=False)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        last_error: Exception | None = None
        for attempt in range(max_retries):
            try:
                response = self._request(messages)
                
                try:
                    return json.loads(response)
                except json.JSONDecodeError:
                    # Attempt to repair JSON and retry parsing
                    repaired = repair_json(response)
                    return json.loads(repaired)
                    
            except (json.JSONDecodeError, requests.RequestException) as e:
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
    data_path = Path(data_dir) / "trans_testset_public"
    
    with open(data_path / "newstest2019-src.eng.txt", "r", encoding="utf-8") as f:
        eng_lines = [line.strip() for line in f.readlines()]
    
    with open(data_path / "newstest2019-ref.vie.txt", "r", encoding="utf-8") as f:
        vie_lines = [line.strip() for line in f.readlines()]
    
    return {"name": "newstest2019", "eng": eng_lines, "vie": vie_lines}


def load_tatoeba(data_dir: str) -> dict:
    """Load Tatoeba dataset with English-Vietnamese pairs."""
    data_path = Path(data_dir) / "trans_testset_public" / "tatoeba-test.eng-vie.tsv"
    
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


def normalize_text(text: str, remove_punct: bool = False, lowercase: bool = False) -> str:
    """Normalize text for evaluation."""
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
) -> list[str]:
    """Translate texts using batched concurrent requests."""
    results: list[str] = [""] * len(texts)
    batches = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batches.append((i, batch_texts))
    
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {}
        for start_idx, batch_texts in batches:
            future = executor.submit(translator.translate_batch, batch_texts)
            futures[future] = start_idx
        
        for future in tqdm(as_completed(futures), total=len(batches), desc="Translating"):
            start_idx = futures[future]
            try:
                translations = future.result()
                for j, trans in enumerate(translations):
                    results[start_idx + j] = trans
            except Exception as e:
                print(f"Error in batch starting at {start_idx}: {e}")
                # Results already initialized with empty strings
    
    return results


def evaluate_translations(
    references: list[str], 
    hypotheses: list[str],
    remove_punct: bool = False,
    lowercase: bool = False
) -> dict:
    """Evaluate translations with WER and BLEU."""
    # Normalize texts
    norm_refs = [normalize_text(r, remove_punct, lowercase) for r in tqdm(references, desc="Normalizing refs", leave=False)]
    norm_hyps = [normalize_text(h, remove_punct, lowercase) for h in tqdm(hypotheses, desc="Normalizing hyps", leave=False)]
    
    # Compute WER using jiwer
    wer = compute_wer(norm_refs, norm_hyps)
    
    # Compute BLEU using sacrebleu
    bleu = compute_bleu(norm_refs, norm_hyps)
    
    return {"wer": wer, "bleu": bleu}


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
                hypotheses = translate_texts(
                    source_texts, translator, 
                    cfg.batch_size, cfg.concurrency
                )
                
                # Evaluate with different normalization settings
                for norm_name, (remove_punct, lowercase) in [
                    ("original", (False, False)),
                    ("normalized", (True, True)),
                ]:
                    metrics = evaluate_translations(
                        reference_texts, hypotheses, 
                        remove_punct=remove_punct, 
                        lowercase=lowercase
                    )
                    
                    result = {
                        "dataset": dataset["name"],
                        "direction": direction,
                        "normalization": norm_name,
                        "wer": round(metrics["wer"], 2),
                        "bleu": round(metrics["bleu"], 2),
                    }
                    results.append(result)
                    
                    print(f"  [{norm_name}] WER: {result['wer']:.2f}%, BLEU: {result['bleu']:.2f}")
                
                pbar.update(1)
    
    # Save results
    output_path = cfg.get("output_path", "benchmark_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"Results saved to {output_path}")
    
    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Dataset':<15} {'Direction':<10} {'Norm':<12} {'WER':<10} {'BLEU':<10}")
    print("-"*80)
    for r in results:
        print(f"{r['dataset']:<15} {r['direction']:<10} {r['normalization']:<12} {r['wer']:<10.2f} {r['bleu']:<10.2f}")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    run_benchmark(cfg)


if __name__ == "__main__":
    main()
