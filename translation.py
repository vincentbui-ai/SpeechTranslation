"""Translation data generation module."""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import hydra
import requests
from omegaconf import DictConfig
from tqdm import tqdm

from data_loader import load_samples, load_translated, get_untranslated, save_jsonl


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
    stop_tokens: list[str] = field(default_factory=lambda: [
        "<|end_of_text|>", "<|eot_id|>", "<|endoftext|>", "<|im_end|>"
    ])

    @classmethod
    def from_dict(cls, cfg: DictConfig) -> "LLMConfig":
        return cls(
            name=cfg.name, url=cfg.url, model=cfg.model, host=cfg.host,
            auth_token=cfg.get("auth_token"), max_tokens=cfg.max_tokens,
            temperature=cfg.temperature, reasoning_effort=cfg.reasoning_effort,
            stop_tokens=list(cfg.stop_tokens)
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
        response = requests.post(self.config.url, headers=self.headers, data=json.dumps(payload))
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def translate_batch(self, texts: list[str]) -> list[str]:
        system_prompt = f"""You are a professional translator. Translate sentences from {self.source_lang} to {self.target_lang}.

Input: JSON array of sentences
Output: JSON array of translated sentences (same order, same length)

Only output the JSON array, nothing else."""

        user_prompt = json.dumps(texts, ensure_ascii=False)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        response = self._request(messages)
        return json.loads(response)

    def translate_samples(self, samples: list[dict]) -> list[dict]:
        texts = [s["text"] for s in samples]
        translations = self.translate_batch(texts)
        return [
            {
                "audio_filepath": s["audio_filepath"],
                "duration": s.get("duration"),
                "ori_text": s["text"],
                "ori_lang": self.source_lang,
                "tgt_lang": self.target_lang,
                "tgt_text": t
            }
            for s, t in zip(samples, translations)
        ]


def run_translation(cfg: DictConfig):
    samples = load_samples(list(cfg.input_paths))
    translated = load_translated(cfg.translated_path) if cfg.translated_path else []
    untranslated = get_untranslated(samples, translated)
    
    print(f"Total: {len(samples)}, Translated: {len(translated)}, Remaining: {len(untranslated)}")
    
    if not untranslated:
        save_jsonl(translated, cfg.output_path)
        return
    
    translator = Translator(cfg.source_lang, cfg.target_lang, cfg.llm)
    results = list(translated)
    
    batches = [untranslated[i:i + cfg.batch_size] for i in range(0, len(untranslated), cfg.batch_size)]
    
    with ThreadPoolExecutor(max_workers=cfg.concurrency) as executor:
        futures = {executor.submit(translator.translate_samples, batch): batch for batch in batches}
        for future in tqdm(as_completed(futures), total=len(batches), desc="Translating"):
            results.extend(future.result())
            save_jsonl(results, cfg.output_path)
    
    print(f"Done! Saved {len(results)} samples to {cfg.output_path}")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    run_translation(cfg)


if __name__ == "__main__":
    main()
