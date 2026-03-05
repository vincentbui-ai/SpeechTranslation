# Copyright (c) 2025, Vincent Bui.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unicodedata
from dataclasses import dataclass
from typing import List, Optional

import sacrebleu


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_unicode(text: str) -> str:
    """NFC-normalize and strip leading/trailing whitespace."""
    return unicodedata.normalize("NFC", text).strip()


def _levenshtein(a: list, b: list) -> int:
    """Levenshtein edit distance between two token sequences."""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            prev, dp[j] = dp[j], (
                prev if a[i - 1] == b[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            )
    return dp[m]


# ---------------------------------------------------------------------------
# Sample-level functions
# ---------------------------------------------------------------------------

def compute_cer(reference: str, hypothesis: str, normalize: bool = True) -> float:
    """
    Compute Character Error Rate (CER) for a single pair.

    CER = edit_distance(ref_chars, hyp_chars) / len(ref_chars)

    Args:
        reference:  Ground-truth text.
        hypothesis: Model output text.
        normalize:  Apply Unicode NFC normalization before comparison.

    Returns:
        CER in [0, ∞).  Returns 0.0 when both strings are empty.

    Raises:
        ValueError: If reference is empty but hypothesis is not.
    """
    if normalize:
        reference = _normalize_unicode(reference)
        hypothesis = _normalize_unicode(hypothesis)

    ref_chars = list(reference)
    hyp_chars = list(hypothesis)

    if len(ref_chars) == 0:
        if len(hyp_chars) == 0:
            return 0.0
        raise ValueError("Reference is empty but hypothesis is not.")

    return _levenshtein(ref_chars, hyp_chars) / len(ref_chars)


def compute_wer(reference: str, hypothesis: str, normalize: bool = True) -> float:
    """
    Compute Word Error Rate (WER) for a single pair.

    WER = edit_distance(ref_words, hyp_words) / len(ref_words)

    Args:
        reference:  Ground-truth text.
        hypothesis: Model output text.
        normalize:  Apply Unicode NFC normalization before comparison.

    Returns:
        WER in [0, ∞).  Returns 0.0 when both strings are empty.

    Raises:
        ValueError: If reference is empty but hypothesis is not.
    """
    if normalize:
        reference = _normalize_unicode(reference)
        hypothesis = _normalize_unicode(hypothesis)

    ref_words = reference.split()
    hyp_words = hypothesis.split()

    if len(ref_words) == 0:
        if len(hyp_words) == 0:
            return 0.0
        raise ValueError("Reference is empty but hypothesis is not.")

    return _levenshtein(ref_words, hyp_words) / len(ref_words)


# ---------------------------------------------------------------------------
# Corpus-level result dataclass
# ---------------------------------------------------------------------------

@dataclass
class MetricsResult:
    """Aggregated evaluation metrics over a corpus."""

    cer: float
    """Macro-averaged Character Error Rate (%)."""

    wer: float
    """Macro-averaged Word Error Rate (%)."""

    bleu: float
    """Corpus-level BLEU score (sacrebleu)."""

    num_samples: int
    """Number of evaluated pairs."""

    def __str__(self) -> str:
        return (
            f"Samples : {self.num_samples}\n"
            f"CER     : {self.cer:.2f}%\n"
            f"WER     : {self.wer:.2f}%\n"
            f"BLEU    : {self.bleu:.2f}"
        )


# ---------------------------------------------------------------------------
# Corpus-level evaluator
# ---------------------------------------------------------------------------

class MetricsEvaluator:
    """
    Compute CER, WER and BLEU over a list of (reference, hypothesis) pairs.

    Usage::

        evaluator = MetricsEvaluator()
        result = evaluator.evaluate(references, hypotheses)
        print(result)
    """

    def __init__(self, normalize: bool = True, bleu_tokenize: str = "13a"):
        """
        Initialize evaluator.

        Args:
            normalize:      Apply Unicode NFC normalization to CER/WER inputs.
            bleu_tokenize:  sacrebleu tokenizer.  Use ``"char"`` for character-
                            level BLEU (useful for Vietnamese/Chinese output),
                            ``"13a"`` (default) for standard MT tokenization.
        """
        self.normalize = normalize
        self.bleu_tokenize = bleu_tokenize

    def evaluate(
        self,
        references: List[str],
        hypotheses: List[str],
        skip_empty: bool = True,
    ) -> MetricsResult:
        """
        Evaluate all metrics on the full corpus.

        Args:
            references:  List of ground-truth strings.
            hypotheses:  List of model output strings (same length).
            skip_empty:  Skip pairs where the reference is empty instead of
                         raising an error.

        Returns:
            :class:`MetricsResult` with aggregated CER, WER and BLEU.

        Raises:
            ValueError: If ``references`` and ``hypotheses`` differ in length.
        """
        if len(references) != len(hypotheses):
            raise ValueError(
                f"Length mismatch: {len(references)} references vs "
                f"{len(hypotheses)} hypotheses."
            )

        cer_scores: List[float] = []
        wer_scores: List[float] = []
        valid_refs: List[str] = []
        valid_hyps: List[str] = []

        for ref, hyp in zip(references, hypotheses):
            ref_norm = _normalize_unicode(ref) if self.normalize else ref
            if not ref_norm:
                if skip_empty:
                    continue
                raise ValueError(f"Empty reference encountered: {repr(ref)}")

            cer_scores.append(compute_cer(ref, hyp, normalize=self.normalize))
            wer_scores.append(compute_wer(ref, hyp, normalize=self.normalize))
            valid_refs.append(ref)
            valid_hyps.append(hyp)

        n = len(cer_scores)
        if n == 0:
            return MetricsResult(cer=0.0, wer=0.0, bleu=0.0, num_samples=0)

        avg_cer = sum(cer_scores) / n * 100
        avg_wer = sum(wer_scores) / n * 100

        bleu_result = sacrebleu.corpus_bleu(
            valid_hyps,
            [valid_refs],
            tokenize=self.bleu_tokenize,
        )

        return MetricsResult(
            cer=avg_cer,
            wer=avg_wer,
            bleu=bleu_result.score,
            num_samples=n,
        )

    def evaluate_single(self, reference: str, hypothesis: str) -> MetricsResult:
        """
        Convenience wrapper for a single (reference, hypothesis) pair.

        Args:
            reference:  Ground-truth text.
            hypothesis: Model output text.

        Returns:
            :class:`MetricsResult` for the single pair.
        """
        return self.evaluate([reference], [hypothesis])
