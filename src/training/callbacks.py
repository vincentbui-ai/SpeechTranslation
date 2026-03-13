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

import re
from collections import deque
from difflib import SequenceMatcher
from typing import Dict, List, Set


class EarlyStoppingChecker:
    """
    Smart early stopping checker for question generation with multiple criteria
    """

    def __init__(
        self,
        min_new_unique_threshold: int = 10,
        similarity_threshold: float = 0.85,
        plateau_window: int = 3,
        min_efficiency_threshold: float = 0.3,
    ):
        """
        Initialize early stopping checker

        Args:
            min_new_unique_threshold: Minimum new unique questions per batch
            similarity_threshold: Semantic similarity threshold (0-1) to consider duplicates
            plateau_window: Number of batches to check for plateau
            min_efficiency_threshold: Minimum efficiency (unique/total) to continue
        """
        self.min_new_unique_threshold = min_new_unique_threshold
        self.similarity_threshold = similarity_threshold
        self.plateau_window = plateau_window
        self.min_efficiency_threshold = min_efficiency_threshold

        # State tracking
        self.seen_questions: Set[str] = set()
        self.seen_normalized: Set[str] = set()
        self.seen_fingerprints: Set[str] = set()
        self.batch_history: deque = deque(maxlen=plateau_window)
        self.total_generated = 0
        self.total_unique = 0

    def normalize_question(self, question: str) -> str:
        """
        Normalize question for exact duplicate detection

        Args:
            question: Raw question text

        Returns:
            Normalized question string
        """
        # Lowercase and remove punctuation
        normalized = question.lower()
        normalized = re.sub(r"[?!.,;:\-\(\)\[\]\"\']+", "", normalized)
        # Remove extra whitespace
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    def get_question_fingerprint(self, question: str) -> str:
        """
        Generate semantic fingerprint for similarity detection
        Uses word order-independent representation

        Args:
            question: Raw question text

        Returns:
            Fingerprint string
        """
        # Normalize
        normalized = self.normalize_question(question)

        # Extract keywords (remove common Vietnamese question words)
        stop_words = {
            "là",
            "gì",
            "như",
            "thế",
            "nào",
            "có",
            "được",
            "không",
            "bao",
            "nhiêu",
            "ở",
            "đâu",
            "khi",
            "nào",
            "ai",
            "sao",
            "thì",
            "của",
            "cho",
            "với",
            "và",
            "hay",
            "hoặc",
            "nhưng",
            "mà",
            "để",
            "về",
            "từ",
            "trong",
            "ngoài",
            "trên",
            "dưới",
            "giữa",
            "cách",
            "làm",
            "hãy",
            "giúp",
            "tôi",
            "mình",
            "em",
            "anh",
            "chị",
        }

        words = normalized.split()
        keywords = [w for w in words if w not in stop_words and len(w) > 1]

        # Sort keywords for order-independent comparison
        fingerprint = " ".join(sorted(keywords))
        return fingerprint

    def is_similar_to_existing(self, question: str) -> bool:
        """
        Check if question is semantically similar to existing questions
        Uses multiple similarity metrics

        Args:
            question: Question to check

        Returns:
            True if similar to existing question
        """
        fingerprint = self.get_question_fingerprint(question)

        # Check fingerprint exact match
        if fingerprint in self.seen_fingerprints:
            return True

        # Check fuzzy similarity with recent questions (last 100 for performance)
        recent_questions = list(self.seen_questions)[-100:]
        for existing in recent_questions:
            # Use SequenceMatcher for fuzzy matching
            similarity = SequenceMatcher(
                None,
                self.normalize_question(question),
                self.normalize_question(existing),
            ).ratio()

            if similarity >= self.similarity_threshold:
                return True

        return False

    def add_question(self, question: str) -> bool:
        """
        Add question to tracking and check if it's unique

        Args:
            question: Question to add

        Returns:
            True if question is unique and added, False if duplicate
        """
        self.total_generated += 1

        # Check exact duplicate
        normalized = self.normalize_question(question)
        if normalized in self.seen_normalized:
            return False

        # Check semantic similarity
        if self.is_similar_to_existing(question):
            return False

        # Add to tracking
        self.seen_questions.add(question)
        self.seen_normalized.add(normalized)
        self.seen_fingerprints.add(self.get_question_fingerprint(question))
        self.total_unique += 1

        return True

    def add_batch_result(self, new_unique_count: int, batch_size: int) -> None:
        """
        Record batch generation result

        Args:
            new_unique_count: Number of new unique questions in batch
            batch_size: Total questions generated in batch
        """
        efficiency = new_unique_count / batch_size if batch_size > 0 else 0
        self.batch_history.append(
            {"new_unique": new_unique_count, "efficiency": efficiency}
        )

    def should_stop(self) -> tuple[bool, str]:
        """
        Check if generation should stop based on multiple criteria

        Returns:
            tuple: (should_stop: bool, reason: str)
        """
        if len(self.batch_history) == 0:
            return False, ""

        latest_batch = self.batch_history[-1]

        # Criterion 1: Too few new unique questions in latest batch
        if latest_batch["new_unique"] < self.min_new_unique_threshold:
            return (
                True,
                f"Low unique count: {latest_batch['new_unique']} < {self.min_new_unique_threshold}",
            )

        # Criterion 2: Efficiency too low
        if latest_batch["efficiency"] < self.min_efficiency_threshold:
            return (
                True,
                f"Low efficiency: {latest_batch['efficiency']:.1%} < {self.min_efficiency_threshold:.1%}",
            )

        # Criterion 3: Plateau detection (if we have enough history)
        if len(self.batch_history) >= self.plateau_window:
            # Check if unique count is consistently decreasing
            unique_counts = [b["new_unique"] for b in self.batch_history]
            is_decreasing = all(
                unique_counts[i] >= unique_counts[i + 1]
                for i in range(len(unique_counts) - 1)
            )

            # Check if average efficiency in window is low
            avg_efficiency = sum(b["efficiency"] for b in self.batch_history) / len(
                self.batch_history
            )

            if is_decreasing and avg_efficiency < self.min_efficiency_threshold * 1.5:
                return (
                    True,
                    f"Plateau detected: decreasing trend, avg efficiency {avg_efficiency:.1%}",
                )

        # Criterion 4: Very low recent efficiency (last 2 batches)
        if len(self.batch_history) >= 2:
            recent_efficiency = (
                sum(b["efficiency"] for b in list(self.batch_history)[-2:]) / 2
            )
            if recent_efficiency < self.min_efficiency_threshold * 0.5:
                return (
                    True,
                    f"Very low recent efficiency: {recent_efficiency:.1%}",
                )

        return False, ""

    def get_statistics(self) -> Dict[str, any]:
        """
        Get current statistics

        Returns:
            Dictionary with statistics
        """
        overall_efficiency = (
            self.total_unique / self.total_generated if self.total_generated > 0 else 0
        )

        return {
            "total_generated": self.total_generated,
            "total_unique": self.total_unique,
            "overall_efficiency": overall_efficiency,
            "batches_processed": len(self.batch_history),
            "recent_efficiency": (
                self.batch_history[-1]["efficiency"] if self.batch_history else 0
            ),
        }

    def reset(self) -> None:
        """Reset all tracking state"""
        self.seen_questions.clear()
        self.seen_normalized.clear()
        self.seen_fingerprints.clear()
        self.batch_history.clear()
        self.total_generated = 0
        self.total_unique = 0
