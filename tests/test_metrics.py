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

import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.metrics import MetricsEvaluator, MetricsResult, compute_cer, compute_wer


class TestComputeCER(unittest.TestCase):

    def test_basic(self):
        self.assertAlmostEqual(compute_cer("hello", "hello"), 0.0)
        self.assertAlmostEqual(compute_cer("", ""), 0.0)
        self.assertAlmostEqual(compute_cer("cat", "bat"), 1 / 3)
        self.assertAlmostEqual(compute_cer("abc", "xyz"), 1.0)
        with self.assertRaises(ValueError):
            compute_cer("", "abc")

    def test_unicode_normalization(self):
        nfc = "\u1ed5"          # ổ precomposed (o + circumflex + hook above)
        nfd = "o\u0302\u0309"   # o + combining circumflex + combining hook above
        self.assertAlmostEqual(compute_cer(nfc, nfd, normalize=True), 0.0)
        self.assertGreater(compute_cer(nfc, nfd, normalize=False), 0.0)


class TestComputeWER(unittest.TestCase):

    def test_basic(self):
        self.assertAlmostEqual(compute_wer("the cat sat", "the cat sat"), 0.0)
        self.assertAlmostEqual(compute_wer("", ""), 0.0)
        self.assertAlmostEqual(compute_wer("the cat sat", "the dog sat"), 1 / 3)
        self.assertAlmostEqual(compute_wer("one two", "three four"), 1.0)
        with self.assertRaises(ValueError):
            compute_wer("", "word")

    def test_whitespace_normalization(self):
        self.assertAlmostEqual(
            compute_wer("  hello world  ", "hello world", normalize=True), 0.0
        )


class TestMetricsEvaluator(unittest.TestCase):

    def setUp(self):
        self.evaluator = MetricsEvaluator()

    def test_perfect_corpus(self):
        refs = ["the cat sat on the mat", "hello world"]
        hyps = ["the cat sat on the mat", "hello world"]
        result = self.evaluator.evaluate(refs, hyps)
        self.assertAlmostEqual(result.cer, 0.0)
        self.assertAlmostEqual(result.wer, 0.0)
        self.assertAlmostEqual(result.bleu, 100.0, delta=0.01)

    def test_skip_empty_and_mismatch(self):
        with self.assertRaises(ValueError):
            self.evaluator.evaluate(["a", "b"], ["a"])
        result = self.evaluator.evaluate(["hello", "", "world"], ["hello", "x", "world"], skip_empty=True)
        self.assertEqual(result.num_samples, 2)

    def test_result_str_format(self):
        result = MetricsResult(cer=8.42, wer=15.31, bleu=32.17, num_samples=1000)
        output = str(result)
        self.assertIn("8.42", output)
        self.assertIn("15.31", output)
        self.assertIn("32.17", output)
        self.assertIn("1000", output)


if __name__ == "__main__":
    unittest.main()
