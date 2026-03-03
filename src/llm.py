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

import os

import google.generativeai as genai
from dotenv import load_dotenv


class GeminiLLM:
    """
    Wrapper class for Gemini LLM model
    """

    def __init__(
        self,
        model_name: str = "gemini-2.0-flash",
        temperature: float = 0.2,
        max_output_tokens: int = 4096,
        top_p: float = 0.95,
        top_k: int = 20,
    ):
        """
        Initialize and configure Gemini LLM model

        Args:
            model_name: Name of the Gemini model to use
            temperature: Controls randomness (0.0-1.0)
            max_output_tokens: Maximum tokens in response
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter

        Raises:
            RuntimeError: If API key is not found
        """
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Missing GEMINI_API_KEY or GOOGLE_API_KEY environment variable."
            )

        genai.configure(api_key=api_key)

        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
                "top_p": top_p,
                "top_k": top_k,
            },
        )

    def generate_content(self, prompt: str) -> genai.types.GenerateContentResponse:
        """
        Generate content from prompt

        Args:
            prompt: Input prompt text

        Returns:
            Response from the model
        """
        return self.model.generate_content(prompt)
