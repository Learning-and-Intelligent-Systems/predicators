"""Tests for the large language model interface."""

import os
import shutil

import pytest
from PIL import Image

from predicators import utils
from predicators.pretrained_model_interface import GoogleGeminiVLM, \
    LargeLanguageModel, OpenAILLM, VisionLanguageModel, OpenAIVLM


class _DummyLLM(LargeLanguageModel):

    def get_id(self):
        return "dummy"

    def _sample_completions(self,
                            prompt,
                            imgs,
                            temperature,
                            seed,
                            stop_token=None,
                            num_completions=1):
        del imgs  # unused.
        completions = []
        for _ in range(num_completions):
            completion = (f"Prompt: {prompt}. Seed: {seed}. "
                          f"Temp: {temperature:.1f}. Stop: {stop_token}.")
            completions.append(completion)
        return completions


class _DummyVLM(VisionLanguageModel):

    def get_id(self):
        return "dummy"

    def _sample_completions(self,
                            prompt,
                            imgs,
                            temperature,
                            seed,
                            stop_token=None,
                            num_completions=1):
        del imgs  # unused.
        completions = []
        for _ in range(num_completions):
            completion = (f"Prompt: {prompt}. Seed: {seed}. "
                          f"Temp: {temperature:.1f}. Stop: {stop_token}.")
            completions.append(completion)
        return completions


def test_large_language_model():
    """Tests for LargeLanguageModel()."""
    cache_dir = "_fake_llm_cache_dir"
    utils.reset_config({"pretrained_model_prompt_cache_dir": cache_dir})
    # Remove the fake cache dir in case it's lying around from old tests.
    shutil.rmtree(cache_dir, ignore_errors=True)
    # Query a dummy LLM.
    llm = _DummyLLM()
    assert llm.get_id() == "dummy"
    completions = llm.sample_completions("Hello!",
                                         None,
                                         0.5,
                                         123,
                                         stop_token="#stop",
                                         num_completions=3)
    expected_completion = "Prompt: Hello!. Seed: 123. Temp: 0.5. Stop: #stop."
    assert completions == [expected_completion] * 3
    # Query it again, covering the case where we load from disk.
    completions = llm.sample_completions("Hello!",
                                         None,
                                         0.5,
                                         123,
                                         stop_token="#stop",
                                         num_completions=3)
    assert completions == [expected_completion] * 3
    # Query with temperature 0.
    completions = llm.sample_completions("Hello!",
                                         None,
                                         0.0,
                                         123,
                                         num_completions=3)
    expected_completion = "Prompt: Hello!. Seed: 123. Temp: 0.0. Stop: None."
    assert completions == [expected_completion] * 3
    # Clean up the cache dir.
    shutil.rmtree(cache_dir)
    # Test llm_use_cache_only.
    utils.update_config({"llm_use_cache_only": True})
    with pytest.raises(ValueError) as e:
        completions = llm.sample_completions("Hello!",
                                             None,
                                             0.5,
                                             123,
                                             num_completions=3)
    assert "No cached response found for prompt." in str(e)


def test_vision_language_model():
    """Tests for LargeLanguageModel()."""
    cache_dir = "_fake_vlm_cache_dir"
    utils.reset_config({"pretrained_model_prompt_cache_dir": cache_dir})
    # Remove the fake cache dir in case it's lying around from old tests.
    shutil.rmtree(cache_dir, ignore_errors=True)
    # Query a dummy LLM.
    vlm = _DummyVLM()
    assert vlm.get_id() == "dummy"
    dummy_img = Image.new('RGB', (100, 100))
    completions = vlm.sample_completions("Hello!", [dummy_img],
                                         0.5,
                                         123,
                                         stop_token="#stop",
                                         num_completions=1)
    expected_completion = "Prompt: Hello!. Seed: 123. Temp: 0.5. Stop: #stop."
    assert completions == [expected_completion] * 1
    # Clean up the cache dir.
    shutil.rmtree(cache_dir)


def test_openai_llm():
    """Tests for OpenAILLM()."""
    cache_dir = "_fake_llm_cache_dir"
    utils.reset_config({"pretrained_model_prompt_cache_dir": cache_dir})
    if "OPENAI_API_KEY" not in os.environ:  # pragma: no cover
        os.environ["OPENAI_API_KEY"] = "dummy API key"
    # Create an OpenAILLM with the curie model.
    llm = OpenAILLM("text-curie-001")
    assert llm.get_id() == "openai-text-curie-001"
    # Uncomment this to test manually, but do NOT uncomment in master, because
    # each query costs money.
    # completions = llm.sample_completions("Hi", 0.5, 123, num_completions=2)
    # assert len(completions) == 2
    # completions2 = llm.sample_completions("Hi", 0.5, 123, num_completions=2)
    # assert completions == completions2
    # shutil.rmtree(cache_dir)


def test_gemini_vlm():
    """Tests for GoogleGeminiVLM()."""
    cache_dir = "_fake_llm_cache_dir"
    utils.reset_config({"pretrained_model_prompt_cache_dir": cache_dir})
    if "GOOGLE_API_KEY" not in os.environ:  # pragma: no cover
        os.environ["GOOGLE_API_KEY"] = "dummy API key"
    # Create an OpenAILLM with the curie model.
    vlm = GoogleGeminiVLM("gemini-pro-vision")
    assert vlm.get_id() == "Google-gemini-pro-vision"


def test_openai_vlm():
    """Tests for GoogleGeminiVLM()."""
    cache_dir = "_fake_llm_cache_dir"
    utils.reset_config({"pretrained_model_prompt_cache_dir": cache_dir})
    if "OPENAI_API_KEY" not in os.environ:  # pragma: no cover
        os.environ["OPENAI_API_KEY"] = "dummy API key"
    # Create an OpenAILLM with the curie model.
    vlm = OpenAIVLM("gpt-4-turbo")
    assert vlm.get_id() == "OpenAI-gpt-4-turbo"
