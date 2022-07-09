"""Tests for the large language model interface."""

import shutil

from predicators.src import utils
from predicators.src.llm_interface import LargeLanguageModel


class _DummyLLM(LargeLanguageModel):

    def get_id(self):
        return "dummy"

    def _sample_completions(self,
                            prompt,
                            temperature,
                            seed,
                            num_completions=1):
        completions = []
        for _ in range(num_completions):
            completion = (f"Prompt was: {prompt}. Seed: {seed}. "
                          f"Temp: {temperature:.1f}.")
            completions.append(completion)
        return completions


def test_large_language_model():
    """Tests for LargeLanguageModel()."""
    cache_dir = "_fake_llm_cache_dir"
    utils.reset_config({"llm_prompt_cache_dir": cache_dir})
    # Remove the fake cache dir in case it's lying around from old tests.
    shutil.rmtree(cache_dir, ignore_errors=True)
    # Query a dummy LLM.
    llm = _DummyLLM()
    assert llm.get_id() == "dummy"
    completions = llm.sample_completions("Hello world!", 0.5, 123, 3)
    expected_completion = "Prompt was: Hello world!. Seed: 123. Temp: 0.5."
    assert completions == [expected_completion] * 3
    # Clean up the cache dir.
    shutil.rmtree(cache_dir)
