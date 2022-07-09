"""Interface to pretrained large language models."""

import abc
import logging
import os

from predicators.src.settings import CFG

# This is a special string that we assume will never appear in a prompt, and
# which we use to separate prompt and completion in the cache. The reason to
# do it this way, rather than saving the prompt and responses separately,
# is that we want it to be easy to browse the cache as text files.
_CACHE_SEP = "\n####$$$###$$$####$$$$###$$$####$$$###$$$###\n"


class LargeLanguageModel(abc.ABC):
    """A pretrained large language model."""

    @abc.abstractmethod
    def get_id(self) -> str:
        """Get a string identifier for this LLM.

        This identifier should include sufficient information so that
        querying the same model with the same prompt and same identifier
        should yield the same result (assuming temperature 0).
        """
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def sample_completion(self,
                          prompt: str,
                          temperature: float,
                          seed: int,
                          num_completions: int = 1) -> str:
        """Sample one or more completions from a prompt.

        Higher temperatures will increase the variance in the responses.

        The seed may not be used, and the results may therefore not be
        reproducible, for LLMs where we only have access through an API that
        does not expose the ability to set a random seed.
        """
        raise NotImplementedError("Override me!")

    def get_most_likely_completion(self, prompt: str) -> str:
        """Get the most likely completion from a prompt.

        This is a separate method from sample_completion() because the
        result can be cached to disk, which is important when queries
        are expensive.
        """
        # Set up the cache file.
        assert _CACHE_SEP not in prompt
        os.makedirs(CFG.llm_prompt_cache_dir, exist_ok=True)
        llm_id = self.get_id()
        cache_filename = f"{llm_id}_{hash(prompt)}.txt"
        cache_filepath = os.path.join(CFG.llm_prompt_cache_dir, cache_filename)
        if not os.path.exists(cache_filepath):
            logging.debug(f"Querying LLM {llm_id} with new prompt.")
            # Query the LLM.
            completion = self.sample_completion(prompt,
                                                temperature=0,
                                                seed=CFG.seed,
                                                num_completions=1)
            # Cache the completion.
            cache_str = prompt + _CACHE_SEP + completion
            with open(cache_filepath, 'w', encoding='utf-8') as f:
                f.write(cache_str)
            logging.debug(f"Saved LLM response to {cache_filepath}.")
        # Load the saved completion.
        with open(cache_filepath, 'r', encoding='utf-8') as f:
            cache_str = f.read()
        logging.debug(f"Loaded LLM response from {cache_filepath}.")
        cached_prompt, completion = cache_str.split(_CACHE_SEP)
        assert cached_prompt == prompt
        return completion
