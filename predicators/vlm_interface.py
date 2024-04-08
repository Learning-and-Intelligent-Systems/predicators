"""Interface to pretrained vision language models. Takes significant
inspiration from llm_interface.py.

NOTE: for now, we always assume that images will be appended to the end
of the text prompt. Interleaving text and images is currently not
supported, but should be doable in the future.
"""

import abc
import logging
import os
from typing import List, Optional

import google.generativeai as genai
import imagehash
import openai
import PIL.Image

from predicators.settings import CFG

# This is a special string that we assume will never appear in a prompt, and
# which we use to separate prompt and completion in the cache. The reason to
# do it this way, rather than saving the prompt and responses separately,
# is that we want it to be easy to browse the cache as text files.
_CACHE_SEP = "\n####$$$###$$$####$$$$###$$$####$$$###$$$###\n"


class VisionLanguageModel(abc.ABC):
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
    def _sample_completions(self,
                            prompt: str,
                            imgs: List[PIL.Image.Image],
                            temperature: float,
                            seed: int,
                            stop_token: Optional[str] = None,
                            num_completions: int = 1) -> List[str]:
        """This is the main method that subclasses must implement.

        This helper method is called by sample_completions(), which
        caches the prompts and responses to disk.
        """
        raise NotImplementedError("Override me!")

    def sample_completions(self,
                           prompt: str,
                           imgs: List[PIL.Image.Image],
                           temperature: float,
                           seed: int,
                           stop_token: Optional[str] = None,
                           num_completions: int = 1) -> List[str]:
        """Sample one or more completions from a prompt.

        Higher temperatures will increase the variance in the responses.

        The seed may not be used and the results may therefore not be
        reproducible for VLMs where we only have access through an API that
        does not expose the ability to set a random seed.

        Responses are saved to disk.
        """
        # Set up the cache file.
        assert _CACHE_SEP not in prompt
        os.makedirs(CFG.llm_prompt_cache_dir, exist_ok=True)
        vlm_id = self.get_id()
        prompt_id = hash(prompt)
        # We also need to hash all the images in the prompt.
        img_hash_list: List[str] = []
        for img in imgs:
            img_hash_list.append(str(imagehash.phash(img)))
        imgs_id = "".join(img_hash_list)
        # If the temperature is 0, the seed does not matter.
        if temperature == 0.0:
            config_id = f"most_likely_{num_completions}_{stop_token}"
        else:
            config_id = f"{temperature}_{seed}_{num_completions}_{stop_token}"
        cache_foldername = f"{vlm_id}_{config_id}_{prompt_id}_{imgs_id}"
        cache_folderpath = os.path.join(CFG.llm_prompt_cache_dir,
                                        cache_foldername)
        os.makedirs(cache_folderpath, exist_ok=True)
        cache_filename = "prompt.txt"
        cache_filepath = os.path.join(CFG.llm_prompt_cache_dir,
                                      cache_foldername, cache_filename)
        if not os.path.exists(cache_filepath):
            if CFG.llm_use_cache_only:
                raise ValueError("No cached response found for LLM prompt.")
            logging.debug(f"Querying VLM {vlm_id} with new prompt.")
            # Query the VLM.
            completions = self._sample_completions(prompt, imgs, temperature,
                                                   seed,
                                                   num_completions)
            # Cache the completion.
            cache_str = prompt + _CACHE_SEP + _CACHE_SEP.join(completions)
            with open(cache_filepath, 'w', encoding='utf-8') as f:
                f.write(cache_str)
            # Also save the images for easy debugging.
            imgs_folderpath = os.path.join(cache_folderpath, "imgs")
            os.makedirs(imgs_folderpath,exist_ok=True)
            for i, img in enumerate(imgs):
                filename_suffix = str(i) + ".jpg"
                img.save(os.path.join(imgs_folderpath, filename_suffix))
            logging.debug(f"Saved VLM response to {cache_filepath}.")
        # Load the saved completion.
        with open(cache_filepath, 'r', encoding='utf-8') as f:
            cache_str = f.read()
        logging.debug(f"Loaded VLM response from {cache_filepath}.")
        assert cache_str.count(_CACHE_SEP) == num_completions
        cached_prompt, completion_strs = cache_str.split(_CACHE_SEP, 1)
        assert cached_prompt == prompt
        completions = completion_strs.split(_CACHE_SEP)
        return completions


class GoogleGeminiVLM(VisionLanguageModel):
    """Interface to the Google Gemini VLM (1.5).

    Assumes that an environment variable ...
    """

    def __init__(self, model_name: str) -> None:
        """See https://ai.google.dev/models/gemini for the list of available
        model names."""
        self._model_name = model_name
        assert "GOOGLE_API_KEY" in os.environ
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self._model = genai.GenerativeModel(self._model_name)

    def get_id(self) -> str:
        return f"Google-{self._model_name}"

    def _sample_completions(self,
                            prompt: str,
                            imgs: List[PIL.Image.Image],
                            temperature: float,
                            seed: int,
                            num_completions: int = 1) -> List[str]:
        del seed  # unused
        generation_config = genai.types.GenerationConfig(
            candidate_count=num_completions,
            temperature=temperature)
        response = self._model.generate_content(
            [prompt] + imgs, generation_config=generation_config)
        response.resolve()
        return [response.text]
