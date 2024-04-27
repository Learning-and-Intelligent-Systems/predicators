"""Interface to pretrained large models.

These might be joint Vision-Language Models (VLM's) or Large Language
Models (LLM's)
"""

import abc
import logging
import os
import time
from typing import List, Optional

import google
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


class PretrainedLargeModel(abc.ABC):
    """A pretrained large vision or language model."""

    @abc.abstractmethod
    def get_id(self) -> str:
        """Get a string identifier for this model.

        This identifier should include sufficient information so that
        querying the same model with the same prompt and same identifier
        should yield the same result (assuming temperature 0).
        """
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def _sample_completions(self,
                            prompt: str,
                            imgs: Optional[List[PIL.Image.Image]],
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
                           imgs: Optional[List[PIL.Image.Image]],
                           temperature: float,
                           seed: int,
                           stop_token: Optional[str] = None,
                           num_completions: int = 1) -> List[str]:
        """Sample one or more completions from a prompt.

        Higher temperatures will increase the variance in the responses.
        The seed may not be used and the results may therefore not be
        reproducible for models where we only have access through an API
        that does not expose the ability to set a random seed. Responses
        are saved to disk.
        """
        # Set up the cache file.
        assert _CACHE_SEP not in prompt
        os.makedirs(CFG.pretrained_model_prompt_cache_dir, exist_ok=True)
        model_id = self.get_id()
        prompt_id = hash(prompt)
        config_id = f"{temperature}_{seed}_{num_completions}_" + \
                f"{stop_token}"
        # If the temperature is 0, the seed does not matter.
        if temperature == 0.0:
            config_id = f"most_likely_{num_completions}_{stop_token}"
        cache_foldername = f"{model_id}_{config_id}_{prompt_id}"
        if imgs is not None:
            # We also need to hash all the images in the prompt.
            img_hash_list: List[str] = []
            for img in imgs:
                img_hash_list.append(str(imagehash.phash(img)))
            imgs_id = "".join(img_hash_list)
            cache_foldername += f"{imgs_id}"
        cache_folderpath = os.path.join(CFG.pretrained_model_prompt_cache_dir,
                                        cache_foldername)
        os.makedirs(cache_folderpath, exist_ok=True)
        cache_filename = "prompt.txt"
        cache_filepath = os.path.join(CFG.pretrained_model_prompt_cache_dir,
                                      cache_foldername, cache_filename)
        if not os.path.exists(cache_filepath):
            if CFG.llm_use_cache_only:
                raise ValueError("No cached response found for prompt.")
            logging.debug(f"Querying model {model_id} with new prompt.")
            # Query the model.
            completions = self._sample_completions(prompt, imgs, temperature,
                                                   seed, stop_token,
                                                   num_completions)
            # Cache the completion.
            cache_str = prompt + _CACHE_SEP + _CACHE_SEP.join(completions)
            with open(cache_filepath, 'w', encoding='utf-8') as f:
                f.write(cache_str)
            if imgs is not None:
                # Also save the images for easy debugging.
                imgs_folderpath = os.path.join(cache_folderpath, "imgs")
                os.makedirs(imgs_folderpath, exist_ok=True)
                for i, img in enumerate(imgs):
                    filename_suffix = str(i) + ".jpg"
                    img.save(os.path.join(imgs_folderpath, filename_suffix))
            logging.debug(f"Saved model response to {cache_filepath}.")
        # Load the saved completion.
        with open(cache_filepath, 'r', encoding='utf-8') as f:
            cache_str = f.read()
        logging.debug(f"Loaded model response from {cache_filepath}.")
        assert cache_str.count(_CACHE_SEP) == num_completions
        cached_prompt, completion_strs = cache_str.split(_CACHE_SEP, 1)
        assert cached_prompt == prompt
        completions = completion_strs.split(_CACHE_SEP)
        return completions


class VisionLanguageModel(PretrainedLargeModel):
    """A class for all VLM's."""

    def sample_completions(
            self,
            prompt: str,
            imgs: Optional[List[PIL.Image.Image]],
            temperature: float,
            seed: int,
            stop_token: Optional[str] = None,
            num_completions: int = 1) -> List[str]:  # pragma: no cover
        assert imgs is not None
        return super().sample_completions(prompt, imgs, temperature, seed,
                                          stop_token, num_completions)


class LargeLanguageModel(PretrainedLargeModel):
    """A class for all LLM's."""

    def sample_completions(
            self,
            prompt: str,
            imgs: Optional[List[PIL.Image.Image]],
            temperature: float,
            seed: int,
            stop_token: Optional[str] = None,
            num_completions: int = 1) -> List[str]:  # pragma: no cover
        assert imgs is None
        return super().sample_completions(prompt, imgs, temperature, seed,
                                          stop_token, num_completions)


class OpenAILLM(LargeLanguageModel):
    """Interface to openAI LLMs (GPT-3).

    Assumes that an environment variable OPENAI_API_KEY is set to a
    private API key for beta.openai.com.
    """

    def __init__(self, model_name: str) -> None:
        """See https://beta.openai.com/docs/models/gpt-3 for the list of
        available model names."""
        self._model_name = model_name
        # Note that max_tokens is the maximum response length (not prompt).
        # From OpenAI docs: "The token count of your prompt plus max_tokens
        # cannot exceed the model's context length."
        self._max_tokens = CFG.llm_openai_max_response_tokens
        assert "OPENAI_API_KEY" in os.environ
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def get_id(self) -> str:
        return f"openai-{self._model_name}"

    def _sample_completions(
            self,
            prompt: str,
            imgs: Optional[List[PIL.Image.Image]],
            temperature: float,
            seed: int,
            stop_token: Optional[str] = None,
            num_completions: int = 1) -> List[str]:  # pragma: no cover
        del imgs, seed  # unused
        response = openai.Completion.create(
            model=self._model_name,  # type: ignore
            prompt=prompt,
            temperature=temperature,
            max_tokens=self._max_tokens,
            stop=stop_token,
            n=num_completions)
        assert len(response["choices"]) == num_completions
        text_responses = [
            response["choices"][i]["text"] for i in range(num_completions)
        ]
        return text_responses


class GoogleGeminiVLM(VisionLanguageModel):
    """Interface to the Google Gemini VLM (1.5).

    Assumes that an environment variable GOOGLE_API_KEY is set with the
    necessary API key to query the particular model name.
    """

    def __init__(self, model_name: str) -> None:
        """See https://ai.google.dev/models/gemini for the list of available
        model names."""
        self._model_name = model_name
        assert "GOOGLE_API_KEY" in os.environ
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self._model = genai.GenerativeModel(self._model_name)  # pylint:disable=no-member

    def get_id(self) -> str:
        return f"Google-{self._model_name}"

    def _sample_completions(
            self,
            prompt: str,
            imgs: Optional[List[PIL.Image.Image]],
            temperature: float,
            seed: int,
            stop_token: Optional[str] = None,
            num_completions: int = 1) -> List[str]:  # pragma: no cover
        del seed, stop_token  # unused
        assert imgs is not None
        generation_config = genai.types.GenerationConfig(  # pylint:disable=no-member
            candidate_count=num_completions,
            temperature=temperature)
        response = None
        while response is None:
            try:
                response = self._model.generate_content(
                    [prompt] + imgs,
                    generation_config=generation_config)  # type: ignore
                break
            except google.api_core.exceptions.ResourceExhausted:
                # In this case, we've hit a rate limit. Simply wait 3s and
                # try again.
                logging.debug(
                    "Hit rate limit for Gemini queries; trying again in 3s!")
                time.sleep(3.0)
        response.resolve()
        return [response.text]
