"""Interface to pretrained large models.

These might be joint Vision-Language Models (VLM's) or Large Language
Models (LLM's)
"""

import abc
import base64
import logging
import os
from io import BytesIO
from typing import Collection, Dict, List, Optional, Union

import google.generativeai as genai
import imagehash
import numpy as np
import openai
import PIL.Image
from tenacity import retry, stop_after_attempt, wait_random_exponential

from predicators.settings import CFG

# This is a special string that we assume will never appear in a prompt, and
# which we use to separate prompt and completion in the cache. The reason to
# do it this way, rather than saving the prompt and responses separately,
# is that we want it to be easy to browse the cache as text files.
_CACHE_SEP = "\n####$$$###$$$####$$$$###$$$####$$$###$$$###\n"


class PretrainedLargeModel(abc.ABC):
    """A pretrained large vision or language model."""

    def __init__(self, system_instruction: Optional[str] = None):
        """Initialize the model with a system instruction."""
        self.system_instruction = system_instruction
        self.chat_history = []

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

    def sample_completions(
        self,
        prompt: str,
        imgs: Optional[List[PIL.Image.Image]],
        temperature: float,
        seed: int,
        stop_token: Optional[str] = None,
        num_completions: int = 1,
        cache_chat_session: bool = False,
    ) -> List[str]:
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
        prompt_id = hash(CFG.vlm_system_instruction + prompt)
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
            # NOTE: it's possible that this string (the concatenated hashes of
            # each image) is very long. This would make the final cache
            # foldername long. In many operating systems, the maximum folder
            # name length is 255 characters. To shorten this foldername more, we
            # can hash this string into a shorter string. For example, look at
            # https://stackoverflow.com/questions/57263436/hash-like-string-shortener-with-decoder  # pylint:disable=line-too-long
            imgs_id = hash("".join(img_hash_list))
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
            # cache_str = f"System instructions: {CFG.vlm_system_instruction}\n"+\
            cache_str = prompt + _CACHE_SEP + _CACHE_SEP.join(completions)
            with open(cache_filepath, 'w', encoding='utf-8') as f:
                f.write(cache_str)
            if imgs is not None:
                # Also save the images for easy debugging.
                imgs_folderpath = os.path.join(cache_folderpath, "imgs")
                os.makedirs(imgs_folderpath, exist_ok=True)
                for i, img in enumerate(imgs):
                    filename_suffix = str(i) + ".png"
                    # filename_suffix = str(i) + ".jpg"
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

        if cache_chat_session and self.get_id().startswith("OpenAI"):
            assert isinstance(self, OpenAIModel), "Have only implemented "\
                "OpenAIModel"
            system_message = self.prepare_system_message()
            messages = self.prepare_vision_messages(prefix=prompt,
                                                    images=imgs,
                                                    detail="auto")
            if not self.chat_history:
                # if chat history is empty, we need to include the system message
                messages = system_message + messages
            else:
                # if not, we include the chat history
                messages = self.chat_history + messages
            self.chat_history.extend(messages)
            # TODO: Cache the completion.
            self.chat_history.append({
                "role": "assistant",
                "content": completions
            })
        else:
            self.chat_history = []

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
        num_completions: int = 1,
        cache_chat_session: bool = False,
    ) -> List[str]:  # pragma: no cover
        assert imgs is not None
        return super().sample_completions(prompt, imgs, temperature, seed,
                                          stop_token, num_completions,
                                          cache_chat_session)


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


class OpenAIModel():
    """Common interface with methods for all OpenAI-based models."""

    def set_openai_key(self, key: Optional[str] = None) -> None:
        """Set the OpenAI API key."""
        if key is None:
            assert "OPENAI_API_KEY" in os.environ
            key = os.environ["OPENAI_API_KEY"]

    @retry(wait=wait_random_exponential(min=1, max=60),
           stop=stop_after_attempt(10))
    def call_openai_api(self,
                        messages: list,
                        model: str = "gpt-4",
                        seed: Optional[int] = None,
                        max_tokens: int = 32,
                        temperature: float = 0.2,
                        verbose: bool = False) -> str:  # pragma: no cover
        """Make an API call to OpenAI."""
        client = openai.OpenAI()
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            seed=seed,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if verbose:
            logging.debug(f"OpenAI API response: {completion}")
        assert len(completion.choices) == 1
        assert completion.choices[0].message.content is not None
        return completion.choices[0].message.content


class GoogleGeminiModel(PretrainedLargeModel):
    """Common interface and methods for all Gemini-based models.

    Assumes that an environment variable GOOGLE_API_KEY is set with the
    necessary API key to query the particular model name.
    """

    def __init__(self,
                 model_name: str,
                 system_instruction: Optional[str] = None) -> None:
        """See https://ai.google.dev/models/gemini for the list of available
        model names."""
        super().__init__(system_instruction)
        self._model_name = model_name
        assert "GOOGLE_API_KEY" in os.environ
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE"
            },
        ]
        # pylint:disable=no-member
        self._model = genai.GenerativeModel(
            model_name=self._model_name,
            safety_settings=self.safety_settings,
            system_instruction=self.system_instruction)
        if CFG.vlm_use_chat_mode:
            self.chat_session = self._model.start_chat()

        if CFG.vlm_model_name == "gemini-1.5-pro-exp-0801":
            keys = [
                os.getenv("GOOGLE_API_KEY"),
                os.getenv("GOOGLE_API_KEY1"),
                os.getenv("GOOGLE_API_KEY2"),
                os.getenv("GOOGLE_API_KEY3"),
                os.getenv("GOOGLE_API_KEY4"),
                os.getenv("GOOGLE_API_KEY5"),
            ]
        else:
            keys = [
                os.getenv("GOOGLE_API_KEY"),
            ]

        self.key_gen = self.key_generator(keys)

    def key_generator(self, keys):
        while True:
            for i, key in enumerate(keys):
                logging.debug(f"Using api key {i}")
                if key is not None:
                    yield key

    def get_new_key(self):
        return next(self.key_gen)

    def reset_chat_session(self) -> None:
        """Reset the chat session."""
        # make sure the instance has the chat_seesion attribute
        assert hasattr(self, "chat_session")
        self.chat_session = self._model.start_chat()


class OpenAILLM(LargeLanguageModel, OpenAIModel):
    """Interface to openAI LLMs.

    Assumes that an environment variable OPENAI_API_KEY is set to a
    private API key for beta.openai.com.
    """

    def __init__(self,
                 model_name: str,
                 system_instruction: Optional[str] = None) -> None:
        """See https://platform.openai.com/docs/models for the list of
        available model names."""
        super().__init__(system_instruction)
        self._model_name = model_name
        # Note that max_tokens is the maximum response length (not prompt).
        # From OpenAI docs: "The token count of your prompt plus max_tokens
        # cannot exceed the model's context length."
        self._max_tokens = CFG.vlm_openai_max_response_tokens
        self.set_openai_key()

    def get_id(self) -> str:
        return f"openai-{self._model_name}"

    def _sample_completions(
        self,
        prompt: str,
        imgs: Optional[List[PIL.Image.Image]],
        temperature: float,
        seed: int,
        stop_token: Optional[str] = None,
        num_completions: int = 1,
    ) -> List[str]:  # pragma: no cover
        del imgs, seed, stop_token  # unused
        messages = [{"role": "user", "content": prompt, "type": "text"}]
        responses = [
            self.call_openai_api(messages,
                                 model=self._model_name,
                                 temperature=temperature)
            for _ in range(num_completions)
        ]
        return responses


class GoogleGeminiLLM(LargeLanguageModel, GoogleGeminiModel):
    """Interface to the Google Gemini VLM (1.5).

    Assumes that an environment variable GOOGLE_API_KEY is set with the
    necessary API key to query the particular model name.
    """

    @retry(wait=wait_random_exponential(min=1, max=60),
           stop=stop_after_attempt(10))
    def _sample_completions(
        self,
        prompt: str,
        imgs: Optional[List[PIL.Image.Image]],
        temperature: float,
        seed: int,
        stop_token: Optional[str] = None,
        num_completions: int = 1,
    ) -> List[str]:  # pragma: no cover
        del seed, stop_token  # unused
        assert imgs is None
        generation_config = genai.types.GenerationConfig(  # pylint:disable=no-member
            candidate_count=num_completions,
            temperature=temperature)
        response = self._model.generate_content(
            [prompt], generation_config=generation_config)  # type: ignore
        response.resolve()
        return [response.text]

    def get_id(self) -> str:
        return f"Google-{self._model_name}"


class OpenAIVLM(VisionLanguageModel, OpenAIModel):
    """Interface for OpenAI's VLMs, including GPT-4 Turbo (and preview
    versions)."""

    def __init__(self,
                 model_name: str,
                 system_instruction: Optional[str] = None):
        """Initialize with a specific model name."""
        super().__init__(system_instruction)
        self.model_name = model_name
        # Note that max_tokens is the maximum response length (not prompt).
        # From OpenAI docs: "The token count of your prompt plus max_tokens
        # cannot exceed the model's context length."
        self._max_tokens = CFG.vlm_openai_max_response_tokens
        self.set_openai_key()

    def prepare_vision_messages(
        self,
        images: List[PIL.Image.Image],
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
        image_size: Optional[int] = 512,
        detail: str = "auto"
    ) -> List[Dict[str, Union[str, List[Dict[str, str]], List[Dict[
            str, Collection[str]]]]]]:
        """Prepare text and image messages for the OpenAI API."""
        content: List[Dict[str, Union[str, Collection[str]]]] = []
        if prefix:
            content.append({"text": prefix, "type": "text"})
        # assert images or self.chat_history
        assert detail in ["auto", "low", "high"]
        for img in images:
            img_resized = img
            if image_size:
                factor = image_size / max(img.size)
                img_resized = img.resize(
                    (int(img.size[0] * factor), int(img.size[1] * factor)))
            # Convert the image to PNG format and encode it in base64
            buffer = BytesIO()
            img_resized.save(buffer, format="PNG")
            buf = buffer.getvalue()
            frame = base64.b64encode(buf).decode("utf-8")
            content_str = {
                "image_url": {
                    "url": f"data:image/png;base64,{frame}",
                    "detail": "auto"
                },
                "type": "image_url"
            }
            content.append(content_str)
        if suffix:
            content.append({"text": suffix, "type": "text"})
        return [{"role": "user", "content": content}]

    def prepare_system_message(self):
        system_message = []
        if CFG.vlm_system_instruction and self.system_instruction is not None:
            system_message.append({
                "role":
                "system",
                "content": [{
                    "type": "text",
                    "text": self.system_instruction
                }]
            })
        return system_message

    def get_id(self) -> str:
        """Get an identifier for the model."""
        return f"OpenAI-{self.model_name}"

    def _sample_completions(
        self,
        prompt: str,
        imgs: Optional[List[PIL.Image.Image]],
        temperature: float,
        seed: int,
        stop_token: Optional[str] = None,
        num_completions: int = 1,
    ) -> List[str]:  # pragma: no cover
        """Query the model and get responses."""
        del seed, stop_token  # unused.

        # Allow images to be None if chat history is not empty.
        if imgs is None and not self.chat_history:
            raise ValueError("images cannot be None")
        system_message = self.prepare_system_message()
        messages = self.prepare_vision_messages(prefix=prompt,
                                                images=imgs,
                                                detail="auto")
        if not self.chat_history:
            # if chat history is empty, we need to include the system message
            messages = system_message + messages
        else:
            # if not, we include the chat history
            messages = self.chat_history + messages

        responses = [
            self.call_openai_api(messages,
                                 model=self.model_name,
                                 max_tokens=self._max_tokens,
                                 temperature=temperature)
            for _ in range(num_completions)
        ]
        return responses






class GoogleGeminiVLM(VisionLanguageModel, GoogleGeminiModel):
    """Interface to the Google Gemini VLM (1.5).

    Assumes that an environment variable GOOGLE_API_KEY is set with the
    necessary API key to query the particular model name.
    """

    @retry(wait=wait_random_exponential(min=1, max=60),
           stop=stop_after_attempt(60000))
    def _sample_completions(
        self,
        prompt: str,
        imgs: Optional[List[PIL.Image.Image]],
        temperature: float,
        seed: int,
        stop_token: Optional[str] = None,
        num_completions: int = 1,
    ) -> List[str]:  # pragma: no cover
        del seed, stop_token  # unused
        assert imgs is not None or self.chat_history
        genai.configure(api_key=self.get_new_key())
        # pylint:disable=no-member
        self._model = genai.GenerativeModel(
            model_name=self._model_name,
            safety_settings=self.safety_settings,
            system_instruction=self.system_instruction)

        if CFG.vlm_use_chat_mode:
            self.chat_session = self._model.start_chat(
                history=self.chat_history if self.chat_history else None)

        # pylint:disable=no-member
        generation_config = genai.types.GenerationConfig(
            candidate_count=num_completions, temperature=temperature)
        if CFG.vlm_use_chat_mode:
            logging.debug("Using chat mode instead of sample completions.")
            response = self.chat_session.send_message(
                [prompt] + imgs,
                generation_config=generation_config)  # type: ignore
        else:
            response = self._model.generate_content(
                [prompt] + imgs,
                generation_config=generation_config)  # type: ignore
        response.resolve()
        return [response.text]

    def get_id(self) -> str:
        return f"Google-{self._model_name}"
