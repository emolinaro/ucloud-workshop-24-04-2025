import logging
import json
import difflib
import re
import os
import requests
import pytesseract

from PIL import Image, ImageOps
from io import BytesIO
from typing import Union, List, Dict, Optional, Any, Tuple
from tenacity import retry, stop_after_attempt, wait_random, wait_exponential
from openai import OpenAI

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from label_studio_sdk.label_interface.objects import PredictionValue
from label_studio_sdk.label_interface.object_tags import ImageTag, ParagraphsTag
from label_studio_sdk.label_interface.control_tags import ControlTag, ObjectTag

logger = logging.getLogger(__name__)

@retry(wait=wait_exponential(multiplier=1, min=1, max=10), stop=stop_after_attempt(5))
def chat_completion_call(messages, params, *args, **kwargs):
    """
    Request to OpenAI API (NVIDIA NIM, OpenAI)

    Args:
        messages: list of messages
        params: dict with parameters
           Example:
               ```json
              {
                "api_key": "YOUR_API_KEY",
                "provider": "openai",
                "model": "gpt-4",
                "num_responses": 1,
                "temperature": 0.7
                }```
    """
    provider = params.get("provider", OpenAIInteractive.OPENAI_PROVIDER)
    model = params.get("model", OpenAIInteractive.OPENAI_MODEL)
    if provider == "openai":
        client = OpenAI(
            api_key=params.get("api_key", OpenAIInteractive.O_KEY),
        )
        if not model:
            model = 'gpt-3.5-turbo'
    elif provider == "nvidia":
        client = OpenAI(
            api_key=params.get("api_key", OpenAIInteractive.NVIDIA_NIM_KEY),
            base_url=params.get('base_url', OpenAIInteractive.NVIDIA_NIM_ENDPOINT).rstrip('/')
        )
        if not model:
            model = 'nvidia/nemotron-4-340b-instruct'
    elif provider == "ollama":
        client = OpenAI(
            base_url=params.get('base_url', OpenAIInteractive.OLLAMA_ENDPOINT),
            # required but ignored
            api_key='ollama',
        )
    else:
        raise

    request_params = {
        "messages": messages,
        "model": model,
        "n": params.get("num_responses", OpenAIInteractive.NUM_RESPONSES),
        "temperature": params.get("temperature", OpenAIInteractive.TEMPERATURE),
        "max_tokens": params.get("max_tokens", OpenAIInteractive.MAX_TOKENS)
    }

    completion = client.chat.completions.create(**request_params, timeout=30)

    return completion


def gpt(messages: Union[List[Dict], str], params, *args, **kwargs):
    """
    Calls the LLM and extracts the generated assistant response.
    It looks for the last occurrence of the word 'assistant' and returns everything after that.
    """
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]

    logger.debug(f"OpenAI request: {messages}, params={params}")
    completion = chat_completion_call(messages, params)
    logger.debug(f"OpenAI response: {completion}")

    raw_responses = [choice.message.content for choice in completion.choices]
    trimmed_responses = []

    for resp in raw_responses:
        # Find the last occurrence of 'assistant' as a whole word, possibly surrounded by whitespace or newlines
        match = list(re.finditer(r'\bassistant\b', resp, flags=re.IGNORECASE))
        if match:
            last_pos = match[-1].end()
            trimmed = resp[last_pos:].strip()
        else:
            trimmed = resp.strip()

        trimmed_responses.append(trimmed)

    return trimmed_responses


class OpenAIInteractive(LabelStudioMLBase):
    """
    """
    PROMPT_PREFIX = os.getenv("PROMPT_PREFIX", "prompt")
    USE_INTERNAL_PROMPT_TEMPLATE = bool(int(os.getenv("USE_INTERNAL_PROMPT_TEMPLATE", 1)))
    # if set, this prompt will be used at the beginning of the session
    DEFAULT_PROMPT = os.getenv('DEFAULT_PROMPT')
    PROMPT_TEMPLATE = os.getenv("PROMPT_TEMPLATE", '**Source Text**:\n\n"{text}"\n\n**Task Directive**:\n\n"{prompt}"')
    PROMPT_TAG = "TextArea"
    SUPPORTED_INPUTS = ("Image", "Text", "HyperText", "Paragraphs")
    NUM_RESPONSES = int(os.getenv("NUM_RESPONSES", 1))
    TEMPERATURE = float(os.getenv("TEMPERATURE", 0.7))
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", 500))
    
    OPENAI_PROVIDER = os.getenv("OPENAI_PROVIDER", "nvidia")
    OPENAI_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL")
    NVIDIA_NIM_PROVIDER = os.getenv("NVIDIA_NIM_PROVIDER", "nvidia")
    NVIDIA_NIM_ENDPOINT = os.getenv("NVIDIA_NIM_ENDPOINT", "https://integrate.api.nvidia.com/v1")
    NVIDIA_NIM_KEY = os.getenv("NVIDIA_NIM_API_KEY")
    NVIDIA_NIM_MODEL = os.getenv("NVIDIA_NIM_MODEL")
    OLLAMA_ENDPOINT = os.getenv("OLLAMA_ENDPOINT")

    def setup(self):
        if self.DEFAULT_PROMPT and os.path.isfile(self.DEFAULT_PROMPT):
            logger.info(f"Reading default prompt from file: {self.DEFAULT_PROMPT}")
            with open(self.DEFAULT_PROMPT) as f:
                self.DEFAULT_PROMPT = f.read()

    def _ocr(self, image_url):
        # Open the image containing the text
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
        image = ImageOps.exif_transpose(image)

        # Run OCR on the image
        text = pytesseract.image_to_string(image)
        return text

    def _get_text(self, task_data, object_tag):
        """
        """
        data = task_data.get(object_tag.value_name)

        if data is None:
            return None

        if isinstance(object_tag, ImageTag):
            return self._ocr(data)
        elif isinstance(object_tag, ParagraphsTag):
            return json.dumps(data)
        else:
            return data

    def _get_prompts(self, context, prompt_tag) -> List[str]:
        """Getting prompt values
        """
        if context:
            # Interactive mode - get prompt from context
            result = context.get('result')
            for item in result:
                if item.get('from_name') == prompt_tag.name:
                    return item['value']['text']
        # Initializing - get existing prompt from storage
        elif prompt := self.get(prompt_tag.name):
            return [prompt]
        # Default prompt
        elif self.DEFAULT_PROMPT:
            if self.USE_INTERNAL_PROMPT_TEMPLATE:
                logger.error('Using both `DEFAULT_PROMPT` and `USE_INTERNAL_PROMPT_TEMPLATE` is not supported. '
                             'Please either specify `USE_INTERNAL_PROMPT_TEMPLATE=0` or remove `DEFAULT_PROMPT`. '
                             'For now, no prompt will be used.')
                return []
            return [self.DEFAULT_PROMPT]

        return []

    def _match_choices(self, response: List[str], original_choices: List[str]) -> List[str]:
        # assuming classes are separated by newlines
        # TODO: support other guardrails
        matched_labels = []
        predicted_classes = response[0].splitlines()

        for pred in predicted_classes:
            scores = list(map(lambda l: difflib.SequenceMatcher(None, pred, l).ratio(), original_choices))
            matched_labels.append(original_choices[scores.index(max(scores))])

        return matched_labels

    def _find_choices_tag(self, object_tag):
        """Classification predictor
        """
        li = self.label_interface

        try:
            choices_from_name, _, _ = li.get_first_tag_occurence(
                'Choices',
                self.SUPPORTED_INPUTS,
                to_name_filter=lambda s: s == object_tag.name,
            )

            return li.get_control(choices_from_name)
        except:
            return None

    def _find_textarea_tag(self, prompt_tag, object_tag):
        """Free-form text predictor
        """
        li = self.label_interface

        try:
            textarea_from_name, _, _ = li.get_first_tag_occurence(
                'TextArea',
                self.SUPPORTED_INPUTS,
                name_filter=lambda s: s != prompt_tag.name,
                to_name_filter=lambda s: s == object_tag.name,
            )

            return li.get_control(textarea_from_name)
        except:
            return None

    def _find_prompt_tags(self) -> Tuple[ControlTag, ObjectTag]:
        """Find prompting tags in the config
        """
        li = self.label_interface
        prompt_from_name, prompt_to_name, value = li.get_first_tag_occurence(
            # prompt tag
            self.PROMPT_TAG,
            # supported input types
            self.SUPPORTED_INPUTS,
            # if multiple <TextArea> are presented, use one with prefix specified in PROMPT_PREFIX
            name_filter=lambda s: s.startswith(self.PROMPT_PREFIX))

        return li.get_control(prompt_from_name), li.get_object(prompt_to_name)

    def _validate_tags(self, choices_tag: str, textarea_tag: str) -> None:
        if not choices_tag and not textarea_tag:
            raise ValueError('No supported tags found: <Choices> or <TextArea>')

    def _generate_normalized_prompt(self, text: str, prompt: str, task_data: Dict, labels: Optional[List[str]]) -> str:
        """
        """
        if self.USE_INTERNAL_PROMPT_TEMPLATE:
            norm_prompt = self.PROMPT_TEMPLATE.format(text=text, prompt=prompt, labels=labels)
        else:
            norm_prompt = prompt.format(labels=labels, **task_data)

        return norm_prompt

    def _generate_response_regions(self, response: List[str], prompt_tag,
                                   choices_tag: ControlTag, textarea_tag: ControlTag, prompts: List[str]) -> List:
        """
        """
        regions = []

        if choices_tag and len(response) > 0:
            matched_labels = self._match_choices(response, choices_tag.labels)
            regions.append(choices_tag.label(matched_labels))

        if textarea_tag:
            regions.append(textarea_tag.label(text=response))

        # not sure why we need this but it was in the original code
        regions.append(prompt_tag.label(text=prompts))

        return regions

    def _predict_single_task(self, task_data: Dict, prompt_tag: Any, object_tag: Any, prompt: str,
                             choices_tag: ControlTag, textarea_tag: ControlTag, prompts: List[str]) -> Dict:
        """
        """
        text = self._get_text(task_data, object_tag)
        # Add {labels} to the prompt if choices tag is present
        labels = choices_tag.labels if choices_tag else None
        norm_prompt = self._generate_normalized_prompt(text, prompt, task_data, labels=labels)

        # run inference
        # this are params provided through the web interface
        response = gpt(norm_prompt, self.extra_params)
        regions = self._generate_response_regions(response, prompt_tag, choices_tag, textarea_tag, prompts)

        return PredictionValue(result=regions, score=0.1, model_version=str(self.model_version))

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        """
        """
        predictions = []

        # prompt tag contains the prompt in the config
        # object tag contains what we plan to label
        prompt_tag, object_tag = self._find_prompt_tags()
        prompts = self._get_prompts(context, prompt_tag)

        if prompts:
            prompt = "\n".join(prompts)

            choices_tag = self._find_choices_tag(object_tag)
            textarea_tag = self._find_textarea_tag(prompt_tag, object_tag)
            self._validate_tags(choices_tag, textarea_tag)

            for task in tasks:
                # preload all task data fields, they are needed for prompt
                task_data = self.preload_task_data(task, task['data'])
                pred = self._predict_single_task(task_data, prompt_tag, object_tag, prompt,
                                                 choices_tag, textarea_tag, prompts)
                predictions.append(pred)

        return ModelResponse(predictions=predictions)

    def _prompt_diff(self, old_prompt, new_prompt):
        """
        """
        old_lines = old_prompt.splitlines()
        new_lines = new_prompt.splitlines()
        diff = difflib.unified_diff(old_lines, new_lines, lineterm="")

        return "\n".join(
            line for line in diff if line.startswith(('+',)) and not line.startswith(('+++', '---')))

    def fit(self, event, data, **additional_params):
        """
        """
        logger.debug(f'Data received: {data}')
        if event not in ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED'):
            return

        prompt_tag, object_tag = self._find_prompt_tags()
        prompts = self._get_prompts(data['annotation'], prompt_tag)

        if not prompts:
            logger.debug(f'No prompts recorded.')
            return

        prompt = '\n'.join(prompts)
        current_prompt = self.get(prompt_tag.name)

        # find substrings that differ between current and new prompt
        # if there are no differences, skip training
        if current_prompt:
            diff = self._prompt_diff(current_prompt, prompt)
            if not diff:
                logger.debug('No prompt diff found.')
                return

            logger.debug(f'Prompt diff: {diff}')
        self.set(prompt_tag.name, prompt)
        model_version = self.bump_model_version()

        logger.debug(f'Updated model version to {str(model_version)}')
