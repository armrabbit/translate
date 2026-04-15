from typing import Any

from .gpt import GPTTranslation
from ...utils.translator_utils import MODEL_MAP


class DeepseekTranslation(GPTTranslation):
    """Translation engine using Deepseek models with OpenAI-compatible API."""
    
    def __init__(self):
        super().__init__()
        self.supports_images = False
        self.api_base_url = "https://api.deepseek.com/v1"
    
    def initialize(self, settings: Any, source_lang: str, target_lang: str, model_name: str, **kwargs) -> None:
        """
        Initialize Deepseek translation engine.
        
        Args:
            settings: Settings object with credentials
            source_lang: Source language name
            target_lang: Target language name
            model_name: Deepseek model name
        """
        # Call BaseLLMTranslation's initialize
        super(GPTTranslation, self).initialize(settings, source_lang, target_lang, **kwargs)
        
        self.model_name = model_name
        credentials = settings.get_credentials(settings.ui.tr('Deepseek'))
        self.api_key = credentials.get('api_key', '')
        self.model = MODEL_MAP.get(self.model_name)
        self.timeout = 120

    def _perform_translation(self, user_prompt: str, system_prompt: str, image) -> str:
        """
        Perform translation using Deepseek's OpenAI-compatible chat completions API.

        Deepseek expects `max_tokens` in the payload (not `max_completion_tokens`).
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": user_prompt}]
            }
        ]

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
        }

        return self._make_api_request(payload, headers)

