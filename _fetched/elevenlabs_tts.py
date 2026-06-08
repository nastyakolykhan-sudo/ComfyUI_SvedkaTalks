import logging
from typing import Optional

import requests
import torch

from .base import ElevenLabsBase


class ElevenLabsTTSNode(ElevenLabsBase):
    tts_models_cache = None
    last_tts_model_fetch_time = 0

    @classmethod
    def fetch_elevenlabs_models(cls):
        current_time = cls.last_voice_fetch_time
        if cls.tts_models_cache is None or (
            current_time - cls.last_tts_model_fetch_time > cls.cache_duration
        ):
            cls.tts_models_cache = [
                "eleven_multilingual_v2",
                "eleven_english_sts_v2",
                "eleven_turbo_v2",
            ]
            cls.last_tts_model_fetch_time = current_time
        return cls.tts_models_cache

    @classmethod
    def INPUT_TYPES(cls):
        voices = cls.fetch_elevenlabs_voices()
        models = cls.fetch_elevenlabs_models()
        return {
            "required": {
                "text": (
                    "STRING",
                    {"multiline": True, "default": "Hello, how are you?"},
                ),
                "previous_text": ("STRING", {"multiline": True, "default": ""}),
                "next_text": ("STRING", {"multiline": True, "default": ""}),
                "voice": (voices,),
                "model": (models,),
                "stability": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1},
                ),
                "similarity_boost": (
                    "FLOAT",
                    {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.1},
                ),
                "style": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1},
                ),
                "speed": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.25, "max": 4.0, "step": 0.05},
                ),
                "use_speaker_boost": (
                    "BOOLEAN",
                    {
                        "default": True,
                    },
                ),
            },
            "optional": {
                "input_text": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "generate_speech"
    CATEGORY = "ElevenLabs"

    def __init__(self):
        super().__init__()

    def generate_speech(
        self,
        text: str,
        previous_text: Optional[str],
        next_text: Optional[str],
        voice: str,
        model: str,
        stability: float,
        similarity_boost: float,
        style: float,
        speed: float,
        use_speaker_boost: bool,
        input_text: Optional[str] = None,
    ):
        final_text = input_text if input_text is not None else text

        voice_id = self.voices_map.get(voice)
        if not voice_id:
            logging.error(
                f"Voice ID not found for selection '{voice}'. Refreshing voices cache."
            )
            self.fetch_elevenlabs_voices()
            voice_id = self.voices_map.get(voice)
            if not voice_id:
                return (
                    {"waveform": torch.zeros(1, 1, 1).float(), "sample_rate": 44100},
                )

        api_key = self._get_api_key()
        if api_key is None:
            return ({"waveform": torch.zeros(1, 1, 1).float(), "sample_rate": 44100},)

        headers = {
            "Accept": "audio/mpeg",
            "xi-api-key": api_key,
            "Content-Type": "application/json",
        }

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        payload = {
            "text": final_text,
            "model_id": model,
            "voice_settings": {
                "stability": stability,
                "similarity_boost": similarity_boost,
                "style": style,
                "speed": speed,
                "use_speaker_boost": use_speaker_boost,
            },
        }
        if previous_text:
            payload["previous_text"] = previous_text
        if next_text:
            payload["next_text"] = next_text
        try:
            response = requests.post(url, headers=headers, json=payload)
        except requests.exceptions.RequestException as e:
            logging.error(f"Error in text-to-speech: {str(e)}")
            return ({"waveform": torch.zeros(1, 1, 1).float(), "sample_rate": 44100},)

        if response.status_code == 200:
            waveform, sample_rate = self._load_audio_from_bytes(response.content)
            if waveform is None or sample_rate is None:
                return (
                    {"waveform": torch.zeros(1, 1, 1).float(), "sample_rate": 44100},
                )

            if waveform.dim() != 3:
                logging.info(
                    f"Unexpected tensor dimension {waveform.dim()}. Reshaping to 3D."
                )
                waveform = waveform.view(1, 1, -1)

            return ({"waveform": waveform, "sample_rate": sample_rate},)
        else:
            logging.error(f"API Error: {response.status_code} - {response.text}")
            return ({"waveform": torch.zeros(1, 1, 1).float(), "sample_rate": 44100},)

    @classmethod
    def IS_CHANGED(
        cls,
        text,
        previous_text,
        next_text,
        voice,
        model,
        stability,
        similarity_boost,
        style,
        speed,
        use_speaker_boost,
        input_text=None,
    ):
        return (
            text,
            previous_text,
            next_text,
            voice,
            model,
            stability,
            similarity_boost,
            style,
            speed,
            use_speaker_boost,
            input_text,
        )
