import io
import logging
import os
import time

import numpy as np
import requests
import torch
from pydub import AudioSegment


class ElevenLabsBase:
    voices_cache = None
    voices_map = {}
    last_voice_fetch_time = 0
    cache_duration = 3600  # seconds

    @classmethod
    def _get_api_key(cls):
        api_key = os.getenv("ELEVENLABS_API_KEY", "").strip()
        if not api_key:
            logging.error(
                "ELEVENLABS_API_KEY is not set. Please configure it in your environment."
            )
            return None
        return api_key

    @classmethod
    def fetch_elevenlabs_voices(cls):
        current_time = time.time()
        if cls.voices_cache is None or (
            current_time - cls.last_voice_fetch_time > cls.cache_duration
        ):
            voice_list = []
            voice_map = {}
            api_key = cls._get_api_key()
            if api_key is None:
                cls.voices_cache = ["missing_api_key"]
                cls.voices_map = {}
                return cls.voices_cache

            url = "https://api.elevenlabs.io/v2/voices"
            next_page_token = None
            name_counts = {}
            try:
                while True:
                    params = {"page_size": 100}
                    if next_page_token:
                        params["next_page_token"] = next_page_token

                    response = requests.get(
                        url, headers={"xi-api-key": api_key}, params=params
                    )
                    response.raise_for_status()
                    payload = response.json()

                    voices = payload.get("voices", [])
                    for voice in voices:
                        base_name = voice.get("name", "unnamed_voice")
                        count = name_counts.get(base_name, 0) + 1
                        name_counts[base_name] = count
                        display_name = (
                            base_name if count == 1 else f"{base_name} ({count})"
                        )
                        voice_list.append(display_name)
                        voice_id = voice.get("voice_id")
                        if voice_id:
                            voice_map[display_name] = voice_id

                    has_more = payload.get("has_more", False)
                    next_page_token = payload.get("next_page_token")
                    if not has_more or not next_page_token:
                        break

                cls.voices_cache = voice_list if voice_list else ["no_voices_found"]
                cls.voices_map = voice_map
                cls.last_voice_fetch_time = current_time
            except requests.exceptions.RequestException as e:
                logging.error(f"Error fetching voices: {e}")
                if cls.voices_cache is None:
                    cls.voices_cache = ["error_fetching_voices"]
                cls.voices_map = {}
        return cls.voices_cache

    def ensure_3d_tensor(self, tensor):
        if tensor.dim() == 1:
            return tensor.unsqueeze(0).unsqueeze(0)
        elif tensor.dim() == 2:
            return tensor.unsqueeze(0)
        elif tensor.dim() > 3:
            return tensor.squeeze().unsqueeze(0)
        return tensor

    def _load_audio_from_bytes(self, audio_bytes: bytes):
        """Decode ElevenLabs MP3 response into torch waveform."""
        if not audio_bytes:
            logging.error("Empty audio response received.")
            return None, None

        try:
            audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")
        except Exception as e:
            preview = audio_bytes[:80]
            try:
                preview_text = preview.decode("utf-8", errors="replace")
            except Exception:
                preview_text = str(preview)
            logging.error(f"Error decoding MP3 audio: {e}. Preview: {preview_text}")
            return None, None

        sample_rate = audio.frame_rate
        sample_width = audio.sample_width
        channels = audio.channels

        samples = np.array(audio.get_array_of_samples())
        if channels > 1:
            samples = samples.reshape(-1, channels).T  # channels x samples
        else:
            samples = samples.reshape(1, -1)

        max_val = float(2 ** (8 * sample_width - 1))
        waveform = torch.from_numpy(samples).unsqueeze(0).float() / max_val
        waveform = self.ensure_3d_tensor(waveform)

        return waveform, sample_rate
