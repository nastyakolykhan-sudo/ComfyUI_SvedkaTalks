"""
ST ElevenLabs TTS Node
-----------------------
Self-contained ElevenLabs Text-to-Speech node.
Source: sysL-padawan/comfyui-elevenlabs-integration interface
Uses the ElevenLabs REST API directly (no SDK required, but SDK used if installed).

Node key: "ElevenlabsTextToSpeech" — matches workflow mapping exactly.
"""

import io
import torch
import numpy as np


def _tts_via_requests(text, api_key, voice_id, model_id, stability, similarity_boost,
                      style, speed, use_speaker_boost, previous_text="", next_text="",
                      language_code=None):
    """Call ElevenLabs TTS REST API and return raw audio bytes."""
    import requests

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    }
    payload = {
        "text": text,
        "model_id": model_id,
        "voice_settings": {
            "stability": stability,
            "similarity_boost": similarity_boost,
            "style": style,
            "use_speaker_boost": use_speaker_boost,
            "speed": speed,
        },
    }
    if previous_text:
        payload["previous_text"] = previous_text
    if next_text:
        payload["next_text"] = next_text
    if language_code and language_code not in ("", "auto"):
        payload["language_code"] = language_code

    resp = requests.post(url, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    return resp.content


def _decode_mp3_bytes(audio_bytes, sample_rate=44100):
    """
    Decode raw MP3/PCM bytes to a torch waveform tensor.
    Tries soundfile, then torchaudio, then pydub as fallback.
    Returns (waveform [1, C, T], sample_rate).
    """
    buf = io.BytesIO(audio_bytes)

    # Try soundfile
    try:
        import soundfile as sf
        data, sr = sf.read(buf, dtype="float32", always_2d=True)
        # data: [T, C] → [C, T]
        waveform = torch.from_numpy(data.T).unsqueeze(0)  # [1, C, T]
        return waveform, sr
    except Exception:
        pass

    buf.seek(0)

    # Try torchaudio
    try:
        import torchaudio
        waveform, sr = torchaudio.load(buf)
        return waveform.unsqueeze(0), sr  # [1, C, T]
    except Exception:
        pass

    buf.seek(0)

    # Try pydub
    try:
        from pydub import AudioSegment
        seg = AudioSegment.from_file(buf)
        samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
        samples /= float(2 ** (seg.sample_width * 8 - 1))
        if seg.channels > 1:
            samples = samples.reshape(-1, seg.channels).T
        else:
            samples = samples[np.newaxis, :]
        waveform = torch.from_numpy(samples).unsqueeze(0)  # [1, C, T]
        return waveform, seg.frame_rate
    except Exception:
        pass

    raise RuntimeError(
        "[SvedkaTalks] Could not decode ElevenLabs audio. "
        "Install soundfile, torchaudio, or pydub."
    )


class STElevenlabsTextToSpeech:
    """
    Convert text to speech using the ElevenLabs API.
    Outputs a ComfyUI AUDIO dict.

    Widget order matches the original ElevenlabsTextToSpeech node:
      text, api_key, voice_id, model_id, style_prompt (unused),
      language, stability, use_speaker_boost, similarity_boost, style, speed
    """

    _MODELS = [
        "eleven_multilingual_v2",
        "eleven_english_sts_v2",
        "eleven_turbo_v2",
        "eleven_turbo_v2_5",
        "eleven_flash_v2",
        "eleven_flash_v2_5",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": "Hello from SvedkaTalks.",
                }),
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                }),
                "voice_id": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "ElevenLabs voice ID (from your ElevenLabs account).",
                }),
                "model_id": (cls._MODELS, {"default": "eleven_multilingual_v2"}),
                "style_prompt": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Optional style description (not all models support this).",
                }),
                "language": ("STRING", {
                    "default": "auto",
                    "multiline": False,
                    "tooltip": "Language code (e.g. 'en', 'es') or 'auto'.",
                }),
                "stability": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                }),
                "use_speaker_boost": ("BOOLEAN", {"default": True}),
                "similarity_boost": ("FLOAT", {
                    "default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01,
                }),
                "style": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Style exaggeration (0 = original, 1 = max).",
                }),
                "speed": ("FLOAT", {
                    "default": 1.0, "min": 0.25, "max": 4.0, "step": 0.05,
                }),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "run"
    CATEGORY = "SvedkaTalks/Audio"
    TITLE = "ST: ElevenLabs TTS"

    def run(self, text, api_key, voice_id, model_id, style_prompt,
            language, stability, use_speaker_boost, similarity_boost, style, speed):

        if not api_key.strip():
            raise ValueError("[SvedkaTalks] ElevenLabs api_key is empty.")
        if not voice_id.strip():
            raise ValueError("[SvedkaTalks] ElevenLabs voice_id is empty.")

        lang = language.strip() if language.strip() not in ("", "auto") else None

        audio_bytes = _tts_via_requests(
            text=text,
            api_key=api_key.strip(),
            voice_id=voice_id.strip(),
            model_id=model_id,
            stability=stability,
            similarity_boost=similarity_boost,
            style=style,
            speed=speed,
            use_speaker_boost=use_speaker_boost,
            language_code=lang,
        )

        waveform, sample_rate = _decode_mp3_bytes(audio_bytes)
        return ({"waveform": waveform, "sample_rate": sample_rate},)


# ===========================================================================
# NODE MAPPINGS
# ===========================================================================

NODE_CLASS_MAPPINGS = {
    "ElevenlabsTextToSpeech": STElevenlabsTextToSpeech,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ElevenlabsTextToSpeech": "ST: ElevenLabs TTS",
}
