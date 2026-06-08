"""
ST VHS / Audio Blend Nodes
--------------------------
Minimal extracts / rewrites from VideoHelperSuite and custom audio blend logic.

Includes:
  - ST: VHS_GetImageCount  — count frames in an IMAGE batch
  - ST: SET_AudioBlend     — mix two audio streams with per-stream volume
"""

import torch
import torch.nn.functional as nnf


# ===========================================================================
# VHS_GetImageCount
# ===========================================================================

class STVHSGetImageCount:
    """Return the number of frames in an IMAGE batch. Matches VHS_GetImageCount."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",)}}

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("count",)
    FUNCTION = "run"
    CATEGORY = "SvedkaTalks/Video"

    def run(self, images):
        return (images.shape[0],)


# ===========================================================================
# SET_AudioBlend
# ===========================================================================

class STSETAudioBlend:
    """
    Mix two AUDIO streams.
    audio1 and audio2 can have different lengths — shorter is zero-padded.
    vol1 / vol2 control per-stream volume (1.0 = unchanged).
    Matches SET_AudioBlend interface: widgets [vol1, vol2].
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio1": ("AUDIO",),
                "vol1":   ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.01}),
                "vol2":   ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.01}),
            },
            "optional": {
                "audio2": ("AUDIO",),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio_out",)
    FUNCTION = "run"
    CATEGORY = "SvedkaTalks/Audio"

    def run(self, audio1, vol1, vol2, audio2=None):
        wf1 = audio1["waveform"]
        sr1 = audio1["sample_rate"]

        if audio2 is None:
            return ({"waveform": wf1 * vol1, "sample_rate": sr1},)

        wf2 = audio2["waveform"]
        sr2 = audio2["sample_rate"]

        # Resample audio2 to sr1 if needed (simple: only if same waveform length is close)
        if sr1 != sr2:
            # Use torchaudio resample if available, else skip
            try:
                import torchaudio.functional as F_audio
                wf2 = F_audio.resample(wf2, sr2, sr1)
            except ImportError:
                pass  # Proceed with mismatched rate — will sound slightly off

        # Zero-pad to match lengths
        l1, l2 = wf1.shape[-1], wf2.shape[-1]
        if l1 > l2:
            wf2 = nnf.pad(wf2, (0, l1 - l2))
        elif l2 > l1:
            wf1 = nnf.pad(wf1, (0, l2 - l1))

        blended = wf1 * vol1 + wf2 * vol2
        return ({"waveform": blended, "sample_rate": sr1},)


# ===========================================================================
# NODE MAPPINGS
# ===========================================================================

NODE_CLASS_MAPPINGS = {
    "VHS_GetImageCount": STVHSGetImageCount,
    "SET_AudioBlend":    STSETAudioBlend,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VHS_GetImageCount": "ST: VHS_GetImageCount",
    "SET_AudioBlend":    "ST: SET_AudioBlend",
}
