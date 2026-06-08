"""
ST Audio Nodes
--------------
Minimal extracts from ComfyUI_RyanOnTheInside_CUSTOM.

Includes:
  - ST: AudioPad                 (pad_start / pad_end in seconds)
  - ST: AudioFeatureExtractor
  - ST: FeatureToFlexIntParam
"""

import torch
import torch.nn.functional as nnf
import numpy as np


# ===========================================================================
# AudioPad — self-contained (no external dependencies)
# ===========================================================================

def _pad_audio(waveform, pad_left, pad_right, pad_mode):
    return nnf.pad(waveform, (pad_left, pad_right), mode=pad_mode)


class STAudioPad:
    """
    Pad audio with silence (or reflected/replicated signal) at start and end.
    pad_start / pad_end are in seconds (FLOAT).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "pad_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3600.0, "step": 0.1}),
                "pad_end":   ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3600.0, "step": 0.1}),
                "pad_mode":  (["constant", "reflect", "replicate", "circular"],),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "run"
    CATEGORY = "SvedkaTalks/Audio"

    def run(self, audio, pad_start, pad_end, pad_mode):
        waveform    = audio["waveform"]
        sample_rate = audio["sample_rate"]
        pad_l = int(pad_start * sample_rate)
        pad_r = int(pad_end   * sample_rate)
        padded = _pad_audio(waveform, pad_l, pad_r, pad_mode)
        return ({"waveform": padded, "sample_rate": sample_rate},)


# ===========================================================================
# Audio feature infrastructure (pulled from _roti sub-package)
# ===========================================================================

def _load_audio_feature():
    """Lazy import AudioFeature from _roti to avoid top-level import errors."""
    try:
        from ._roti.features_audio import AudioFeature
        return AudioFeature
    except Exception as e:
        print(f"[SvedkaTalks] AudioFeature import failed: {e}")
        return None


# ===========================================================================
# AudioFeatureExtractor
# ===========================================================================

class STAudioFeatureExtractor:
    """
    Extract per-frame audio features (amplitude, RMS, spectral, etc.).
    Returns a FEATURE object and the resolved frame_count.
    Matches AudioFeatureExtractor interface.
    """

    # Extraction methods — lazy resolved from AudioFeature at first call
    _methods = None

    @classmethod
    def _get_methods(cls):
        if cls._methods is None:
            AudioFeature = _load_audio_feature()
            if AudioFeature is not None:
                try:
                    cls._methods = AudioFeature.get_extraction_methods()
                except Exception:
                    cls._methods = ["amplitude_envelope"]
            else:
                cls._methods = ["amplitude_envelope"]
        return cls._methods

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio":            ("AUDIO",),
                "extraction_method": (cls._get_methods(),),
                "frame_rate":       ("FLOAT", {"default": 30.0, "min": 1.0, "max": 120.0, "step": 0.1}),
                "frame_count":      ("INT",   {"default": 0,    "min": 0,   "max": 999999}),
                "width":            ("INT",   {"default": 512,  "min": 64,  "max": 4096, "step": 64}),
                "height":           ("INT",   {"default": 512,  "min": 64,  "max": 4096, "step": 64}),
            }
        }

    RETURN_TYPES = ("FEATURE", "INT",)
    RETURN_NAMES = ("feature", "frame_count",)
    FUNCTION = "run"
    CATEGORY = "SvedkaTalks/Audio"

    def run(self, audio, extraction_method, frame_rate, frame_count, width, height):
        AudioFeature = _load_audio_feature()
        if AudioFeature is None:
            raise RuntimeError("[SvedkaTalks] AudioFeature could not be loaded.")

        waveform    = audio["waveform"]
        sample_rate = audio["sample_rate"]
        natural_frames = int((waveform.shape[-1] / sample_rate) * frame_rate)
        target_frames  = frame_count if frame_count > 0 else natural_frames

        feature = AudioFeature(
            width=width,
            height=height,
            feature_name=extraction_method,
            audio=audio,
            frame_count=target_frames,
            frame_rate=frame_rate,
            feature_type=extraction_method,
        )
        feature.extract()
        return (feature, target_frames)


# ===========================================================================
# FeatureToFlexIntParam
# ===========================================================================

class STFeatureToFlexIntParam:
    """
    Map a FEATURE's per-frame values to a list of INTs in [lower, upper] range.
    Matches FeatureToFlexIntParam interface.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "feature":         ("FEATURE",),
                "lower_threshold": ("INT", {"default": 0,   "min": -10000, "max": 10000, "step": 1}),
                "upper_threshold": ("INT", {"default": 100, "min": -10000, "max": 10000, "step": 1}),
                "invert_output":   ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("PARAMETER",)
    FUNCTION = "run"
    CATEGORY = "SvedkaTalks/Audio"

    def run(self, feature, lower_threshold, upper_threshold, invert_output):
        n = feature.frame_count
        values = np.array([feature.get_value_at_frame(i) for i in range(n)], dtype=np.float64)

        lo = np.array(lower_threshold if isinstance(lower_threshold, (list, tuple)) else [lower_threshold] * n, dtype=np.float64)
        hi = np.array(upper_threshold if isinstance(upper_threshold, (list, tuple)) else [upper_threshold] * n, dtype=np.float64)
        # Pad/trim to length n
        if len(lo) < n: lo = np.pad(lo, (0, n - len(lo)), mode='edge')
        if len(hi) < n: hi = np.pad(hi, (0, n - len(hi)), mode='edge')
        lo, hi = lo[:n], hi[:n]

        min_val = getattr(feature, 'min_value', values.min())
        max_val = getattr(feature, 'max_value', values.max())

        if max_val == min_val:
            normalized = lo.copy()
        else:
            normalized = lo + (hi - lo) * (values - min_val) / (max_val - min_val)

        if invert_output:
            normalized = hi - (normalized - lo)

        return (np.round(normalized).astype(int).tolist(),)


# ===========================================================================
# PreviewFeature
# ===========================================================================

class STPreviewFeature:
    """
    Render a FEATURE's per-frame values as a matplotlib chart.
    Displays as an image preview below the node. Matches PreviewFeature interface.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "feature": ("FEATURE",),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "preview"
    CATEGORY = "SvedkaTalks/Audio"
    OUTPUT_NODE = True

    def preview(self, feature, prompt=None, extra_pnginfo=None):
        import os
        import random
        import matplotlib.pyplot as plt
        from io import BytesIO
        from PIL import Image as PILImage
        import folder_paths

        width, height = 960, 540
        values = [feature.get_value_at_frame(i) for i in range(feature.frame_count)]

        actual_min = min(values)
        actual_max = max(values)

        plt.figure(figsize=(width / 100, height / 100), dpi=100)
        plt.style.use('dark_background')
        plt.plot(values, color='dodgerblue', linewidth=2)
        plt.xlabel('Frame', color='white', fontsize=14)
        plt.ylabel('Value', color='white', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.3, color='gray')
        plt.tick_params(axis='both', colors='white', labelsize=12)

        max_ticks = 10
        step = max(1, len(values) // max_ticks)
        plt.xticks(range(0, len(values), step), [str(x) for x in range(0, len(values), step)])

        y_range = actual_max - actual_min
        if y_range == 0:
            y_range = 1.0
            pad = 0.1
        else:
            pad = 0.05
        plt.ylim(actual_min - pad * y_range, actual_max + pad * y_range)

        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['bottom'].set_color('white')
        plt.gca().spines['left'].set_color('white')
        plt.title(f'Feature: {feature.name}', color='white', fontsize=16)
        plt.tight_layout(pad=0.5)

        buf = BytesIO()
        plt.savefig(buf, format='png', facecolor='black', edgecolor='none')
        buf.seek(0)
        plt.close()

        img = PILImage.open(buf)
        img_array = np.array(img)
        buf.close()

        output_dir = folder_paths.get_temp_directory()
        prefix = "feature_preview_" + ''.join(random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(5))
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            prefix, output_dir, img_array.shape[1], img_array.shape[0]
        )
        file = f"{filename}_{counter:05}_.png"
        PILImage.fromarray(np.clip(img_array, 0, 255).astype(np.uint8)).save(
            os.path.join(full_output_folder, file), compress_level=1
        )

        return {"ui": {"images": [{"filename": file, "subfolder": subfolder, "type": "temp"}]}}


# ===========================================================================
# NODE MAPPINGS
# ===========================================================================

NODE_CLASS_MAPPINGS = {
    "AudioPad":               STAudioPad,
    "AudioFeatureExtractor":  STAudioFeatureExtractor,
    "FeatureToFlexIntParam":  STFeatureToFlexIntParam,
    "PreviewFeature":         STPreviewFeature,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioPad":               "ST: AudioPad",
    "AudioFeatureExtractor":  "ST: AudioFeatureExtractor",
    "FeatureToFlexIntParam":  "ST: FeatureToFlexIntParam",
    "PreviewFeature":         "ST: PreviewFeature",
}
