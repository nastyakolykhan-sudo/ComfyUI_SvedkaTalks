"""
ST SeC Nodes
------------
Lazy-loading wrappers for SeC (Segment Concept) video segmentation.
Source: 9nate-drake/Comfyui-SecNodes (Apache 2.0)

Node interfaces are defined statically (fast startup).
Actual inference code is imported only when a node is executed.

Includes:
  - SeCModelLoader
  - SeCVideoSegmentation
"""

import os
import sys

# Path to the bundled SeC implementation
_SEC_DIR = os.path.join(os.path.dirname(__file__), "_sec")


def _ensure_sec_on_path():
    """Add the bundled _sec directory to sys.path so its imports resolve."""
    if _SEC_DIR not in sys.path:
        sys.path.insert(0, _SEC_DIR)


def _load_sec_impl():
    """Lazy-load the full SeC implementation. Called only at inference time."""
    _ensure_sec_on_path()
    try:
        from _sec import sec_nodes_impl as _impl
        return _impl
    except ImportError:
        # Try direct path import
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "_sec_impl",
            os.path.join(_SEC_DIR, "sec_nodes_impl.py")
        )
        mod = importlib.util.module_from_spec(spec)
        # Inject inference path so relative imports inside sec_nodes_impl work
        sys.path.insert(0, _SEC_DIR)
        spec.loader.exec_module(mod)
        return mod


# ===========================================================================
# SeCModelLoader
# ===========================================================================

class STSeCModelLoader:
    """
    Load the SeC-4B segmentation model.
    Model files must be in ComfyUI's models/sams/ directory.
    """

    @classmethod
    def INPUT_TYPES(cls):
        import torch
        device_choices = ["auto", "cpu"]
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                device_choices.append(f"gpu{i}")

        # Try to list available models; fall back to placeholder if inference not ready
        try:
            _ensure_sec_on_path()
            import importlib.util, folder_paths, os
            sams_dir = os.path.join(folder_paths.models_dir, "sams")
            models = [f for f in os.listdir(sams_dir)
                      if f.endswith(('.safetensors', '.bin', '.pt'))
                      and 'sec' in f.lower().replace('-', '').replace('_', '')]
            if not models:
                models = ["(No SeC models found in models/sams/)"]
        except Exception:
            models = ["SeC-4B-fp16.safetensors"]

        return {
            "required": {
                "model_file": (models,),
                "device": (device_choices, {"default": "auto"}),
            },
            "optional": {
                "use_flash_attn":    ("BOOLEAN", {"default": True}),
                "allow_mask_overlap": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("SEC_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "SvedkaTalks/SeC"
    TITLE = "ST: SeC Model Loader"

    def load_model(self, model_file, device, use_flash_attn=True, allow_mask_overlap=True):
        impl = _load_sec_impl()
        loader = impl.SeCModelLoader()
        return loader.load_model(model_file, device, use_flash_attn, allow_mask_overlap)


# ===========================================================================
# SeCVideoSegmentation
# ===========================================================================

class STSeCVideoSegmentation:
    """
    Segment objects across video frames using the SeC model.
    Provide an initial mask or bounding box to identify the object.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SEC_MODEL", {"tooltip": "SeC model loaded from SeCModelLoader node"}),
                "frames": ("IMAGE",),
                "positive_points":    ("STRING", {"default": ""}),
                "negative_points":    ("STRING", {"default": ""}),
                "tracking_direction": (["bidirectional", "forward", "backward"],),
                "annotation_frame_idx": ("INT", {"default": 0, "min": 0, "max": 9999, "step": 1}),
                "num_objects": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                "max_frames": ("INT", {"default": -1, "min": -1, "max": 9999, "step": 1}),
                "num_iterations": ("INT", {"default": 12, "min": 1, "max": 50, "step": 1}),
                "use_point_tracking": ("BOOLEAN", {"default": False}),
                "allow_new_objects":  ("BOOLEAN", {"default": True}),
                "keep_model_loaded":  ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Keep SeC model in VRAM after segmentation. Recommended on A100 80GB — avoids reloading from disk on every run. Disable only if you need the VRAM for other nodes.",
                }),
            },
            "optional": {
                "bbox":       ("BBOX",),
                "input_mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("MASK", "INT",)
    RETURN_NAMES = ("masks", "object_ids",)
    FUNCTION = "segment"
    CATEGORY = "SvedkaTalks/SeC"
    TITLE = "ST: SeC Video Segmentation"

    def segment(self, model, frames, tracking_direction, annotation_frame_idx,
                num_objects, max_frames, num_iterations, use_point_tracking,
                allow_new_objects, keep_model_loaded=True,
                bbox=None, input_mask=None,
                positive_points="", negative_points=""):
        impl = _load_sec_impl()
        node = impl.SeCVideoSegmentation()
        return node.segment_video(
            model=model,
            frames=frames,
            bbox=bbox,
            input_mask=input_mask,
            tracking_direction=tracking_direction,
            annotation_frame_idx=annotation_frame_idx,
            positive_points=positive_points,
            negative_points=negative_points,
            max_frames_to_track=max_frames,
            mllm_memory_size=num_iterations,
            auto_unload_model=(not keep_model_loaded),
        )


# ===========================================================================
# NODE MAPPINGS
# ===========================================================================

NODE_CLASS_MAPPINGS = {
    "SeCModelLoader":       STSeCModelLoader,
    "SeCVideoSegmentation": STSeCVideoSegmentation,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SeCModelLoader":       "ST: SeC Model Loader",
    "SeCVideoSegmentation": "ST: SeC Video Segmentation",
}
