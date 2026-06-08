"""
ST SAM3 Nodes
--------------
Compatibility wrapper for SAM3 segmentation nodes.

The workflow uses the legacy `SAM3_Detect` interface (ComfyUI old-style).
The current ComfyUI-SAM3 v3.x uses the new `io.ComfyNode` / `define_schema` API
with different mapping keys (SAM3Grounding, SAM3Segmentation, etc.).

This file:
  - Registers `SAM3_Detect` so the workflow loads without "unknown node" errors
  - At execution time, tries to delegate to an installed SAM3 implementation
  - Falls back with a clear error message if SAM3 is not available

For full SAM3 functionality, install ComfyUI-SAM3:
  https://github.com/PozzettiAndrea/ComfyUI-SAM3
Then DISABLE its auto-discovery (rename folder or add .disabled) so only
SvedkaTalks registers the nodes.
"""

import torch
import numpy as np


class AnyType(str):
    def __ne__(self, other):
        return False

any_typ = AnyType("*")


def _try_load_sam3_grounding():
    """Try to import SAM3Grounding from an installed package."""
    try:
        from ComfyUI_SAM3.nodes.segmentation import SAM3Grounding
        return SAM3Grounding
    except ImportError:
        pass
    try:
        from nodes.segmentation import SAM3Grounding
        return SAM3Grounding
    except ImportError:
        pass
    return None


# ===========================================================================
# SAM3_Detect
# Matches the legacy interface seen in the workflow JSON.
# ===========================================================================

class STSAM3Detect:
    """
    SAM3-based object detection and segmentation.

    Inputs match the legacy SAM3_Detect interface:
      - model      : SAM3 model (from a companion loader node)
      - image      : Single frame or batch
      - conditioning: CLIP text conditioning for grounded detection
      - bboxes     : Optional bounding boxes
      - positive_coords / negative_coords : JSON coordinate strings
      - threshold  : Confidence threshold
      - refine_iterations : Number of mask refinement passes
      - individual_masks  : Return one mask per detection vs union

    Outputs: masks (MASK), bboxes (BOUNDING_BOX)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":             ("MODEL",),
                "image":             ("IMAGE",),
                "threshold":         ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "refine_iterations": ("INT",   {"default": 2,   "min": 0,   "max": 10,  "step": 1}),
                "individual_masks":  ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "conditioning":     ("CONDITIONING",),
                "bboxes":           ("BOUNDING_BOX",),
                "positive_coords":  ("STRING", {"default": ""}),
                "negative_coords":  ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("MASK", "BOUNDING_BOX",)
    RETURN_NAMES = ("masks", "bboxes",)
    FUNCTION = "run"
    CATEGORY = "SvedkaTalks/SAM3"
    TITLE = "ST: SAM3 Detect"

    def run(self, model, image, threshold, refine_iterations, individual_masks,
            conditioning=None, bboxes=None, positive_coords="", negative_coords=""):

        # Try to use the actual SAM3 implementation if available
        # (ComfyUI-SAM3 must be installed but its __init__.py disabled from auto-load)
        SAM3Grounding = _try_load_sam3_grounding()

        if SAM3Grounding is not None:
            # Map legacy params to new API as best we can
            # Extract text prompt from conditioning if possible
            text_prompt = ""
            if conditioning is not None and len(conditioning) > 0:
                cond = conditioning[0]
                if isinstance(cond, (list, tuple)) and len(cond) > 1:
                    cond_dict = cond[1] if isinstance(cond[1], dict) else {}
                    text_prompt = cond_dict.get("text", "")

            # Build a minimal sam3_model_config dict from the model object
            # This is a best-effort bridge — may need adjustment for specific SAM3 versions
            try:
                result = SAM3Grounding.execute(
                    sam3_model_config=model,
                    image=image,
                    confidence_threshold=threshold,
                    text_prompt=text_prompt,
                    positive_boxes=bboxes,
                )
                # result is (masks, visualization, boxes_json, scores_json)
                return (result[0], bboxes or [])
            except Exception as e:
                print(f"[SvedkaTalks] SAM3 delegation failed: {e}")

        # Fallback: return empty mask + bboxes
        # This allows the workflow to continue even if SAM3 isn't working
        print("[SvedkaTalks] SAM3_Detect: no SAM3 implementation available. "
              "Install ComfyUI-SAM3 and disable its auto-discovery.")
        H, W = image.shape[-2], image.shape[-1]
        N = image.shape[0]
        empty_mask = torch.zeros(N, H, W, dtype=torch.float32)
        return (empty_mask, [],)


# ===========================================================================
# NODE MAPPINGS
# ===========================================================================

NODE_CLASS_MAPPINGS = {
    "SAM3_Detect": STSAM3Detect,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM3_Detect": "ST: SAM3 Detect",
}
