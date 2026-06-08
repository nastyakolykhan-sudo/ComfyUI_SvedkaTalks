"""
ComfyUI_SvedkaTalks
-------------------
Consolidated custom node package. Contains only nodes used in the SvedkaTalks
workflow — sourced from LayerStyle_CUSTOM, RyanOnTheInside_CUSTOM, and
minimal extracts from SeC, SAM3, ElevenLabs, VideoHelperSuite, and WAS.

All nodes carry the "ST:" prefix so their origin is traceable.
"""

import os
import importlib
import traceback

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

_py_dir = os.path.join(os.path.dirname(__file__), "py")

for filename in sorted(os.listdir(_py_dir)):
    if not filename.endswith(".py") or filename.startswith("_") or filename == "imagefunc.py":
        continue
    module_name = filename[:-3]
    try:
        module = importlib.import_module(f".py.{module_name}", package=__name__)
        if hasattr(module, "NODE_CLASS_MAPPINGS"):
            NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
        if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS"):
            NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)
    except Exception:
        print(f"[SvedkaTalks] Failed to load {filename}:")
        traceback.print_exc()

print(f"[SvedkaTalks] Loaded {len(NODE_CLASS_MAPPINGS)} node(s).")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
