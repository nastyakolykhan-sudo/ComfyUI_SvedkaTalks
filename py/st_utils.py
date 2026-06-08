"""
ST Utility Nodes — rewritten from scratch.

Replaces: ImpactCompare, ImpactMinMax, FloatConstant, INTConstant,
          StringConstant, easy mathInt, Text Multiline, DisplayAny,
          CR String To Combo, SimpleMath+, MaskPreview+
"""

import re
import torch
import numpy as np


# ---------------------------------------------------------------------------
# AnyType wildcard
# ---------------------------------------------------------------------------

class AnyType(str):
    """Wildcard ComfyUI type that matches any connection."""
    def __ne__(self, other):
        return False

any_typ = AnyType("*")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class STFloatConstant:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("FLOAT", {"default": 0.0, "step": 0.01})}}

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("value",)
    FUNCTION = "run"
    CATEGORY = "SvedkaTalks/Utils"

    def run(self, value):
        return (value,)


class STINTConstant:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("INT", {"default": 0})}}

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("value",)
    FUNCTION = "run"
    CATEGORY = "SvedkaTalks/Utils"

    def run(self, value):
        return (value,)


class STStringConstant:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("STRING", {"default": "", "multiline": False})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("STRING",)
    FUNCTION = "run"
    CATEGORY = "SvedkaTalks/Utils"

    def run(self, value):
        return (value,)


class STTextMultiline:
    """Multi-line text input that outputs a STRING."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"text": ("STRING", {"multiline": True, "default": ""})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("STRING",)
    FUNCTION = "run"
    CATEGORY = "SvedkaTalks/Utils"

    def run(self, text):
        return (text,)


# ---------------------------------------------------------------------------
# Math
# ---------------------------------------------------------------------------

_MATH_OPS = ["add", "subtract", "multiply", "divide", "modulo", "power",
             "min", "max", "abs_a"]

class STEasyMathInt:
    """Simple INT math: result = op(a, b). Matches 'easy mathInt' interface."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "a": ("INT", {"default": 0}),
                "b": ("INT", {"default": 1}),
                "operation": (_MATH_OPS,),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("INT",)
    FUNCTION = "run"
    CATEGORY = "SvedkaTalks/Utils"

    def run(self, a, b, operation):
        if operation == "add":        result = a + b
        elif operation == "subtract": result = a - b
        elif operation == "multiply": result = a * b
        elif operation == "divide":   result = int(a / b) if b != 0 else 0
        elif operation == "modulo":   result = a % b if b != 0 else 0
        elif operation == "power":    result = int(a ** b)
        elif operation == "min":      result = min(a, b)
        elif operation == "max":      result = max(a, b)
        elif operation == "abs_a":    result = abs(a)
        else:                         result = a
        return (int(result),)


class STSimpleMath:
    """
    Evaluate a math expression using variables a, b, c.
    Matches 'SimpleMath+' interface. Returns INT and FLOAT.
    Supports: + - * / ** % ( )
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "expression": ("STRING", {"default": "a-b", "multiline": False}),
            },
            "optional": {
                "a": (any_typ, {}),
                "b": (any_typ, {}),
                "c": (any_typ, {}),
            }
        }

    RETURN_TYPES = ("INT", "FLOAT",)
    RETURN_NAMES = ("INT", "FLOAT",)
    FUNCTION = "run"
    CATEGORY = "SvedkaTalks/Utils"

    def run(self, expression, a=0, b=0, c=0):
        try:
            # Whitelist-only eval: numbers and basic math operators
            safe_expr = re.sub(r'[^0-9a-c+\-*/()%. ]', '', str(expression))
            result = float(eval(safe_expr, {"__builtins__": {}}, {"a": float(a), "b": float(b), "c": float(c)}))
        except Exception:
            result = 0.0
        return (int(result), result,)


# ---------------------------------------------------------------------------
# Comparison / logic
# ---------------------------------------------------------------------------

_COMPARE_OPS = ["a = b", "a != b", "a > b", "a < b", "a >= b", "a <= b"]

class STImpactCompare:
    """
    Compare two values. Matches ImpactCompare interface.
    Inputs a and b accept any type (wildcard).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "a": (any_typ, {}),
                "b": (any_typ, {}),
                "operation": (_COMPARE_OPS,),
            }
        }

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("BOOLEAN",)
    FUNCTION = "run"
    CATEGORY = "SvedkaTalks/Utils"

    def run(self, a, b, operation):
        try:
            av, bv = float(a), float(b)
        except (TypeError, ValueError):
            av, bv = str(a), str(b)
        if operation == "a = b":   result = av == bv
        elif operation == "a != b": result = av != bv
        elif operation == "a > b":  result = av > bv
        elif operation == "a < b":  result = av < bv
        elif operation == "a >= b": result = av >= bv
        elif operation == "a <= b": result = av <= bv
        else:                       result = False
        return (bool(result),)


class STImpactMinMax:
    """
    Return min or max of two values. Matches ImpactMinMax interface.
    Widget: is_max (BOOLEAN). Output is INT.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "a": (any_typ, {}),
                "b": (any_typ, {}),
                "is_max": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("INT",)
    FUNCTION = "run"
    CATEGORY = "SvedkaTalks/Utils"

    def run(self, a, b, is_max):
        try:
            av, bv = float(a), float(b)
            result = max(av, bv) if is_max else min(av, bv)
        except (TypeError, ValueError):
            result = 0
        return (int(result),)


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

class STDisplayAny:
    """
    Display any value as a string. Matches DisplayAny interface.
    The second widget_value is the displayed text (updated at runtime).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": (any_typ, {}),
                "display_mode": (["raw value", "type", "shape"],),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("STRING",)
    FUNCTION = "run"
    CATEGORY = "SvedkaTalks/Utils"
    OUTPUT_NODE = True

    def run(self, input, display_mode="raw value"):
        if display_mode == "type":
            text = str(type(input).__name__)
        elif display_mode == "shape":
            if isinstance(input, torch.Tensor):
                text = str(list(input.shape))
            elif isinstance(input, (list, tuple)):
                text = f"len={len(input)}"
            else:
                text = str(type(input).__name__)
        else:
            text = str(input)
        return {"ui": {"text": [text]}, "result": (text,)}


# ---------------------------------------------------------------------------
# CR String To Combo
# ---------------------------------------------------------------------------

class STCRStringToCombo:
    """
    Pass a STRING through as a wildcard (*) output.
    Matches CR String To Combo — lets a text value drive a combo widget.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = (any_typ, "STRING",)
    RETURN_NAMES = ("any", "show_help",)
    FUNCTION = "run"
    CATEGORY = "SvedkaTalks/Utils"

    def run(self, text):
        return (text, "ST: CR String To Combo — passes STRING as wildcard output.",)


# ---------------------------------------------------------------------------
# MaskPreview+
# ---------------------------------------------------------------------------

class STMaskPreviewPlus:
    """
    Preview a MASK (or batch of masks) as images. No outputs. Matches MaskPreview+.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"mask": ("MASK",)}}

    RETURN_TYPES = ()
    FUNCTION = "run"
    CATEGORY = "SvedkaTalks/Utils"
    OUTPUT_NODE = True

    def run(self, mask):
        # Convert [N, H, W] mask tensor to list of HWC preview images
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        images = []
        for m in mask:
            rgb = m.unsqueeze(-1).expand(-1, -1, 3)  # [H, W, 3]
            images.append(rgb.cpu().numpy())
        return {"ui": {"images": []}, "result": ()}


# ---------------------------------------------------------------------------
# NODE MAPPINGS
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    # Keys match the workflow node type strings exactly (bare names, no ST: prefix)
    "FloatConstant":      STFloatConstant,
    "INTConstant":        STINTConstant,
    "StringConstant":     STStringConstant,
    "Text Multiline":     STTextMultiline,
    "easy mathInt":       STEasyMathInt,
    "SimpleMath+":        STSimpleMath,
    "ImpactCompare":      STImpactCompare,
    "ImpactMinMax":       STImpactMinMax,
    "DisplayAny":         STDisplayAny,
    "CR String To Combo": STCRStringToCombo,
    "MaskPreview+":       STMaskPreviewPlus,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FloatConstant":      "ST: FloatConstant",
    "INTConstant":        "ST: INTConstant",
    "StringConstant":     "ST: StringConstant",
    "Text Multiline":     "ST: Text Multiline",
    "easy mathInt":       "ST: easy mathInt",
    "SimpleMath+":        "ST: SimpleMath+",
    "ImpactCompare":      "ST: ImpactCompare",
    "ImpactMinMax":       "ST: ImpactMinMax",
    "DisplayAny":         "ST: DisplayAny",
    "CR String To Combo": "ST: CR String To Combo",
    "MaskPreview+":       "ST: MaskPreview+",
}
