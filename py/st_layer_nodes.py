"""
ST Layer Nodes
--------------
Copied from ComfyUI_LayerStyle_CUSTOM and ComfyUI_LayerStyle, with ST: prefix.

Includes:
  - ST: OuterGlow V2 Fast   (from outer_glow_v2_fast.py)
  - ST: GrowMaskWithBlur CUSTOM (from grow_mask_with_blur_custom.py)
  - ST: MaskToValue         (from mask_to_value.py)
  - ST: HSV Value           (from color_to_HSVvalue.py)
  - ST: GetColorToneV2      (new — uses imagefunc helpers)
  - ST: GetImageRangeFromBatch (new)
  - ST: MaskFix+            (simple morphology rewrite)
"""

import copy
import torch
import os
import torch.nn.functional as F
import numpy as np
import scipy.ndimage
from concurrent.futures import ThreadPoolExecutor
from itertools import groupby
from PIL import Image
from tqdm import tqdm

from .imagefunc import (
    log, tensor2pil, pil2tensor, image2mask,
    step_value, BLEND_MODES,
    AnyType,
    get_image_color_tone, get_image_color_average,
    Hex_to_HSV_255level, RGB_to_Hex,
)

any_typ = AnyType("*")


# ===========================================================================
# OuterGlow V2 Fast
# ===========================================================================

def _hex_to_rgb(inhex: str) -> tuple:
    inhex = inhex.strip()
    if not inhex.startswith('#'):
        raise ValueError(f'Invalid Hex Code: {inhex}')
    if len(inhex) == 4:
        inhex = "#" + "".join([c * 2 for c in inhex[1:]])
    return (int(inhex[1:3], 16), int(inhex[3:5], 16), int(inhex[5:7], 16))


def _build_diamond(radius: int) -> np.ndarray:
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return (np.abs(x) + np.abs(y) <= radius).astype(np.uint8)


def _expand_mask_np(mask_np, grow, blur):
    if grow != 0:
        footprint = _build_diamond(abs(grow))
        if grow < 0:
            out = scipy.ndimage.grey_erosion(mask_np, footprint=footprint)
        else:
            out = scipy.ndimage.grey_dilation(mask_np, footprint=footprint)
    else:
        out = mask_np
    if blur > 0:
        out = scipy.ndimage.gaussian_filter(out, sigma=blur / 3.0)
    return np.clip(out, 0.0, 1.0)


def _alpha_composite_np(canvas, overlay, alpha):
    a = alpha[:, :, np.newaxis]
    return canvas * (1.0 - a) + overlay * a


def _process_frame_cpu(args):
    (canvas_np, layer_np, mask_np,
     _glow_range, _brightness, _blur, _opacity,
     blend_mode_fn, glow_rgb, light_rgb) = args

    H, W = canvas_np.shape[:2]
    blur_factor = _blur / 20.0
    canvas_rgba = np.empty((H, W, 4), dtype=float)
    canvas_rgba[:, :, :3] = canvas_np.astype(float)
    canvas_rgba[:, :, 3] = 255.0
    source_rgba = np.empty((H, W, 4), dtype=float)
    source_rgba[:, :, 3] = 255.0
    grow = _glow_range
    for x in range(_brightness):
        blur_val = grow * blur_factor
        t = x / _brightness
        source_rgba[:, :, 0] = glow_rgb[0] + (light_rgb[0] - glow_rgb[0]) * t
        source_rgba[:, :, 1] = glow_rgb[1] + (light_rgb[1] - glow_rgb[1]) * t
        source_rgba[:, :, 2] = glow_rgb[2] + (light_rgb[2] - glow_rgb[2]) * t
        alpha = _expand_mask_np(mask_np, grow, blur_val)
        op = step_value(1, _opacity, _brightness, x) / 100.0
        blended_rgba = blend_mode_fn(canvas_rgba, source_rgba, op)
        canvas_rgba[:, :, :3] = _alpha_composite_np(canvas_rgba[:, :, :3], blended_rgba[:, :, :3], alpha)
        grow = grow - int(_glow_range / _brightness)
    result_rgb = _alpha_composite_np(canvas_rgba[:, :, :3], layer_np.astype(float), mask_np)
    return np.clip(result_rgb, 0, 255).astype(np.uint8)


_GPU_BLEND_MODES = {
    'normal':            lambda b, s, op: b * (1 - op) + s * op,
    'linear dodge(add)': lambda b, s, op: torch.clamp(b + s * op, 0, 1),
    'screen':            lambda b, s, op: 1 - (1 - b) * (1 - s * op),
    'lighten':           lambda b, s, op: torch.maximum(b, s * op),
    'multiply':          lambda b, s, op: b * (s * op) + b * (1 - op),
    'color dodge':       lambda b, s, op: torch.clamp(b / (1 - s * op + 1e-6), 0, 1),
    'dodge':             lambda b, s, op: torch.clamp(b / (1 - s * op + 1e-6), 0, 1),
    'hard light':        lambda b, s, op: torch.where(s * op < 0.5, 2 * b * s * op, 1 - 2 * (1 - b) * (1 - s * op)),
    'linear light':      lambda b, s, op: torch.clamp(b + 2 * s * op - 1, 0, 1),
    'overlay':           lambda b, s, op: torch.where(b < 0.5, 2 * b * s * op, 1 - 2 * (1 - b) * (1 - s * op)),
    'darken':            lambda b, s, op: torch.minimum(b, s * op),
    'difference':        lambda b, s, op: torch.abs(b - s * op),
    'exclusion':         lambda b, s, op: b + s * op - 2 * b * s * op,
    'subtract':          lambda b, s, op: torch.clamp(b - s * op, 0, 1),
    'divide':            lambda b, s, op: torch.clamp(b / (s * op + 1e-6), 0, 1),
}


def _expand_mask_gpu(mask, grow, blur, device):
    if grow != 0:
        radius = abs(grow)
        ksize = 2 * radius + 1
        padding = radius
        if grow > 0:
            mask = F.max_pool2d(mask, kernel_size=ksize, stride=1, padding=padding)
        else:
            mask = 1.0 - F.max_pool2d(1.0 - mask, kernel_size=ksize, stride=1, padding=padding)
    if blur > 0:
        sigma = blur / 3.0
        k = int(6 * sigma + 1) | 1
        k = max(k, 3)
        coords = torch.arange(k, dtype=torch.float32, device=device) - k // 2
        gauss = torch.exp(-0.5 * (coords / sigma) ** 2)
        gauss = gauss / gauss.sum()
        kh = gauss.view(1, 1, 1, k)
        mask = F.conv2d(mask, kh.expand(1, 1, 1, k), padding=(0, k // 2))
        kv = gauss.view(1, 1, k, 1)
        mask = F.conv2d(mask, kv.expand(1, 1, k, 1), padding=(k // 2, 0))
    return mask.clamp(0, 1)


def _process_batch_gpu(b_tensors, l_tensors, masks_tensor,
                       glow_range_list, brightness_list, blur_list, opacity_list,
                       blend_mode, glow_rgb, light_rgb, device):
    N = len(b_tensors)
    canvas = torch.cat(b_tensors, dim=0).to(device)
    layer  = torch.cat(l_tensors, dim=0).to(device)
    mask_list = [m.to(device).unsqueeze(0).unsqueeze(0) for m in masks_tensor]
    blend_fn = _GPU_BLEND_MODES[blend_mode]
    glow_t  = torch.tensor(glow_rgb,  dtype=torch.float32, device=device) / 255.0  # [3]
    light_t = torch.tensor(light_rgb, dtype=torch.float32, device=device) / 255.0  # [3]
    results = []
    with torch.no_grad():
        for i in range(N):
            c = canvas[i]; l = layer[i]; m = mask_list[i]
            _glow_range = glow_range_list[i] if i < len(glow_range_list) else glow_range_list[-1]
            _brightness = brightness_list[i] if i < len(brightness_list) else brightness_list[-1]
            _blur       = blur_list[i]       if i < len(blur_list)       else blur_list[-1]
            _opacity    = opacity_list[i]    if i < len(opacity_list)    else opacity_list[-1]
            blur_factor = _blur / 20.0
            grow = _glow_range
            result = c.clone()
            for x in range(_brightness):
                blur_val = grow * blur_factor
                t = x / _brightness                                    # matches original exactly
                color = glow_t + (light_t - glow_t) * t               # [3]
                alpha = _expand_mask_gpu(m, grow, blur_val, device)
                alpha_hw = alpha[0, 0]
                op = step_value(1, _opacity, _brightness, x) / 100.0
                s = color.view(1, 1, 3).expand_as(result)
                blended = blend_fn(result, s, op)
                a = alpha_hw.unsqueeze(-1)
                result = result * (1 - a) + blended * a
                grow = grow - int(_glow_range / _brightness)
            m_hw = mask_list[i][0, 0]
            a = m_hw.unsqueeze(-1)
            result = result * (1 - a) + l * a
            results.append(result.unsqueeze(0).cpu())
    return results


class STOuterGlowV2Fast:

    def __init__(self):
        self.NODE_NAME = 'ST: OuterGlow V2 Fast'

    @classmethod
    def INPUT_TYPES(cls):
        modes = copy.copy(BLEND_MODES)
        chop_mode_list = ["screen", "linear dodge(add)", "color dodge", "lighten",
                          "dodge", "hard light", "linear light"]
        for i in chop_mode_list:
            modes.pop(i)
        chop_mode_list.extend(list(modes.keys()))
        return {
            "required": {
                "background_image": ("IMAGE",),
                "layer_image": ("IMAGE",),
                "invert_mask": ("BOOLEAN", {"default": True}),
                "blend_mode": (chop_mode_list,),
                "light_color": ("STRING", {"default": "#FFBF30"}),
                "glow_color": ("STRING", {"default": "#FE0000"}),
                "use_gpu": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "layer_mask": ("MASK",),
                "opacity": ("INT", {"default": 100, "min": 0, "max": 100, "step": 1}),
                "brightness": ("INT", {"default": 5, "min": 2, "max": 20, "step": 1}),
                "glow_range": ("INT", {"default": 48, "min": -9999, "max": 9999, "step": 1}),
                "blur": ("INT", {"default": 25, "min": 0, "max": 9999, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'outer_glow_v2_fast'
    CATEGORY = 'SvedkaTalks/Layer'

    def outer_glow_v2_fast(self, background_image, layer_image,
                           invert_mask, blend_mode, light_color, glow_color,
                           use_gpu=False, opacity=100, brightness=5,
                           glow_range=48, blur=25, layer_mask=None):
        b_images, l_images, l_masks = [], [], []
        for b in background_image:
            b_images.append(torch.unsqueeze(b, 0))
        for l in layer_image:
            l_images.append(torch.unsqueeze(l, 0))
            m = tensor2pil(l)
            if m.mode == 'RGBA':
                l_masks.append(m.split()[-1])
        if layer_mask is not None:
            if layer_mask.dim() == 2:
                layer_mask = torch.unsqueeze(layer_mask, 0)
            l_masks = []
            for m in layer_mask:
                if invert_mask:
                    m = 1 - m
                l_masks.append(tensor2pil(torch.unsqueeze(m, 0)).convert('L'))
        if len(l_masks) == 0:
            log(f"Error: {self.NODE_NAME} no mask found.", message_type='error')
            return (background_image,)

        max_batch = max(len(b_images), len(l_images), len(l_masks))
        glow_range_list = glow_range if isinstance(glow_range, (list, tuple)) else [glow_range] * max_batch
        brightness_list = brightness if isinstance(brightness, (list, tuple)) else [brightness] * max_batch
        blur_list       = blur       if isinstance(blur,       (list, tuple)) else [blur]       * max_batch
        opacity_list    = opacity    if isinstance(opacity,    (list, tuple)) else [opacity]    * max_batch
        glow_rgb = _hex_to_rgb(glow_color)
        light_rgb = _hex_to_rgb(light_color)

        if use_gpu and blend_mode in _GPU_BLEND_MODES:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if device.type == 'cuda':
                masks_tensor, b_tensors, l_tensors = [], [], []
                for i in range(max_batch):
                    bg  = b_images[i] if i < len(b_images) else b_images[-1]
                    lay = l_images[i] if i < len(l_images) else l_images[-1]
                    mp  = l_masks[i]  if i < len(l_masks)  else l_masks[-1]
                    b_tensors.append(bg); l_tensors.append(lay)
                    masks_tensor.append(torch.from_numpy(np.array(mp, dtype=np.float32) / 255.0))
                ret = _process_batch_gpu(b_tensors, l_tensors, masks_tensor,
                                         glow_range_list, brightness_list, blur_list, opacity_list,
                                         blend_mode, glow_rgb, light_rgb, device)
                log(f"{self.NODE_NAME} {len(ret)} frame(s) on GPU.", message_type='finish')
                return (torch.cat(ret, dim=0),)

        blend_mode_fn = BLEND_MODES[blend_mode]
        frame_args = []
        for i in range(max_batch):
            bg_pil   = tensor2pil(b_images[i] if i < len(b_images) else b_images[-1]).convert('RGB')
            lay_pil  = tensor2pil(l_images[i] if i < len(l_images) else l_images[-1]).convert('RGB')
            mask_pil = l_masks[i] if i < len(l_masks) else l_masks[-1]
            if mask_pil.size != lay_pil.size:
                mask_pil = Image.new('L', lay_pil.size, 'white')
            _gr = glow_range_list[i] if i < len(glow_range_list) else glow_range_list[-1]
            _br = brightness_list[i] if i < len(brightness_list) else brightness_list[-1]
            _bl = blur_list[i]       if i < len(blur_list)       else blur_list[-1]
            _op = opacity_list[i]    if i < len(opacity_list)    else opacity_list[-1]
            frame_args.append((
                np.array(bg_pil, dtype=np.float32),
                np.array(lay_pil, dtype=np.float32),
                np.array(mask_pil, dtype=np.float32) / 255.0,
                _gr, _br, _bl, _op,
                blend_mode_fn, glow_rgb, light_rgb
            ))
        with ThreadPoolExecutor(max_workers=min(max_batch, 8)) as executor:
            results = list(executor.map(_process_frame_cpu, frame_args))
        ret_images = [torch.from_numpy(r.astype(np.float32) / 255.0).unsqueeze(0) for r in results]
        log(f"{self.NODE_NAME} {len(ret_images)} frame(s) on CPU.", message_type='finish')
        return (torch.cat(ret_images, dim=0),)


# ===========================================================================
# GrowMaskWithBlur CUSTOM
# ===========================================================================

_MAX_RESOLUTION = 8192
_main_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _gaussian_kernel_1d(radius: float, device):
    sigma = max(radius / 3.0, 0.01)
    size = int(radius) * 2 + 1
    x = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def _gaussian_blur_batch_gpu(batch, blur_list):
    device = batch.device
    n = batch.shape[0]
    indexed = [(i, blur_list[i] if i < len(blur_list) else blur_list[-1]) for i in range(n)]
    out_frames = []
    for r, group in groupby(indexed, key=lambda x: round(x[1], 2)):
        indices = [g[0] for g in group]
        if r <= 0:
            out_frames.extend([(i, batch[i]) for i in indices])
            continue
        k1d = _gaussian_kernel_1d(r, device)
        kh = k1d.view(1, 1, 1, -1)
        kv = k1d.view(1, 1, -1, 1)
        pad = len(k1d) // 2
        sub = batch[indices].unsqueeze(1)
        sub = F.pad(sub, (pad, pad, 0, 0), mode='reflect')
        sub = F.conv2d(sub, kh)
        sub = F.pad(sub, (0, 0, pad, pad), mode='reflect')
        sub = F.conv2d(sub, kv)
        sub = sub.squeeze(1)
        out_frames.extend([(indices[j], sub[j]) for j in range(len(indices))])
    out_frames.sort(key=lambda x: x[0])
    return torch.stack([f for _, f in out_frames], dim=0)


def _make_diamond_kernel(radius, device):
    size = 2 * radius + 1
    k = torch.zeros(size, size, dtype=torch.float32, device=device)
    for y in range(size):
        for x in range(size):
            if abs(y - radius) + abs(x - radius) <= radius:
                k[y, x] = 1.0
    return k


def _dilate_erode_gpu(output, expand, tapered_corners):
    r = abs(expand)
    device = output.device
    if tapered_corners:
        kernel = _make_diamond_kernel(r, device)
    else:
        kernel = torch.ones(2 * r + 1, 2 * r + 1, dtype=torch.float32, device=device)
    if expand > 0:
        try:
            import kornia.morphology as morph
            return morph.dilation(output, kernel)
        except Exception:
            return F.max_pool2d(output, kernel_size=2*r+1, stride=1, padding=r)
    else:
        try:
            import kornia.morphology as morph
            return morph.erosion(output, kernel)
        except Exception:
            inv = 1.0 - output
            inv = F.max_pool2d(inv, kernel_size=2*r+1, stride=1, padding=r)
            return 1.0 - inv


class STGrowMaskWithBlurCustom:

    def __init__(self):
        self.NODE_NAME = 'ST: GrowMaskWithBlur CUSTOM'

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "expand": ("INT", {"default": 0, "min": -_MAX_RESOLUTION, "max": _MAX_RESOLUTION, "step": 1}),
                "incremental_expandrate": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "tapered_corners": ("BOOLEAN", {"default": True}),
                "flip_input": ("BOOLEAN", {"default": False}),
                "blur_radius": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "blur_smoothing": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 20.0, "step": 0.5}),
                "lerp_alpha": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "decay_factor": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "fill_holes": ("BOOLEAN", {"default": False}),
            },
        }

    CATEGORY = 'SvedkaTalks/Layer'
    RETURN_TYPES = ("MASK", "MASK",)
    RETURN_NAMES = ("mask", "mask_inverted",)
    FUNCTION = "expand_mask"

    def expand_mask(self, mask, expand, incremental_expandrate, tapered_corners,
                    flip_input, blur_radius, blur_smoothing, lerp_alpha, decay_factor,
                    fill_holes=False):
        if flip_input:
            mask = 1.0 - mask
        growmask = mask.reshape((-1, mask.shape[-2], mask.shape[-1]))
        n_frames = growmask.shape[0]
        blur_list   = [float(v) for v in blur_radius] if isinstance(blur_radius, (list, tuple)) else [float(blur_radius)] * n_frames
        expand_list = [int(v)   for v in expand]      if isinstance(expand,      (list, tuple)) else [int(expand)]        * n_frames
        if blur_smoothing > 0 and len(blur_list) > 1:
            arr = np.array(blur_list, dtype=np.float32)
            arr = scipy.ndimage.gaussian_filter1d(arr, sigma=blur_smoothing)
            blur_list = arr.tolist()
        out = []
        previous_output = None
        for i, m in enumerate(tqdm(growmask, desc="Expanding/Contracting Mask")):
            current_expand = expand_list[i] if i < len(expand_list) else expand_list[-1]
            output = m.unsqueeze(0).unsqueeze(0).to(_main_device)
            if current_expand != 0 and output.max() > 0:
                output = _dilate_erode_gpu(output, current_expand, tapered_corners)
            output = output.squeeze(0).squeeze(0)
            if fill_holes:
                filled_np = scipy.ndimage.binary_fill_holes((output > 0).cpu().numpy())
                output = torch.from_numpy(filled_np.astype(np.float32)).to(output.device)
            if lerp_alpha < 1.0 and previous_output is not None:
                output = lerp_alpha * output + (1 - lerp_alpha) * previous_output
            if decay_factor < 1.0 and previous_output is not None:
                output += decay_factor * previous_output
                if output.max() > 0:
                    output = output / output.max()
            previous_output = output
            out.append(output.cpu())
        stacked = torch.stack(out, dim=0)
        if any(r > 0 for r in blur_list):
            gpu = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            result = _gaussian_blur_batch_gpu(stacked.to(gpu), blur_list).cpu()
        else:
            result = stacked
        log(f"{self.NODE_NAME} Processed {n_frames} frame(s).", message_type='finish')
        return (result, 1.0 - result,)


# ===========================================================================
# MaskToValue
# ===========================================================================

class STMaskToValue:

    def __init__(self):
        self.NODE_NAME = 'ST: MaskToValue'

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "min_val": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.01}),
                "max_val": ("FLOAT", {"default": 48.0, "min": -9999.0, "max": 9999.0, "step": 0.01}),
            },
            "optional": {
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("FLOAT", "INT")
    RETURN_NAMES = ("values_float", "values_int")
    FUNCTION = "mask_to_value"
    CATEGORY = "SvedkaTalks/Layer"

    def mask_to_value(self, mask, min_val, max_val, threshold=0.5):
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)                          # [N, H, W]
        with torch.no_grad():
            if threshold > 0.0:
                coverage = (mask > threshold).float().mean(dim=(1, 2))   # [N] — one GPU op
            else:
                coverage = mask.mean(dim=(1, 2))                          # [N]
            values = (min_val + (max_val - min_val) * coverage).clamp(
                min(min_val, max_val), max(min_val, max_val)
            )
        vals = values.tolist()                                # single CPU transfer
        if len(vals) == 1:
            return (vals[0], int(round(vals[0])))
        return (vals, [int(round(v)) for v in vals])


# ===========================================================================
# HSV Value
# ===========================================================================

class STHSVValue:

    def __init__(self):
        self.NODE_NAME = 'ST: HSV Value'

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"color_value": (any_typ, {})},
        }

    RETURN_TYPES = ("INT", "INT", "INT")
    RETURN_NAMES = ("H", "S", "V")
    FUNCTION = 'run'
    CATEGORY = 'SvedkaTalks/Layer'

    def run(self, color_value):
        H, S, V = 0, 0, 0
        if isinstance(color_value, str):
            H, S, V = Hex_to_HSV_255level(color_value)
        elif isinstance(color_value, tuple):
            H, S, V = Hex_to_HSV_255level(RGB_to_Hex(color_value))
        else:
            log(f"{self.NODE_NAME}: color_value must be str or tuple.", message_type="error")
        return (H, S, V,)


# ===========================================================================
# GetColorToneV2
# ===========================================================================

_COLOR_METHODS = ["average", "dominant"]
_COLOR_REGIONS = ["entire", "background", "subject", "mask"]
_REMOVE_BG     = ["none", "original", "purify", "weak", "strong"]

class STGetColorToneV2:
    """
    Get the dominant or average color of an image (optionally masked).
    Matches LayerUtility: GetColorToneV2 interface.
    """

    def __init__(self):
        self.NODE_NAME = 'ST: GetColorToneV2'

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "method": (_COLOR_METHODS,),
                "region": (_COLOR_REGIONS,),
                "remove_bkg": (_REMOVE_BG,),
                "invert_mask": ("BOOLEAN", {"default": False}),
                "blur_mask": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1}),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "LIST", "MASK",)
    RETURN_NAMES = ("image", "color_in_hex", "HSV color in list", "mask",)
    FUNCTION = "run"
    CATEGORY = "SvedkaTalks/Layer"

    def run(self, image, method, region, remove_bkg, invert_mask, blur_mask, mask=None):
        ret_images, ret_hex, ret_hsv = [], [], []

        # Normalise mask dims
        if mask is not None and mask.dim() == 2:
            mask = mask.unsqueeze(0)

        n = image.shape[0]
        H, W = image.shape[1], image.shape[2]

        # Pre-blur all mask frames in one GPU batch if needed
        blurred_masks = None
        if mask is not None and blur_mask > 0:
            m = mask.clone()
            if invert_mask:
                m = 1.0 - m
            if m.dim() == 2:
                m = m.unsqueeze(0)
            sigma = blur_mask / 3.0
            k = max(int(6 * sigma + 1) | 1, 3)
            coords = torch.arange(k, dtype=torch.float32) - k // 2
            gauss = torch.exp(-0.5 * (coords / sigma) ** 2)
            gauss = (gauss / gauss.sum()).to(m.device)
            mb = m.unsqueeze(1)                               # [N,1,H,W]
            mb = F.conv2d(mb, gauss.view(1,1,1,k), padding=(0, k//2))
            mb = F.conv2d(mb, gauss.view(1,1,k,1), padding=(k//2, 0))
            blurred_masks = mb.squeeze(1).clamp(0, 1)        # [N,H,W]

        for i in range(n):
            img_pil = tensor2pil(image[i]).convert("RGB")

            frame_mask = None
            if mask is not None:
                m_idx = min(i, mask.shape[0] - 1)
                if blurred_masks is not None:
                    m_t = blurred_masks[min(i, blurred_masks.shape[0]-1)]
                else:
                    m_t = mask[m_idx]
                    if invert_mask:
                        m_t = 1.0 - m_t
                frame_mask = tensor2pil(m_t.unsqueeze(0)).convert("L")

            if region == "subject" and frame_mask is not None:
                active_mask = frame_mask
            elif region == "background" and frame_mask is not None:
                active_mask = Image.fromarray(255 - np.array(frame_mask))
            else:
                active_mask = frame_mask

            if method == "average":
                hex_color = get_image_color_average(img_pil, active_mask)
            else:
                hex_color = get_image_color_tone(img_pil, active_mask)

            hsv = Hex_to_HSV_255level(hex_color)

            # Swatch: create as tensor directly — no PIL roundtrip
            r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
            swatch = torch.tensor([r/255.0, g/255.0, b/255.0]).view(1, 1, 3).expand(H, W, 3).unsqueeze(0)
            ret_images.append(swatch)
            ret_hex.append(hex_color)
            ret_hsv.append(hsv)

        out_mask = mask if mask is not None else torch.zeros(n, image.shape[-2], image.shape[-1])
        # Return last hex / hsv for single-frame compat; list for batch
        hex_out = ret_hex[0] if n == 1 else ret_hex
        hsv_out = ret_hsv[0] if n == 1 else ret_hsv
        return (torch.cat(ret_images, dim=0), hex_out, hsv_out, out_mask,)


# ===========================================================================
# GetImageRangeFromBatch
# ===========================================================================

class STGetImageRangeFromBatch:
    """
    Extract a contiguous slice of frames from an IMAGE (and optionally MASK) batch.
    Matches GetImageRangeFromBatch interface:
      inputs : images (IMAGE), masks (MASK, optional), start_index (INT connectable)
      widgets: start_index (INT, default 0), count (INT, default 1)
      outputs: IMAGE, MASK
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "start_index": ("INT", {"default": 0, "min": 0, "max": 9999, "step": 1}),
                "count": ("INT", {"default": 1, "min": 1, "max": 9999, "step": 1}),
            },
            "optional": {
                "masks": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("IMAGE", "MASK",)
    FUNCTION = "run"
    CATEGORY = "SvedkaTalks/Layer"

    def run(self, images, start_index, count, masks=None):
        n = images.shape[0]
        start = max(0, min(start_index, n - 1))
        end   = min(start + count, n)
        out_images = images[start:end]
        if masks is not None:
            if masks.dim() == 2:
                masks = masks.unsqueeze(0)
            nm = masks.shape[0]
            ms = max(0, min(start, nm - 1))
            me = min(ms + count, nm)
            out_masks = masks[ms:me]
        else:
            out_masks = torch.zeros(out_images.shape[0], out_images.shape[-2], out_images.shape[-1])
        return (out_images, out_masks,)


# ===========================================================================
# MaskFix+
# ===========================================================================

class STMaskFixPlus:
    """
    Mask cleanup: erode/dilate, blur, fill holes, invert.
    Matches MaskFix+ interface (5 widget values: erode, dilate, blur, fill_holes, invert).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "erode": ("INT", {"default": 0, "min": 0, "max": 128, "step": 1}),
                "dilate": ("INT", {"default": 0, "min": 0, "max": 128, "step": 1}),
                "blur": ("INT", {"default": 5, "min": 0, "max": 128, "step": 1}),
                "fill_holes": ("INT", {"default": 0, "min": 0, "max": 1, "step": 1}),
                "invert": ("INT", {"default": 0, "min": 0, "max": 1, "step": 1}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("MASK",)
    FUNCTION = "run"
    CATEGORY = "SvedkaTalks/Layer"

    def run(self, mask, erode, dilate, blur, fill_holes, invert):
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        # Pre-build footprints once — same for every frame
        fp_erode  = _build_diamond(erode)  if erode  > 0 else None
        fp_dilate = _build_diamond(dilate) if dilate > 0 else None
        sigma     = blur / 3.0             if blur   > 0 else 0.0

        frames_np = mask.cpu().numpy().astype(np.float32)  # [N, H, W]

        def _process(np_m):
            if fp_erode  is not None: np_m = scipy.ndimage.grey_erosion(np_m,  footprint=fp_erode)
            if fp_dilate is not None: np_m = scipy.ndimage.grey_dilation(np_m, footprint=fp_dilate)
            if sigma > 0:             np_m = scipy.ndimage.gaussian_filter(np_m, sigma=sigma)
            if fill_holes:            np_m = scipy.ndimage.binary_fill_holes(np_m > 0.5).astype(np.float32)
            np_m = np.clip(np_m, 0.0, 1.0)
            if invert:                np_m = 1.0 - np_m
            return np_m

        n = frames_np.shape[0]
        if n == 1:
            out = [_process(frames_np[0])]
        else:
            workers = min(n, (os.cpu_count() or 4))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                out = list(pool.map(_process, frames_np))

        return (torch.from_numpy(np.stack(out, axis=0)),)


# ===========================================================================
# NODE MAPPINGS
# ===========================================================================

NODE_CLASS_MAPPINGS = {
    "LayerStyle: OuterGlow V2 Fast":        STOuterGlowV2Fast,
    "LayerUtility: GrowMaskWithBlur CUSTOM": STGrowMaskWithBlurCustom,
    "LayerUtility: MaskToValue":            STMaskToValue,
    "LayerUtility: HSV Value":              STHSVValue,
    "LayerUtility: GetColorToneV2":         STGetColorToneV2,
    "GetImageRangeFromBatch":               STGetImageRangeFromBatch,
    "MaskFix+":                             STMaskFixPlus,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerStyle: OuterGlow V2 Fast":        "ST: OuterGlow V2 Fast",
    "LayerUtility: GrowMaskWithBlur CUSTOM": "ST: GrowMaskWithBlur CUSTOM",
    "LayerUtility: MaskToValue":            "ST: MaskToValue",
    "LayerUtility: HSV Value":              "ST: HSV Value",
    "LayerUtility: GetColorToneV2":         "ST: GetColorToneV2",
    "GetImageRangeFromBatch":               "ST: GetImageRangeFromBatch",
    "MaskFix+":                             "ST: MaskFix+",
}
