"""USIR v3.0 Cross-Modal Functors & Synesthetic Transpiler (Gap 6).

Implements:
1. UI/DOM Layout Tree Functor F_ui (extract_ui_dom_24d):
   - Maps UI/DOM component descriptors (or HTML layout strings) into 4D structural signals s_m in R^4.
   - Passes signals through Wavelet-Hermite + MCA + RQA encoder to produce 24D symplectic state z in R^24.
2. Visual Saccade Trajectory Functor F_visual (extract_visual_saccade_24d):
   - Maps visual fixations/patches (luminance_contrast, edge_density, negative_space_ratio, chromatic_jump) into s_m in R^4.
   - Passes signals through Wavelet-Hermite + MCA + RQA encoder to produce 24D symplectic state z in R^24.
3. Synesthetic Transpiler (transpile_cross_modal):
   - Transpiles any 24D .usir state vector z = (q, p) into modality-specific generation parameters:
     - "music": bpm_fluctuation, harmonic_dissonance, rest_ratio, rubato_depth, soliton_accent_gain, arch_return_strength.
     - "diffusion": cfg_scale, high_freq_noise_gain, negative_space_ratio, color_contrast_temperature, composition_symmetry, edge_sharpness.
     - "ui_css": whitespace_multiplier, contrast_ratio, border_radius_px, typography_scale_ratio, grid_irregularity, transition_duration_ms.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from usir_py.v2.wavelet_hermite_encoder import WaveletHermiteEncoder, extract_24d_v2


# ==============================================================================
# Functor Component Descriptors
# ==============================================================================

@dataclass
class UiDomNode:
    """Descriptor for a single UI/DOM layout component or element."""
    depth: float = 1.0                # Nesting depth in DOM hierarchy (>= 1.0)
    whitespace_ratio: float = 0.2     # Fraction of element bounding box as whitespace/padding in [0, 1]
    font_scale_ratio: float = 1.0     # Relative font scale ratio to base font (e.g. 1.0, 1.5, 2.5)
    card_uniformity: float = 0.5      # Consistency of border radius / background / elevation in [0, 1]
    contrast_jump: float = 0.1        # Delta luminance / color contrast relative to parent or neighbor in [0, 1]
    padding_rhythm: float = 0.5       # Adherence to spacing scale (e.g., 4/8/16/24/32px rhythm) in [0, 1]


@dataclass
class VisualSaccadePatch:
    """Descriptor for a single visual fixation patch along a saccade trajectory."""
    luminance_contrast: float = 0.2    # Local Michelson or Weber luminance contrast in [0, 1]
    edge_density: float = 0.3          # High-frequency Sobel/Canny edge density in [0, 1]
    negative_space_ratio: float = 0.4  # Fraction of uniform / untextured background in [0, 1]
    chromatic_jump: float = 0.1        # Color shift / delta-E jump from previous fixation in [0, 1]
    duration_ms: float = 200.0         # Fixation duration in milliseconds
    x: float = 0.5                     # Normalized spatial coordinate x in [0, 1]
    y: float = 0.5                     # Normalized spatial coordinate y in [0, 1]


# ==============================================================================
# Functor F_ui: UI/DOM Layout Tree Functor
# ==============================================================================

def _parse_html_to_nodes(html_str: str) -> List[UiDomNode]:
    """Lightweight deterministic DOM structure parser for raw HTML layouts."""
    tags = re.findall(r"<([a-zA-Z0-9]+)([^>]*)>(.*?)(?=<\s*/?\s*[a-zA-Z0-9]+|\Z)", html_str, re.DOTALL)
    if not tags:
        # Fallback for plain text or minimal strings
        return [UiDomNode(depth=1.0, whitespace_ratio=0.3, font_scale_ratio=1.0, card_uniformity=0.6, contrast_jump=0.1, padding_rhythm=0.6)]

    nodes: List[UiDomNode] = []
    current_depth = 1

    for tag_name, attrs, content in tags:
        tag_lower = tag_name.lower()
        if tag_lower in ("html", "body", "head", "meta", "link", "script", "style"):
            continue

        # Depth tracking
        if tag_lower in ("div", "section", "article", "main", "aside", "nav", "ul", "ol"):
            current_depth = min(current_depth + 1, 8)

        # Font scale mapping
        if tag_lower == "h1":
            font_scale = 2.5
        elif tag_lower == "h2":
            font_scale = 2.0
        elif tag_lower == "h3":
            font_scale = 1.5
        elif tag_lower in ("h4", "h5", "h6"):
            font_scale = 1.25
        elif tag_lower in ("small", "sub", "sup"):
            font_scale = 0.8
        else:
            font_scale = 1.0

        # Classes and styles inspection
        attr_lower = attrs.lower()
        whitespace_ratio = 0.2
        if any(k in attr_lower for k in ("p-", "m-", "gap-", "py-", "px-", "padding", "margin")):
            whitespace_ratio += 0.3
        if "empty" in attr_lower or not content.strip():
            whitespace_ratio += 0.4
        whitespace_ratio = min(1.0, whitespace_ratio)

        card_uniformity = 0.5
        if any(k in attr_lower for k in ("card", "rounded", "shadow", "border", "grid", "flex")):
            card_uniformity = 0.85

        contrast_jump = 0.1
        if any(k in attr_lower for k in ("bg-primary", "bg-black", "bg-accent", "text-white", "btn", "danger", "warning")):
            contrast_jump = 0.75
        elif tag_lower in ("button", "a", "mark"):
            contrast_jump = 0.60

        padding_rhythm = 0.6
        if any(k in attr_lower for k in ("gap-4", "p-4", "m-4", "p-2", "p-8")):
            padding_rhythm = 0.90

        nodes.append(
            UiDomNode(
                depth=float(current_depth),
                whitespace_ratio=float(whitespace_ratio),
                font_scale_ratio=float(font_scale),
                card_uniformity=float(card_uniformity),
                contrast_jump=float(contrast_jump),
                padding_rhythm=float(padding_rhythm),
            )
        )

    if not nodes:
        nodes.append(UiDomNode())
    return nodes


def extract_ui_dom_signals(
    nodes_or_html: Union[str, Sequence[Union[Dict[str, Any], UiDomNode]]]
) -> np.ndarray:
    """Extract (M, 4) base structural signals from UI/DOM components.

    Signal 1 (Flux): Font scale dynamic range + depth progression flow.
    Signal 2 (Lattice): Card uniformity + padding rhythm alignment.
    Signal 3 (Void [間]): Whitespace ratio + open negative space.
    Signal 4 (Shock): Contrast jump + sudden scale/depth ruptures.
    """
    if isinstance(nodes_or_html, str):
        nodes = _parse_html_to_nodes(nodes_or_html)
    else:
        nodes = []
        for item in nodes_or_html:
            if isinstance(item, UiDomNode):
                nodes.append(item)
            elif isinstance(item, dict):
                nodes.append(
                    UiDomNode(
                        depth=float(item.get("depth", 1.0)),
                        whitespace_ratio=float(item.get("whitespace_ratio", 0.2)),
                        font_scale_ratio=float(item.get("font_scale_ratio", 1.0)),
                        card_uniformity=float(item.get("card_uniformity", 0.5)),
                        contrast_jump=float(item.get("contrast_jump", 0.1)),
                        padding_rhythm=float(item.get("padding_rhythm", 0.5)),
                    )
                )
            else:
                raise TypeError(f"Unsupported UI node element type: {type(item)}")

    if not nodes:
        return np.zeros((16, 4), dtype=np.float64)

    m = len(nodes)
    signals = np.zeros((m, 4), dtype=np.float64)

    for i, n in enumerate(nodes):
        # 1. Flux: dynamic visual range
        prev_font = nodes[i - 1].font_scale_ratio if i > 0 else n.font_scale_ratio
        font_delta = abs(n.font_scale_ratio - prev_font)
        s1 = 0.40 * min(n.font_scale_ratio / 3.0, 1.0) + 0.35 * min(n.depth / 8.0, 1.0) + 0.25 * min(font_delta, 1.0)

        # 2. Lattice: grid uniformity and padding cadence
        s2 = 0.50 * np.clip(n.card_uniformity, 0.0, 1.0) + 0.50 * np.clip(n.padding_rhythm, 0.0, 1.0)

        # 3. Void [間]: whitespace breath & low-density negative space
        depth_penalty = max(0.0, 1.0 - (n.depth / 10.0))
        s3 = 0.75 * np.clip(n.whitespace_ratio, 0.0, 1.0) + 0.25 * depth_penalty

        # 4. Shock: contrast jump & abrupt visual leap
        s4 = 0.70 * np.clip(n.contrast_jump, 0.0, 1.0) + 0.30 * (1.0 - np.clip(n.card_uniformity, 0.0, 1.0) if n.contrast_jump > 0.4 else 0.0)

        signals[i, 0] = float(np.clip(s1, 0.0, 1.0))
        signals[i, 1] = float(np.clip(s2, 0.0, 1.0))
        signals[i, 2] = float(np.clip(s3, 0.0, 1.0))
        signals[i, 3] = float(np.clip(s4, 0.0, 1.0))

    return signals


def extract_ui_dom_24d(
    nodes_or_html: Union[str, Sequence[Union[Dict[str, Any], UiDomNode]]],
    use_mca: bool = True,
) -> np.ndarray:
    """Functor F_ui: Extract 24D symplectic state z = (q, p) in R^24 from UI/DOM components."""
    signals = extract_ui_dom_signals(nodes_or_html)
    encoder = WaveletHermiteEncoder()
    return encoder.encode_signals(signals, use_mca=use_mca)


# ==============================================================================
# Functor F_visual: Visual Saccade Trajectory Functor
# ==============================================================================

def extract_visual_saccade_signals(
    patches: Sequence[Union[Dict[str, Any], VisualSaccadePatch]]
) -> np.ndarray:
    """Extract (M, 4) base structural signals from visual fixation patches.

    Signal 1 (Flux): Edge density + chromatic variation + spatial movement.
    Signal 2 (Lattice): Spatial coherence + structural rhythm.
    Signal 3 (Void [間]): Negative space ratio + low-frequency background calm.
    Signal 4 (Shock): Luminance contrast jump + chromatic shock.
    """
    clean_patches: List[VisualSaccadePatch] = []
    for p in patches:
        if isinstance(p, VisualSaccadePatch):
            clean_patches.append(p)
        elif isinstance(p, dict):
            clean_patches.append(
                VisualSaccadePatch(
                    luminance_contrast=float(p.get("luminance_contrast", 0.2)),
                    edge_density=float(p.get("edge_density", 0.3)),
                    negative_space_ratio=float(p.get("negative_space_ratio", 0.4)),
                    chromatic_jump=float(p.get("chromatic_jump", 0.1)),
                    duration_ms=float(p.get("duration_ms", 200.0)),
                    x=float(p.get("x", 0.5)),
                    y=float(p.get("y", 0.5)),
                )
            )
        else:
            raise TypeError(f"Unsupported visual patch type: {type(p)}")

    if not clean_patches:
        return np.zeros((16, 4), dtype=np.float64)

    m = len(clean_patches)
    signals = np.zeros((m, 4), dtype=np.float64)

    for i, p in enumerate(clean_patches):
        # 1. Flux: edge activity + fixation movement
        if i > 0:
            dx = p.x - clean_patches[i - 1].x
            dy = p.y - clean_patches[i - 1].y
            saccade_dist = math.sqrt(dx * dx + dy * dy)
        else:
            saccade_dist = 0.1
        s1 = 0.45 * np.clip(p.edge_density, 0.0, 1.0) + 0.30 * min(saccade_dist * 1.5, 1.0) + 0.25 * np.clip(p.chromatic_jump, 0.0, 1.0)

        # 2. Lattice: pattern coherence and stability
        s2 = 0.55 * (1.0 - abs(p.edge_density - 0.5) * 1.8) + 0.45 * (1.0 - np.clip(p.chromatic_jump, 0.0, 1.0))

        # 3. Void [間]: negative space breath
        s3 = 0.80 * np.clip(p.negative_space_ratio, 0.0, 1.0) + 0.20 * (1.0 - np.clip(p.edge_density, 0.0, 1.0))

        # 4. Shock: luminance contrast leap + chromatic spike
        s4 = 0.60 * np.clip(p.luminance_contrast, 0.0, 1.0) + 0.40 * np.clip(p.chromatic_jump, 0.0, 1.0)

        signals[i, 0] = float(np.clip(s1, 0.0, 1.0))
        signals[i, 1] = float(np.clip(s2, 0.0, 1.0))
        signals[i, 2] = float(np.clip(s3, 0.0, 1.0))
        signals[i, 3] = float(np.clip(s4, 0.0, 1.0))

    return signals


def extract_visual_saccade_24d(
    patches: Sequence[Union[Dict[str, Any], VisualSaccadePatch]],
    use_mca: bool = True,
) -> np.ndarray:
    """Functor F_visual: Extract 24D symplectic state z = (q, p) in R^24 from visual fixations."""
    signals = extract_visual_saccade_signals(patches)
    encoder = WaveletHermiteEncoder()
    return encoder.encode_signals(signals, use_mca=use_mca)


# ==============================================================================
# Synesthetic Transpiler
# ==============================================================================

def transpile_cross_modal(z_24d: np.ndarray, target_modality: str) -> Dict[str, float]:
    """Transpile a 24D .usir state vector z = (q, p) into modality generation parameters.

    Args:
        z_24d: 24D numpy array with q in [0, 1]^12, p in R_{>=0}^12.
        target_modality: One of {"music", "diffusion", "ui_css"}.

    Returns:
        Modality-specific generative parameter dictionary.
    """
    z = np.asarray(z_24d, dtype=np.float64).ravel()
    if z.shape[0] < 24:
        raise ValueError(f"State vector must have 24 dimensions, got {z.shape[0]}")

    q = z[:12]
    p = z[12:24]

    # Coordinate aliases:
    # Micro (j=1): q0: Flux, q1: Lattice (Cr), q2: Void (Pr), q3: Shock (Jt)
    # Meso  (j=2): q4: Flux, q5: Lattice (Mp), q6: Void (Cs), q7: Shock (Lp)
    # Macro (j=3): q8: Flux, q9: Lattice (At/RQA), q10: Void (Ma), q11: Shock (Ru/Soliton)

    mod = target_modality.strip().lower()

    if mod == "music":
        bpm_fluctuation = float(np.clip(0.05 + 0.35 * (0.4 * q[0] + 0.3 * q[4] + 0.3 * q[8]) + 0.15 * math.tanh(p[0] + p[4]), 0.0, 1.0))
        harmonic_dissonance = float(np.clip(0.50 * (q[3] + q[7] + q[11]) / 3.0 - 0.30 * (q[1] + q[5] + q[9]) / 3.0 + 0.25, 0.0, 1.0))
        rest_ratio = float(np.clip(0.20 * q[2] + 0.30 * q[6] + 0.50 * q[10], 0.0, 1.0))
        rubato_depth = float(np.clip(0.10 + 0.50 * q[8] + 0.30 * (1.0 - q[1]) * math.tanh(p[8]), 0.0, 1.0))
        soliton_accent_gain = float(np.clip(0.70 * q[11] + 0.30 * q[7] + 0.20 * math.tanh(p[11]), 0.0, 2.0))
        arch_return_strength = float(np.clip(0.80 * q[9] + 0.20 * q[5], 0.0, 1.0))
        return {
            "bpm_fluctuation": bpm_fluctuation,
            "harmonic_dissonance": harmonic_dissonance,
            "rest_ratio": rest_ratio,
            "rubato_depth": rubato_depth,
            "soliton_accent_gain": soliton_accent_gain,
            "arch_return_strength": arch_return_strength,
        }

    elif mod == "diffusion":
        cfg_scale = float(np.clip(4.0 + 8.0 * (0.4 * q[1] + 0.4 * q[5] + 0.2 * q[9]) - 3.0 * q[10], 1.5, 14.0))
        high_freq_noise_gain = float(np.clip(0.10 + 0.65 * q[11] + 0.25 * q[3] + 0.20 * math.tanh(p[3] + p[7]), 0.0, 1.5))
        negative_space_ratio = float(np.clip(0.15 + 0.25 * q[2] + 0.25 * q[6] + 0.35 * q[10], 0.0, 1.0))
        color_contrast_temperature = float(np.clip(0.20 + 0.40 * q[3] + 0.40 * q[7] + 0.20 * q[11], 0.0, 1.0))
        composition_symmetry = float(np.clip(0.60 * q[9] + 0.40 * q[1] - 0.30 * q[3] - 0.30 * q[11] + 0.20, 0.0, 1.0))
        edge_sharpness = float(np.clip(0.50 * q[1] + 0.30 * q[5] + 0.40 * q[3] - 0.30 * q[10], 0.0, 1.0))
        return {
            "cfg_scale": cfg_scale,
            "high_freq_noise_gain": high_freq_noise_gain,
            "negative_space_ratio": negative_space_ratio,
            "color_contrast_temperature": color_contrast_temperature,
            "composition_symmetry": composition_symmetry,
            "edge_sharpness": edge_sharpness,
        }

    elif mod in ("ui_css", "uicss", "ui"):
        whitespace_multiplier = float(np.clip(0.80 + 2.50 * (0.2 * q[2] + 0.3 * q[6] + 0.5 * q[10]), 0.5, 4.0))
        contrast_ratio = float(np.clip(3.0 + 12.0 * (0.5 * q[1] + 0.3 * q[3] + 0.2 * q[11]), 3.0, 18.0))
        border_radius_px = float(np.clip(16.0 * (0.4 * q[2] + 0.4 * q[10] + 0.2 * q[9]) - 10.0 * (q[1] + q[11]), 0.0, 32.0))
        typography_scale_ratio = float(np.clip(1.10 + 0.45 * (0.4 * q[0] + 0.3 * q[8] + 0.3 * q[11]), 1.05, 1.60))
        grid_irregularity = float(np.clip(0.45 * q[3] + 0.30 * q[7] + 0.40 * q[11] - 0.35 * q[1] + 0.10, 0.0, 1.0))
        transition_duration_ms = float(np.clip(150.0 + 500.0 * q[10] - 250.0 * q[11], 50.0, 800.0))
        return {
            "whitespace_multiplier": whitespace_multiplier,
            "contrast_ratio": contrast_ratio,
            "border_radius_px": border_radius_px,
            "typography_scale_ratio": typography_scale_ratio,
            "grid_irregularity": grid_irregularity,
            "transition_duration_ms": transition_duration_ms,
        }

    else:
        raise ValueError(f"Unknown target modality: {target_modality}. Must be 'music', 'diffusion', or 'ui_css'.")
