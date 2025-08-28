from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

from lxml import etree as ET

# Adobe Camera Raw (Lightroom) namespace
NS_CRS = "http://ns.adobe.com/camera-raw-settings/1.0/"
NS_RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
NS_X = "adobe:ns:meta/"

NSMAP = {
    "x": NS_X,
    "rdf": NS_RDF,
    "crs": NS_CRS,
}

# Supported v1 keys (sliders and toggles). See project spec.
SUPPORTED_CRS_KEYS = [
    # Basic
    "Exposure2012",
    "Contrast2012",
    "Highlights2012",
    "Shadows2012",
    "Whites2012",
    "Blacks2012",
    # Presence
    "Clarity2012",
    "Texture",
    "Dehaze",
    # Color
    "Vibrance",
    "Saturation",
    # WB
    "Temperature",
    "Tint",
    # HSL (per-channel)
    "HueAdjustmentRed",
    "HueAdjustmentOrange",
    "HueAdjustmentYellow",
    "HueAdjustmentGreen",
    "HueAdjustmentAqua",
    "HueAdjustmentBlue",
    "HueAdjustmentPurple",
    "HueAdjustmentMagenta",
    "SaturationAdjustmentRed",
    "SaturationAdjustmentOrange",
    "SaturationAdjustmentYellow",
    "SaturationAdjustmentGreen",
    "SaturationAdjustmentAqua",
    "SaturationAdjustmentBlue",
    "SaturationAdjustmentPurple",
    "SaturationAdjustmentMagenta",
    "LuminanceAdjustmentRed",
    "LuminanceAdjustmentOrange",
    "LuminanceAdjustmentYellow",
    "LuminanceAdjustmentGreen",
    "LuminanceAdjustmentAqua",
    "LuminanceAdjustmentBlue",
    "LuminanceAdjustmentPurple",
    "LuminanceAdjustmentMagenta",
    # Toggles
    "EnableProfileCorrections",
    "VignetteAmount",
    # Tone curve (placeholder keys for v1; full curve support added later)
    # Parametric curve slots commonly used by Lightroom PV2012
    "ToneCurvePV2012",
    "ToneCurvePV2012Red",
    "ToneCurvePV2012Green",
    "ToneCurvePV2012Blue",
]

# Slider ranges for normalization and clamping (Lightroom-typical safe bounds)
RANGES_LINEAR: Dict[str, Tuple[float, float]] = {
    # Basic (2012 process)
    "Exposure2012": (-5.0, 5.0),
    "Contrast2012": (-100.0, 100.0),
    "Highlights2012": (-100.0, 100.0),
    "Shadows2012": (-100.0, 100.0),
    "Whites2012": (-100.0, 100.0),
    "Blacks2012": (-100.0, 100.0),
    # Presence
    "Clarity2012": (-100.0, 100.0),
    "Texture": (-100.0, 100.0),
    "Dehaze": (-100.0, 100.0),
    # Color
    "Vibrance": (-100.0, 100.0),
    "Saturation": (-100.0, 100.0),
    # Tint
    "Tint": (-150.0, 150.0),
    # HSL
    "HueAdjustmentRed": (-100.0, 100.0),
    "HueAdjustmentOrange": (-100.0, 100.0),
    "HueAdjustmentYellow": (-100.0, 100.0),
    "HueAdjustmentGreen": (-100.0, 100.0),
    "HueAdjustmentAqua": (-100.0, 100.0),
    "HueAdjustmentBlue": (-100.0, 100.0),
    "HueAdjustmentPurple": (-100.0, 100.0),
    "HueAdjustmentMagenta": (-100.0, 100.0),
    "SaturationAdjustmentRed": (-100.0, 100.0),
    "SaturationAdjustmentOrange": (-100.0, 100.0),
    "SaturationAdjustmentYellow": (-100.0, 100.0),
    "SaturationAdjustmentGreen": (-100.0, 100.0),
    "SaturationAdjustmentAqua": (-100.0, 100.0),
    "SaturationAdjustmentBlue": (-100.0, 100.0),
    "SaturationAdjustmentPurple": (-100.0, 100.0),
    "SaturationAdjustmentMagenta": (-100.0, 100.0),
    "LuminanceAdjustmentRed": (-100.0, 100.0),
    "LuminanceAdjustmentOrange": (-100.0, 100.0),
    "LuminanceAdjustmentYellow": (-100.0, 100.0),
    "LuminanceAdjustmentGreen": (-100.0, 100.0),
    "LuminanceAdjustmentAqua": (-100.0, 100.0),
    "LuminanceAdjustmentBlue": (-100.0, 100.0),
    "LuminanceAdjustmentPurple": (-100.0, 100.0),
    "LuminanceAdjustmentMagenta": (-100.0, 100.0),
    # Vignette
    "VignetteAmount": (-100.0, 100.0),
}

# Temperature uses log scale (Kelvin)
TEMP_MIN_K = 2000.0
TEMP_MAX_K = 50000.0


@dataclass
class DevelopSettings:
    """Container for a subset of Lightroom develop settings (v1 scope)."""
    values: Dict[str, Any]

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        return self.values.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.values[key] = value

    def clamp_inplace(self) -> None:
        for k, v in list(self.values.items()):
            if k == "Temperature":
                self.values[k] = clamp(v, TEMP_MIN_K, TEMP_MAX_K)
            elif k in RANGES_LINEAR:
                lo, hi = RANGES_LINEAR[k]
                self.values[k] = clamp(v, lo, hi)
            elif k in ("EnableProfileCorrections",):
                self.values[k] = int(1 if bool(v) else 0)
            # Tone curves handled as pass-through strings or arrays for now


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def to_norm_linear(x: float, lo: float, hi: float) -> float:
    x = clamp(x, lo, hi)
    return (2.0 * (x - lo) / (hi - lo)) - 1.0


def from_norm_linear(n: float, lo: float, hi: float) -> float:
    n = clamp(n, -1.0, 1.0)
    return lo + (hi - lo) * (n + 1.0) / 2.0


def to_norm_temperature_k(k: float) -> float:
    """Normalize temperature (Kelvin) to [-1, 1] via log scale in [TEMP_MIN_K, TEMP_MAX_K]."""
    k = clamp(k, TEMP_MIN_K, TEMP_MAX_K)
    a = math.log(TEMP_MIN_K)
    b = math.log(TEMP_MAX_K)
    v = (math.log(k) - a) / (b - a)  # in [0,1]
    return v * 2.0 - 1.0


def from_norm_temperature_k(n: float) -> float:
    n = clamp(n, -1.0, 1.0)
    a = math.log(TEMP_MIN_K)
    b = math.log(TEMP_MAX_K)
    v01 = (n + 1.0) / 2.0
    return math.exp(a + (b - a) * v01)


def normalize(settings: DevelopSettings) -> Dict[str, float]:
    """Map Lightroom slider values to normalized space [-1, 1]."""
    out: Dict[str, float] = {}
    for k, v in settings.values.items():
        if k == "Temperature":
            out[k] = to_norm_temperature_k(float(v))
        elif k in RANGES_LINEAR:
            lo, hi = RANGES_LINEAR[k]
            out[k] = to_norm_linear(float(v), lo, hi)
        elif k in ("EnableProfileCorrections",):
            out[k] = 1.0 if int(v) != 0 else -1.0  # binary to {-1, +1}
        else:
            # Non-normalizable fields (tone curves etc.) skipped
            continue
    return out


def denormalize(norm: Dict[str, float]) -> DevelopSettings:
    """Map normalized values back to Lightroom slider space (and clamp)."""
    vals: Dict[str, Any] = {}
    for k, n in norm.items():
        if k == "Temperature":
            vals[k] = from_norm_temperature_k(float(n))
        elif k in RANGES_LINEAR:
            lo, hi = RANGES_LINEAR[k]
            vals[k] = from_norm_linear(float(n), lo, hi)
        elif k in ("EnableProfileCorrections",):
            vals[k] = 1 if float(n) > 0.0 else 0
    ds = DevelopSettings(vals)
    ds.clamp_inplace()
    return ds


def build_neutral_settings() -> DevelopSettings:
    """Return a 'neutral' baseline: Adobe Color intent with zeroed sliders, no masks."""
    vals: Dict[str, Any] = {
        # WB (Adobe neutral assumes 5500K/0 tint as a generic baseline; we store neutral as 5500/0)
        "Temperature": 5500.0,
        "Tint": 0.0,
        # Basic
        "Exposure2012": 0.0,
        "Contrast2012": 0.0,
        "Highlights2012": 0.0,
        "Shadows2012": 0.0,
        "Whites2012": 0.0,
        "Blacks2012": 0.0,
        # Presence
        "Clarity2012": 0.0,
        "Texture": 0.0,
        "Dehaze": 0.0,
        # Color
        "Vibrance": 0.0,
        "Saturation": 0.0,
        # HSL (all zeros imply no channel adjustments)
        "HueAdjustmentRed": 0.0,
        "HueAdjustmentOrange": 0.0,
        "HueAdjustmentYellow": 0.0,
        "HueAdjustmentGreen": 0.0,
        "HueAdjustmentAqua": 0.0,
        "HueAdjustmentBlue": 0.0,
        "HueAdjustmentPurple": 0.0,
        "HueAdjustmentMagenta": 0.0,
        "SaturationAdjustmentRed": 0.0,
        "SaturationAdjustmentOrange": 0.0,
        "SaturationAdjustmentYellow": 0.0,
        "SaturationAdjustmentGreen": 0.0,
        "SaturationAdjustmentAqua": 0.0,
        "SaturationAdjustmentBlue": 0.0,
        "SaturationAdjustmentPurple": 0.0,
        "SaturationAdjustmentMagenta": 0.0,
        "LuminanceAdjustmentRed": 0.0,
        "LuminanceAdjustmentOrange": 0.0,
        "LuminanceAdjustmentYellow": 0.0,
        "LuminanceAdjustmentGreen": 0.0,
        "LuminanceAdjustmentAqua": 0.0,
        "LuminanceAdjustmentBlue": 0.0,
        "LuminanceAdjustmentPurple": 0.0,
        "LuminanceAdjustmentMagenta": 0.0,
        # Toggles
        "EnableProfileCorrections": 0,
        "VignetteAmount": 0.0,
        # Tone curves set to straight 0-255 identity; stored as text sequences in LR XMP
        "ToneCurvePV2012": "(0, 0), (255, 255)",
        "ToneCurvePV2012Red": "(0, 0), (255, 255)",
        "ToneCurvePV2012Green": "(0, 0), (255, 255)",
        "ToneCurvePV2012Blue": "(0, 0), (255, 255)",
    }
    ds = DevelopSettings(vals)
    ds.clamp_inplace()
    return ds


def _ensure_xmp_tree(tree: Optional[ET._ElementTree]) -> ET._ElementTree:
    if tree is not None:
        return tree
    # Create minimal XMP packet
    xmpmeta = ET.Element(ET.QName(NS_X, "xmpmeta"), nsmap=NSMAP)
    rdf = ET.SubElement(xmpmeta, ET.QName(NS_RDF, "RDF"))
    ET.SubElement(rdf, ET.QName(NS_RDF, "Description"))
    return ET.ElementTree(xmpmeta)


def _get_rdf_description(tree: ET._ElementTree) -> ET._Element:
    root = tree.getroot()
    # Expect x:xmpmeta/rdf:RDF/rdf:Description
    rdf = root.find(f".//{{{NS_RDF}}}RDF")
    if rdf is None:
        rdf = ET.SubElement(root, ET.QName(NS_RDF, "RDF"))
    desc = rdf.find(f"./{{{NS_RDF}}}Description")
    if desc is None:
        desc = ET.SubElement(rdf, ET.QName(NS_RDF, "Description"))
    return desc


def read_xmp(path: Path) -> DevelopSettings:
    """Parse XMP file and extract supported develop settings."""
    data: Dict[str, Any] = {}
    # Use a parser that can handle huge files to avoid XMLSyntaxError on large XMPs
    parser = ET.XMLParser(huge_tree=True)
    with open(path, "rb") as f:
        tree = ET.parse(f, parser)
    desc = _get_rdf_description(tree)
    # Attributes with 'crs' namespace live on rdf:Description
    for key in SUPPORTED_CRS_KEYS:
        qn = ET.QName(NS_CRS, key)
        if qn.text in desc.attrib:
            data[key] = _coerce_number_or_text(desc.attrib[qn.text])
    ds = DevelopSettings(data)
    ds.clamp_inplace()
    return ds


def write_xmp(
    path: Path,
    updates: DevelopSettings,
    base: Optional[Path] = None,
    preserve_unrelated: bool = True,
    create_dirs: bool = True,
) -> None:
    """Write or update XMP with provided develop settings.

    - If 'base' is given, start from that XMP's content (preserve unrelated metadata).
    - Otherwise, if 'preserve_unrelated' is True and path exists, load the file first.
    - Else create a minimal packet.
    """
    updates = DevelopSettings(dict(updates.values))  # copy
    updates.clamp_inplace()

    tree: Optional[ET._ElementTree] = None
    if base and base.exists():
        with open(base, "rb") as f:
            tree = ET.parse(f)
    elif preserve_unrelated and path.exists():
        with open(path, "rb") as f:
            tree = ET.parse(f)

    tree = _ensure_xmp_tree(tree)
    desc = _get_rdf_description(tree)

    # Ensure crs namespace is present in the element
    # lxml keeps ns via nsmap on ancestors; attributes may be set via QName
    for key, val in updates.values.items():
        qn = ET.QName(NS_CRS, key)
        desc.set(qn, _to_xmp_value(val))

    if create_dirs:
        path.parent.mkdir(parents=True, exist_ok=True)
    # Pretty print with XML declaration
    xml_bytes = ET.tostring(tree, pretty_print=True, xml_declaration=True, encoding="UTF-8")
    path.write_bytes(xml_bytes)


def _coerce_number_or_text(s: str) -> Any:
    try:
        if "." in s or "e" in s.lower():
            return float(s)
        return int(s)
    except Exception:
        return s


def _to_xmp_value(v: Any) -> str:
    if isinstance(v, bool):
        return "True" if v else "False"
    return str(v)


__all__ = [
    "DevelopSettings",
    "normalize",
    "denormalize",
    "build_neutral_settings",
    "read_xmp",
    "write_xmp",
]