"""
Dry Bean Variety Information and Metadata Module.

Provides botanical profiles, morphological characteristics, and photographic
asset references for the seven dry bean cultivars in the BEANWISE platform.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
IMG_DIR = PROJECT_ROOT / "assets" / "bean_images"

VARIETY_METADATA: Dict[str, Dict[str, Any]] = {
    "SEKER": {
        "name": "SEKER",
        "scientific_name": "Phaseolus vulgaris 'Seker'",
        "common_name": "Sugar Bean / Seker Fasulye",
        "description": "A traditional Turkish sugar bean known for its nearly spherical, highly rounded seed structure. Possesses the highest compactness and roundness indexes among all evaluated cultivars.",
        "key_characteristics": "Small-to-medium size, high sphericity, smooth white seed coat, thin skin, sweet delicate flavor.",
        "culinary_uses": "Ideal for slow-simmered Turkish stews (Kuru Fasulye), salads, and bean purées due to its tender texture.",
        "image_file": "seker.jpg",
        "image_path": IMG_DIR / "seker.jpg",
    },
    "BARBUNYA": {
        "name": "BARBUNYA",
        "scientific_name": "Phaseolus vulgaris 'Barbunya'",
        "common_name": "Cranberry Bean / Borlotti",
        "description": "An oval-to-kidney shaped bean characterized by a distinctive cream-colored seed coat mottled with reddish-brown speckles and high geometric variability.",
        "key_characteristics": "Medium-large grain, mottled speckles, moderate aspect ratio, firm seed coat.",
        "culinary_uses": "Widely used in Mediterranean and Aegean olive oil dishes (Barbunya Pilaki) and hearty braised recipes.",
        "image_file": "barbunya.jpg",
        "image_path": IMG_DIR / "barbunya.jpg",
    },
    "BOMBAY": {
        "name": "BOMBAY",
        "scientific_name": "Phaseolus coccineus / 'Bombay'",
        "common_name": "Giant White Runner Bean",
        "description": "The largest cultivar in the dataset by a significant margin. Exhibiting extreme Area, Perimeter, and Major Axis measurements, it forms a completely distinct morphological cluster.",
        "key_characteristics": "Very large dimensional size (>150,000 pixel area), plump flattened kidney shape, bright white coat.",
        "culinary_uses": "Premium table bean used in oven-baked Greek Gigandes plaki and Turkish luxury mezze dishes.",
        "image_file": "bombay.jpg",
        "image_path": IMG_DIR / "bombay.jpg",
    },
    "CALI": {
        "name": "CALI",
        "scientific_name": "Phaseolus vulgaris 'Cali'",
        "common_name": "Cali Mottled Bean",
        "description": "A long, elongated kidney bean featuring beige-brown mottling. Demonstrates a high aspect ratio and substantial major axis length, distinguishing it from round cultivars.",
        "key_characteristics": "Elongated profile, brown-tan mottling, high major-to-minor axis ratio, rich earthy flavor.",
        "culinary_uses": "Excellent for thick winter stews, chili, and regional casserole preparations.",
        "image_file": "cali.jpg",
        "image_path": IMG_DIR / "cali.jpg",
    },
    "HOROZ": {
        "name": "HOROZ",
        "scientific_name": "Phaseolus vulgaris 'Horoz'",
        "common_name": "Rooster Bean / Long White Kidney",
        "description": "Named for its elongated rooster-spur curved shape. Characterized by high eccentricity, high aspect ratio, and a narrow minor axis length relative to its total span.",
        "key_characteristics": "Cylindrical elongated shape, pure white seed coat, high eccentricity, dense starch content.",
        "culinary_uses": "Classic variety for cold bean salads with onions, parsley, and sumac (Piyaz).",
        "image_file": "horoz.jpg",
        "image_path": IMG_DIR / "horoz.jpg",
    },
    "SIRA": {
        "name": "SIRA",
        "scientific_name": "Phaseolus vulgaris 'Sira'",
        "common_name": "Sira White Bean",
        "description": "A versatile medium-small white bean with balanced physical proportions. Sira exhibits dimensional overlap with Dermason, requiring multi-feature geometric profiling for accurate disambiguation.",
        "key_characteristics": "Medium size, slightly flattened oval, smooth texture, moderate eccentricity.",
        "culinary_uses": "Everyday commercial dry bean used across Turkish cafeterias and family stews.",
        "image_file": "sira.jpg",
        "image_path": IMG_DIR / "sira.jpg",
    },
    "DERMASON": {
        "name": "DERMASON",
        "scientific_name": "Phaseolus vulgaris 'Dermason'",
        "common_name": "Dermason Haricot Bean",
        "description": "The most widely produced dry bean in Turkey. Features a flat-oval shape, medium-small seed dimensions, and balanced geometric shape factors.",
        "key_characteristics": "Flat oval profile, thin skin, high culinary digestibility, uniform white appearance.",
        "culinary_uses": "The gold standard bean for traditional Turkish Kuru Fasulye with spiced pastirma or sucuk.",
        "image_file": "dermason.jpg",
        "image_path": IMG_DIR / "dermason.jpg",
    },
}


def get_variety_info(variety_name: str) -> Dict[str, Any]:
    """Returns variety profile dictionary for a given cultivar name."""
    clean_name = str(variety_name).upper().strip()
    return VARIETY_METADATA.get(
        clean_name,
        {
            "name": clean_name,
            "scientific_name": "Phaseolus vulgaris",
            "common_name": f"{clean_name} Bean",
            "description": f"Standard dry bean variety: {clean_name}.",
            "key_characteristics": "Geometric seed measurements matching UCI Dry Bean parameters.",
            "culinary_uses": "General culinary bean.",
            "image_file": None,
            "image_path": None,
        },
    )


def get_all_varieties() -> List[Dict[str, Any]]:
    """Returns all 7 variety metadata dictionaries."""
    return list(VARIETY_METADATA.values())
