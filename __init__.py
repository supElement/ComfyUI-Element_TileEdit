import logging
import traceback

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]


def _register(name, cls, display):
    NODE_CLASS_MAPPINGS[name] = cls
    NODE_DISPLAY_NAME_MAPPINGS[name] = display


# --- 核心：一体化节点（稳定） ---
try:
    from .nodes import TiledUpscale
    _register("TiledUpscale", TiledUpscale, "Element Tiled Upscale")
except Exception:
    logging.warning("[Element TileUpscale] TiledUpscale failed to load:\n%s", traceback.format_exc())

# --- 拆分版节点 ---
try:
    from .nodes_split import ElementTileSplit
    _register("ElementTileSplit", ElementTileSplit, "Element Tile Split")
except Exception:
    logging.warning("[Element TileUpscale] ElementTileSplit failed to load:\n%s", traceback.format_exc())

try:
    from .nodes_encode import ElementTileEncode
    _register("ElementTileEncode", ElementTileEncode, "Element Tile Encode")
except Exception:
    logging.warning("[Element TileUpscale] ElementTileEncode failed to load:\n%s", traceback.format_exc())

try:
    from .nodes_merge import ElementTileMerge
    _register("ElementTileMerge", ElementTileMerge, "Element Tile Merge")
except Exception:
    logging.warning("[Element TileUpscale] ElementTileMerge failed to load:\n%s", traceback.format_exc())

try:
    from .nodes_patch import ElementKrea2InContextPatch
    _register("ElementKrea2InContextPatch", ElementKrea2InContextPatch, "Element Krea2 In-Context Patch")
except Exception:
    logging.warning("[Element TileUpscale] ElementKrea2InContextPatch failed to load:\n%s", traceback.format_exc())
    
try:
    from .nodes_patch import ElementStrengthSchedule
    _register("ElementStrengthSchedule", ElementStrengthSchedule, "Element Strength Schedule")
except Exception:
    logging.warning("[Element TileUpscale] ElementStrengthSchedule failed to load:\n%s", traceback.format_exc())

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
