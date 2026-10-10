from .nodes import (
    TiledUpscale,
)

NODE_CLASS_MAPPINGS = {
    "TiledUpscale": TiledUpscale,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "TiledUpscale": "Element Tiled Upscale",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
