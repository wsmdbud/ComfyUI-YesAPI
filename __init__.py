"""
ComfyUI-YesAPI: Full-featured ComfyUI custom node suite for https://yesapi.online
Includes 18 nodes across Image Generation/Editing, Async Image Queue, Dual-Protocol LLM/VLM, and Video Generation.
"""
from __future__ import annotations

try:
    from .nodes.image_async_nodes import (
        NODE_CLASS_MAPPINGS as _IMAGE_ASYNC_CLASSES,
        NODE_DISPLAY_NAME_MAPPINGS as _IMAGE_ASYNC_NAMES,
    )
    from .nodes.image_nodes import (
        NODE_CLASS_MAPPINGS as _IMAGE_CLASSES,
        NODE_DISPLAY_NAME_MAPPINGS as _IMAGE_NAMES,
    )
    from .nodes.llm_nodes import (
        NODE_CLASS_MAPPINGS as _LLM_CLASSES,
        NODE_DISPLAY_NAME_MAPPINGS as _LLM_NAMES,
    )
    from .nodes.video_nodes import (
        NODE_CLASS_MAPPINGS as _VIDEO_CLASSES,
        NODE_DISPLAY_NAME_MAPPINGS as _VIDEO_NAMES,
    )
except ImportError:
    from nodes.image_async_nodes import (
        NODE_CLASS_MAPPINGS as _IMAGE_ASYNC_CLASSES,
        NODE_DISPLAY_NAME_MAPPINGS as _IMAGE_ASYNC_NAMES,
    )
    from nodes.image_nodes import (
        NODE_CLASS_MAPPINGS as _IMAGE_CLASSES,
        NODE_DISPLAY_NAME_MAPPINGS as _IMAGE_NAMES,
    )
    from nodes.llm_nodes import (
        NODE_CLASS_MAPPINGS as _LLM_CLASSES,
        NODE_DISPLAY_NAME_MAPPINGS as _LLM_NAMES,
    )
    from nodes.video_nodes import (
        NODE_CLASS_MAPPINGS as _VIDEO_CLASSES,
        NODE_DISPLAY_NAME_MAPPINGS as _VIDEO_NAMES,
    )

__version__ = "1.0.0"

NODE_CLASS_MAPPINGS = {
    **_IMAGE_CLASSES,
    **_IMAGE_ASYNC_CLASSES,
    **_LLM_CLASSES,
    **_VIDEO_CLASSES,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **_IMAGE_NAMES,
    **_IMAGE_ASYNC_NAMES,
    **_LLM_NAMES,
    **_VIDEO_NAMES,
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "__version__"]
