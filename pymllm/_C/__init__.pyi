from __future__ import annotations
from . import engine
__all__ = ['engine', 'get_engine_ctx']
def get_engine_ctx() -> engine.MllmEngineCtx:
    """
    get the singleton instance of the engine context
    """
