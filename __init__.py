from .nodes import StoryLLMLoader, StoryBoardGenerator

NODE_CLASS_MAPPINGS = {
    "StoryLLMLoader": StoryLLMLoader,
    "StoryBoardGenerator": StoryBoardGenerator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StoryLLMLoader": "GGUF LLM Loader (Story)",
    "StoryBoardGenerator": "Storyboard Generator (GGUF)"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
