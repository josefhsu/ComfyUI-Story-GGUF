import os
import json
import folder_paths

try:
    from llama_cpp import Llama
except ImportError:
    print("Error: llama_cpp_python not installed. Please install it to use this node.")
    Llama = None

class StoryLLMLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_path": ("STRING", {"default": "models/llm/checkpoints/model.gguf"}),
                "n_ctx": ("INT", {"default": 2048, "min": 512, "max": 32768, "step": 512}),
                "n_gpu_layers": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
            }
        }

    RETURN_TYPES = ("LLM_MODEL",)
    RETURN_NAMES = ("llm_model",)
    FUNCTION = "load_model"
    CATEGORY = "Story/Loaders"

    def load_model(self, model_path, n_ctx, n_gpu_layers):
        if Llama is None:
            raise ImportError("llama-cpp-python is not installed.")
        
        # Verify model exists
        if not os.path.isfile(model_path):
             # Try looking in ComfyUI models folder if not absolute
             possible_path = os.path.join(folder_paths.base_path, model_path)
             if os.path.isfile(possible_path):
                 model_path = possible_path
             else:
                 raise FileNotFoundError(f"Model not found at {model_path}")

        print(f"Loading LLM from {model_path} with n_ctx={n_ctx}, n_gpu_layers={n_gpu_layers}...")
        llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False
        )
        return (llm,)

class StoryBoardGenerator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "llm_model": ("LLM_MODEL",),
                "story_concept": ("STRING", {"multiline": True, "default": "A young hero sets out to find the lost sword."}),
                "page_count": ("INT", {"default": 4, "min": 1, "max": 10}),
                "style": ("STRING", {"default": "cinematic, 8k, detailed"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING") 
    RETURN_NAMES = ("storyboard_text", "scene_list_json")
    FUNCTION = "generate_storyboard"
    CATEGORY = "Story/Generators"

    def generate_storyboard(self, llm_model, story_concept, page_count, style):
        if llm_model is None:
            raise ValueError("LLM model not loaded.")

        prompt = f"""You are a storyboard artist. Break down the following story concept into {page_count} distinct scenes.
For each scene, provide a visual description suitable for an image generator (like Stable Diffusion).
The style should be: {style}.

Story Concept:
{story_concept}

Output ONLY a valid JSON array of objects, where each object has a "scene_number" and "description".
Example format:
[
  {{"scene_number": 1, "description": "Wide shot of a misty mountain peak..."}},
  {{"scene_number": 2, "description": "Close up... "}}
]

Response:
"""
        
        print("Generating storyboard...")
        output = llm_model(
            prompt,
            max_tokens=1024,
            stop=["User:", "\n\n"],
            echo=False
        )
        
        generated_text = output['choices'][0]['text']
        print(f"LLM Output: {generated_text}")

        # Attempt to parse specific JSON standard to return clean list
        try:
             # Basic cleanup to find JSON block
            start = generated_text.find('[')
            end = generated_text.rfind(']') + 1
            if start != -1 and end != -1:
                json_str = generated_text[start:end]
                scenes = json.loads(json_str) 
                
                # Format for readability
                formatted_text = ""
                for scene in scenes:
                    formatted_text += f"Scene {scene.get('scene_number', '?')}: {scene.get('description', '')}\n\n"
                    
                return (formatted_text, json.dumps(scenes))
            else:
                return (generated_text, "[]") # Fallback
        except json.JSONDecodeError:
            print("Failed to decode JSON from LLM output")
            return (generated_text, "[]")

