import os
import json
import folder_paths
import numpy as np
import torch
from PIL import Image

try:
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import NanoLlavaChatHandler, MoondreamChatHandler
except ImportError:
    print("Error: llama_cpp_python not installed. Please install it to use this node.")
    Llama = None

# Global cache for keeping models loaded
LLM_CACHE = {}

# Register 'LLM' folder in folder_paths if not already there
if "LLM" not in folder_paths.folder_names_and_paths:
    folder_paths.add_model_folder_path("LLM", os.path.join(folder_paths.models_dir, "LLM"))

class StoryLLMLoader:
    @classmethod
    def INPUT_TYPES(s):
        # Scan multiple directories for GGUF models
        model_list = []
        # Areas where users usually put GGUF or LLM files
        folders_to_scan = ["LLM", "text_encoders", "clip", "checkpoints", "diffusion_models", "unet"]
        
        print("\n--- Story GGUF Node: Scanning for models ---")
        for folder in folders_to_scan:
            try:
                # get_filename_list typically returns a list of relative paths
                files = folder_paths.get_filename_list(folder)
                found = [f for f in files if f.lower().endswith(".gguf")]
                if found:
                    print(f"  [+] Found {len(found)} models in '{folder}'")
                    # We store them as "folder/filename" if not in the primary LLM folder
                    for f in found:
                        model_list.append(f)
            except Exception as e:
                # print(f"  [-] Skip scanning '{folder}': {str(e)}")
                pass
        
        # Fallback: manually scan models/LLM if folder_paths failed to register it correctly
        try:
            manual_path = os.path.join(folder_paths.models_dir, "LLM")
            if os.path.exists(manual_path):
                raw_files = os.listdir(manual_path)
                manual_found = [f for f in raw_files if f.lower().endswith(".gguf")]
                if manual_found:
                    print(f"  [+] Found {len(manual_found)} models in manual scan of 'models/LLM'")
                    model_list.extend(manual_found)
        except:
            pass

        model_list = sorted(list(set(model_list)))
        if not model_list:
            print("  [!] WARNING: No .gguf models found in any scanned folder!")
            model_list = ["No .gguf models found - put them in models/LLM/"]
        
        print("--- End of Scan ---\n")

        mmproj_list = ["None"]
        for folder in ["clip", "text_encoders", "LLM"]:
            try:
                files = folder_paths.get_filename_list(folder)
                mmproj_list.extend([f for f in files if "mmproj" in f.lower()])
            except:
                pass

        return {
            "required": {
                "model_name": (model_list, ),
                "n_ctx": ("INT", {"default": 2048, "min": 512, "max": 32768, "step": 512}),
                "n_gpu_layers": ("INT", {"default": -1, "min": -1, "max": 100, "step": 1}),
            },
            "optional": {
                "mmproj_name": (mmproj_list, {"default": "None"}),
            }
        }

    RETURN_TYPES = ("LLM_MODEL",)
    RETURN_NAMES = ("llm_model",)
    FUNCTION = "load_model"
    CATEGORY = "Story/Loaders"

    def load_model(self, model_name, n_ctx, n_gpu_layers, mmproj_name="None"):
        if Llama is None:
            raise ImportError("llama-cpp-python is not installed.")
        
        # Find path
        model_path = folder_paths.get_full_path("LLM", model_name)
        if not model_path:
            for folder in ["text_encoders", "clip", "checkpoints"]:
                model_path = folder_paths.get_full_path(folder, model_name)
                if model_path: break
        
        if not model_path:
            raise FileNotFoundError(f"Model {model_name} not found.")

        mmproj_path = None
        if mmproj_name != "None":
            for folder in ["clip", "text_encoders", "LLM"]:
                mmproj_path = folder_paths.get_full_path(folder, mmproj_name)
                if mmproj_path: break

        # Cache key based on parameters
        cache_key = f"{model_path}_{n_ctx}_{n_gpu_layers}_{mmproj_path}"
        
        if cache_key in LLM_CACHE:
            print(f"Using cached model: {model_name}")
            return (LLM_CACHE[cache_key],)

        print(f"Loading LLM: {model_name} | n_gpu_layers: {n_gpu_layers}")
        
        # Vision handler if mmproj is provided
        chat_handler = None
        if mmproj_path:
            # Basic fallback vision handler
            chat_handler = NanoLlavaChatHandler(mmproj_path, verbose=False)

        llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            chat_handler=chat_handler,
            verbose=False
        )
        
        # Store metadata in the object instead of just the loader
        model_obj = {
            "llm": llm,
            "cache_key": cache_key,
            "is_vision": mmproj_path is not None
        }
        
        return (model_obj,)

class StoryBoardGenerator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "llm_model": ("LLM_MODEL",),
                "story_concept": ("STRING", {"multiline": True, "default": "A hero's journey."}),
                "page_count": ("INT", {"default": 4, "min": 1, "max": 10}),
                "style": ("STRING", {"default": "cinematic, 8k"}),
                "max_tokens": ("INT", {"default": 1024, "min": 1, "max": 8192}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 40, "min": 0, "max": 100}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "image": ("IMAGE",),
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT") 
    RETURN_NAMES = ("storyboard_text", "scene_list_json", "used_seed")
    FUNCTION = "generate_storyboard"
    CATEGORY = "Story/Generators"

    def generate_storyboard(self, llm_model, story_concept, page_count, style, 
                         max_tokens, temperature, top_p, top_k, seed,
                         image=None, keep_model_loaded=False):
        
        llm = llm_model["llm"]
        is_vision = llm_model["is_vision"]
        
        # Handle Vision Model
        messages = []
        if is_vision and image is not None:
            # Convert ComfyUI IMAGE to base64 or temporary file for llama-cpp
            # For simplicity in this env, we'll assume a standard chat format
            # In a real ComfyUI env, we'd need to convert the tensor to a suitable format
            img_path = self.process_image(image)
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"file://{img_path}"}},
                    {"type": "text", "text": f"Based on this image and the concept: '{story_concept}', generate {page_count} storyboard scenes in the style of {style}."}
                ]
            })
        else:
            prompt = self.build_prompt(story_concept, page_count, style)
            messages.append({"role": "user", "content": prompt})

        print(f"Generating storyboard (Seed: {seed})...")
        
        # Set seed
        np.random.seed(seed % (2**32))
        
        try:
            response = llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                seed=seed
            )
            
            generated_text = response['choices'][0]['message']['content']
            
            # Remove <think> tags if present
            if "<think>" in generated_text:
                import re
                generated_text = re.sub(r'<think>.*?</think>', '', generated_text, flags=re.DOTALL).strip()
            
            print(f"Generation Complete.")

            # Memory management
            if keep_model_loaded:
                LLM_CACHE[llm_model["cache_key"]] = llm_model
            elif llm_model["cache_key"] in LLM_CACHE:
                del LLM_CACHE[llm_model["cache_key"]]

            # Parse JSON
            formatted_text, json_output = self.parse_scenes(generated_text)
            return (formatted_text, json_output, seed)

        except Exception as e:
            print(f"Inference Error: {str(e)}")
            return (f"Error: {str(e)}", "[]", seed)

    def build_prompt(self, story_concept, page_count, style):
        return f"""You are a storyboard artist. Break down the following story concept into {page_count} distinct scenes.
For each scene, provide a visual description suitable for an image generator.
Style: {style}.

Story Concept:
{story_concept}

Output ONLY a valid JSON array.
[
  {{"scene_number": 1, "description": "Scene 1 description..."}},
  ...
]
Response:"""

    def parse_scenes(self, text):
        try:
            start = text.find('[')
            end = text.rfind(']') + 1
            if start != -1 and end != -1:
                json_str = text[start:end]
                scenes = json.loads(json_str) 
                formatted = ""
                for s in scenes:
                    formatted += f"Scene {s.get('scene_number', '?')}: {s.get('description', '')}\n\n"
                return formatted, json.dumps(scenes)
        except:
            pass
        return text, "[]"

    def process_image(self, image_tensor):
        # image_tensor is [B, H, W, C]
        import tempfile
        i = 255. * image_tensor[0].cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        img.save(tmp.name)
        return tmp.name

