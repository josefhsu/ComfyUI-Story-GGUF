# ComfyUI-Story-GGUF

[中文](#中文) | [English](#english)

---

## 中文

### 簡介
這是一個為 ComfyUI 設計的自定義節點，旨在利用**本地 GGUF 格式的大型語言模型 (LLM)** 自動生成故事分鏡描述。它不需要串接 Ollama 或其他外部 API，完全在本地運行。

### 主要功能
- **內置 LLM 支持**: 使用 `llama-cpp-python` 直接加載 `.gguf` 模型。
- **分鏡生成器**: 自動將故事概念拆解為指定數量的分鏡。
- **視覺化描述**: 生成適合圖像生成模型（如 Stable Diffusion）使用的畫面描述。
- **JSON 輸出**: 支持 JSON 格式輸出，便於後續自動化處理。

### 安裝步驟
1. 將資料夾移動至 ComfyUI 的 `custom_nodes` 目錄。
2. 開啟終端機並安裝依賴：
   ```bash
   pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
   ```
   *(請根據您的 CUDA 版本選擇對應鏈接，或者使用 `cpu` 版本)*

### 使用方法
1. **GGUF LLM Loader (Story)**: 加載您的 GGUF 模型路徑並設置 `n_gpu_layers`。例如 `models/LLM/your_model.gguf`。
2. **Storyboard Generator (GGUF)**: 輸入故事概念、分鏡數量與風格。
3. 接上文字顯示節點查看結果。

---

## English

### Introduction
A custom node for ComfyUI designed to generate storyboard descriptions using **local GGUF Large Language Models (LLM)**. Run your LLMs directly within ComfyUI without needing Ollama or external APIs.

### Key Features
- **Embedded LLM Support**: Load `.gguf` models directly via `llama-cpp-python`.
- **Storyboard Generator**: Automatically break down story concepts into a specified number of scenes.
- **Visual Descriptions**: Generate detailed visual prompts suitable for image generators like Stable Diffusion.
- **Structured Output**: Provides both formatted text and JSON array outputs.

### Installation
1. Move this folder to your ComfyUI `custom_nodes` directory.
2. Install dependencies via terminal:
   ```bash
   pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
   ```
   *(Select the appropriate link for your CUDA version or use `cpu`)*

### How to Use
1. **GGUF LLM Loader (Story)**: Load your GGUF model path and configure `n_gpu_layers`.
2. **Storyboard Generator (GGUF)**: Enter your story concept, set page count and style.
3. Connect to a text display node to view results.

---

### Workflow Example
An example workflow `workflow_example.json` is included in the repository.
