# ComfyUI-Story-GGUF (Optimized)

[中文](#中文) | [English](#english)

---

## 中文

### 簡介
這是一個強大的 ComfyUI 自定義節點，專為**本地 GGUF 模型**設計，支援分鏡故事生成。本次優化參考了 `GGUFInference` 節點，加入了自動偵測、視覺模型 (VLM) 支援以及進階採樣功能。

### 主要功能
- **自動偵測模型**: 自動掃描 `models/LLM`, `models/text_encoders`, `models/clip` 目錄下的 `.gguf` 與 `mmproj` 檔案。
- **視覺模型 (VLM) 支援**: 支援傳入圖片，讓 LLM 根據畫面內容生成分鏡（需搭配 `mmproj` 檔案）。
- **進階採樣控制**: 加入 `Seed`, `Temperature`, `Top-P`, `Top-K` 等參數，精確控制生成結果。
- **智慧記憶體管理**: 支援 `keep_model_loaded` 選項，避免頻繁加載跳轉。
- **噪音清理**: 自動移除 `<think>` 標籤，適配思考型模型。

### 安裝步驟
1. 進入 `ComfyUI/custom_nodes` 資料夾。
2. 安裝依賴：
   ```bash
   pip install -r requirements.txt
   ```
   *註：若需 GPU 加速，請參考 llama-cpp-python 官方文檔安裝對應 CUDA 版本的 wheel。*

### 使用方法
1. **StoryLLMLoader**: 從下拉選單選擇模型與 `mmproj`（若使用視覺模型）。
2. **StoryBoardGenerator**: 輸入故事大綱，可選連接 `image` 分鏡參考圖。

---

## English

### Introduction
A powerful ComfyUI custom node for **local GGUF model** storyboard generation. Optimized with features like auto-detection, Vision-Language Model (VLM) support, and advanced sampling.

### Key Features
- **Auto-detection**: Automatically scans `models/LLM`, `models/text_encoders`, and `models/clip` for `.gguf` and `mmproj` files.
- **Vision Support (VLM)**: Generate stories based on input images using vision-language models (requires `mmproj`).
- **Advanced Sampling**: Full control over `Seed`, `Temperature`, `Top-P`, and `Top-K`.
- **Memory Management**: Maintain models in memory using `keep_model_loaded` for faster consecutive runs.
- **Thinking Tag Clean-up**: Automatically strips `<think>` tags from output.

### Installation
1. Navigate to `ComfyUI/custom_nodes`.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: For GPU acceleration, install the appropriate llama-cpp-python wheel for your CUDA version.*

### Usage
1. **StoryLLMLoader**: Select your model and optional `mmproj` from the dropdown.
2. **StoryBoardGenerator**: Enter your story concept and optionally connect an reference `image`.

---

### Workflow Example
An updated example `workflow_example.json` is provided in the repository.
