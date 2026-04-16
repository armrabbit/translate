# Comic Translate
English | [Korean](docs/README_ko.md) | [French](docs/README_fr.md) | [Simplified Chinese](docs/README_zh-CN.md)

Comic Translate is a desktop app for translating comic pages (manga, webtoon, and western comics) with OCR + LLM-based translation + inpainting + text rendering.

This project was developed based on [ogkalu2/comic-translate](https://github.com/ogkalu2/comic-translate).

## Highlights
- Automatic batch translation for many pages.
- Manual editing workflow with stage-by-stage controls:
  - `Detect` -> `Recognize` -> `Translate` -> `Segment` -> `Clean` -> `Render`
- `All` + `Verify` flow in Manual mode:
  - Runs full multi-page pipeline in stages.
  - Pauses between stages until you press `Verify`.
- Webtoon mode with lazy loading for long-strip reading/editing.
- Multi-provider translation support, including `Deepseek-v3`.
- Project save/load support (`.ctpr`) with state restoration.

## Supported Translation Engines
From current settings UI:
- `Gemini-3.0-Flash`
- `GPT-4.1`
- `GPT-4.1-mini`
- `Claude-4.6-Sonnet`
- `Claude-4.5-Haiku`
- `Deepseek-v3`
- `Custom`

## Requirements
- Python `3.12` recommended.
- Install dependencies from `requirements.txt`.
- Optional:
  - NVIDIA GPU: `onnxruntime-gpu`
  - Archive tools for some comic formats (`.cbr`): `WinRAR` or `7-Zip` in `PATH`

## Installation (From Source)
1. Clone repo
```bash
git clone https://github.com/armrabbit/translate.git
cd translate
```

2. Create virtual environment
```bash
python -m venv .venv
```

3. Activate virtual environment
- Windows (PowerShell):
```powershell
.\.venv\Scripts\Activate.ps1
```
- macOS/Linux:
```bash
source .venv/bin/activate
```

4. Install dependencies
```bash
pip install -r requirements.txt
```

5. Run app
```bash
python comic.py
```

### Alternative with uv
```bash
uv init --python 3.12
uv add -r requirements.txt --compile-bytecode
uv run comic.py
```

## Basic Usage
1. Open images/archives/PDF in the app.
2. Choose source and target language.
3. Pick translator + OCR in Settings.
4. Use one of these modes:
   - Automatic: click `Translate All`
   - Manual: use stage buttons or `All` + `Verify`
5. Export/save translated pages or project.

## Manual Mode: `All` + `Verify`
The `All` button runs the staged full-page workflow and pauses at checkpoints.  
Press `Verify` to continue to the next stage.

Flow:
1. Detect + Recognize (all pages)
2. Verify
3. Translate (all pages)
4. Verify
5. Segment (all pages)
6. Verify
7. Clean (all pages)

## Update Checker (GitHub Releases)
The app checks updates from GitHub `releases/latest`.

Default repo:
- Owner: `armrabbit`
- Repo: `translate`

You can override with environment variables:
- `COMICTRANSLATE_UPDATE_REPO_OWNER`
- `COMICTRANSLATE_UPDATE_REPO_NAME`

Example:
```powershell
$env:COMICTRANSLATE_UPDATE_REPO_OWNER="your-owner"
$env:COMICTRANSLATE_UPDATE_REPO_NAME="your-repo"
python comic.py
```

Important: update detection uses GitHub Releases. A normal `git push` without creating a new Release will not appear as an app update.

## Troubleshooting
### `JSONDecodeError` during translation
- Cause: LLM returned malformed JSON.
- Current behavior: parser attempts recovery and logs debug context in `modules/utils/translator_utils.py`.
- If still failing:
  - retry translation
  - reduce prompt complexity / extra context
  - switch model/provider

### `RarCannotExec("Cannot find working tool")`
- Install `WinRAR` or `7-Zip`.
- Add installation folder to system `PATH`.

### No GPU acceleration
- Install GPU runtime manually if needed:
```bash
pip install onnxruntime-gpu
```

## Default Shortcuts
From `app/shortcuts.py`:
- Undo: `Ctrl+Z`
- Redo: `Ctrl+Y`
- Delete Selected Box: `Delete`
- Restore Text Blocks: `Ctrl+Shift+R`

## Tech Stack
- UI: `PySide6`
- OCR: `manga-ocr`, `Pororo`, `PPOCRv5`, optional cloud OCR
- Detection: `RT-DETR-v2` pipeline
- Inpainting: `LaMa`, `AOT`
- Translation: multi-LLM provider architecture

## License
Apache License 2.0. See `LICENSE`.
