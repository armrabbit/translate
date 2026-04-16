# Changelog

All notable changes are tracked here.

## v2.8.5
- Add automatic dependency install for AI upscaler (`torch`, `realesrgan`, `basicsr`) on first use.
- Add env switch `COMICTRANSLATE_AUTO_INSTALL_UPSCALER_DEPS=0` to disable auto-install.
- Fix potential lock deadlock during first Real-ESRGAN model download initialization.

## v2.8.4
- Make editor `Image Upscaler` fallback automatically to Lanczos when Real-ESRGAN dependencies are missing.
- Prevent hard error popups when `torch/realesrgan/basicsr` are not installed.

## v2.8.3
- Switch image upscaler to AI Real-ESRGAN (anime model) with automatic model download.
- Keep Lanczos as fallback in export/batch flows when AI dependencies are unavailable.
- Add dedicated `Image Upscaler` button in editor that requires AI backend.

## v2.8.2
- Fix paint/restore stroke disappearing after mouse release.
- Commit: `0938c91`

## v2.8.1
- Replace release-installer updater with Git-based update flow (`git fetch`/`git pull --ff-only`).
- Auto-sync dependencies after update (`pip install -r requirements.txt`).
- Commit: `ffb1fc5`

## v2.8.0
- Add `Paint` tool and `Restore from Original` tool.
- Commit: `43de0a2`

## v2.7.9
- Handle missing GitHub release metadata (404) without noisy popup/errors.
- Commit: `57490e4`

## v2.7.8
- Improve workflow performance in manual pipeline paths.
- Harden update checker and translation JSON handling.
- Commit: `4b5426c`

## v2.7.7
- Fix webtoon page targeting in `All` workflow.
- Fix OCR bbox integer casting issue.
- Improve update repo parsing/check behavior.
- Commit: `e2eb717`

## v2.7.6
- Harden LLM translation JSON parsing and add debug logs.
- Commit: `848c307`

## v2.7.5
- Fix staged verify state and update checker edge cases.
- Commit: `d7076c6`

## v2.7.4
- Add `Verify` button for staged `All` workflow.
- Commit: `030df54`

## v2.7.3
- Allow Deepseek translation without requiring login.
- Commit: `79d34e2`

## v2.7.2
- Add Thai source support.
- Add Deepseek integration.
- Commit: `42f3bf2`

## v2.7.1
- Baseline version before the changes above.
