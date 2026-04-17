# Changelog

All notable changes are tracked here.

## v2.8.13
- Improve page-switch responsiveness by adding a RAM fast path in async navigation (skip worker dispatch when image is already cached).
- Improve sidebar thumbnail performance by moving archive materialization off the UI thread and increasing thumbnail cache/preload window.
- Reduce thumbnail CPU cost by avoiding upscaling tiny images and switching thumbnail resample mode from Lanczos to Bilinear.
- Optimize image history updates by preferring in-memory current frame for comparisons, reducing unnecessary disk reads during edits.

## v2.8.12
- Fix page reload edge case after upscaling: image loader now recovers from stale/missing history entries and falls back to latest valid frame.
- Fix potential failure when returning to edited pages by lazily materializing `image_data` in `load_image_state` when cache is empty.
- Harden `SetImageCommand` history reads to avoid index mismatch crashes between `in_memory_history` and `image_history`.

## v2.8.11
- Fix `Restore` tool baseline resolution by using composed restore source (`original + persisted clean patches`) instead of raw original only.
- Fix `Restore` tool failing after size-changing edits (e.g. upscaler) by remapping to a matching-size history image when available.
- Add safer restore mismatch logging for easier debugging when no compatible history frame exists.

## v2.8.10
- Fix editor camera jump after AI upscaler: keep viewport scene bounds synced to new image size when replacing pixmap.
- Fix post-upscale pan lock (unable to move to the right) by preserving view using image bounds instead of stale scene bounds.

## v2.8.9
- Harden AI upscaler dependency recovery by checking both `torch` and `torchvision` before skipping install.
- Fix dependency guidance text to include `torchvision` in manual install command.
- Validate downloaded Real-ESRGAN model file size and auto-redownload if file is invalid/corrupted.
- Make editor `Image Upscaler` enforce AI backend (`strict_ai`) instead of silently falling back to Lanczos.
- Fix `SyntaxWarning` in Pororo utility punctuation stripping (`invalid escape sequence`) to keep Python 3.12+ clean.

## v2.8.8
- Fix AI upscaler model URL (old URL returned 404) by switching fallback to a valid Real-ESRGAN release asset.
- Add GitHub Releases API discovery for `RealESRGAN_x4plus_anime_6B.pth` so model download can adapt when release tags change.
- Improve model download logs and fallback handling across multiple candidate URLs.

## v2.8.7
- Fix Real-ESRGAN startup failure with newer `torchvision` where `torchvision.transforms.functional_tensor` is missing.
- Add runtime compatibility shim so `basicsr/realesrgan` can import successfully without downgrading `torchvision`.

## v2.8.6
- Improve AI upscaler auto-install reliability on Windows by forcing `--user` install and adding user site-packages to `sys.path`.
- Include underlying install/import reason in fallback error logs.
- Reduce repeated fallback warnings (first warning only, subsequent messages as debug).

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
