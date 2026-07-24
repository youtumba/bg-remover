# BG Remover ✂️

**A free online tool for removing image backgrounds and AI upscaling — right in the browser.**

Live: **https://youtumba.github.io/bg-remover/**

Open the page, drop in an image, and download the result with a transparent background as PNG or WEBP. The Upscale tab enlarges images ×2 or ×4 while a neural network reconstructs the details. The interface is in English by default; Russian is available via the language switcher in the top-right corner.

## How to use

1. Open the page.
2. Pick a tab: **Background removal** or **Upscale ×2 / ×4**.
3. Choose a model in the dropdown (the default works well for most cases).
4. Drag & drop an image, paste it from the clipboard (Ctrl+V), or click to browse.
5. Wait for processing to finish and download the result.

## Choosing a model (Background removal)

| Model | Size | Best for |
|---|---|---|
| **RMBG 1.4** (default) | ~44 MB | Renders, game art, anime, graphics — the universal choice |
| ISNet Compact | ~42 MB | Slow connections: fastest download, basic quality |
| ISNet Balanced | ~84 MB | Middle ground between download size and quality |
| ISNet Full | ~170 MB | Best quality for real photos |

## Upscale tab

| Mode | Best for |
|---|---|
| Swin2SR ×2 | Renders, art, screenshots |
| Swin2SR ×4 | Renders, art, screenshots |
| Swin2SR ×4 Real-world | Photos, noisy or compressed images |

- The upscaling model (~50 MB) is downloaded once and cached by the browser.
- Large images are automatically split into overlapping tiles, upscaled piece by piece and stitched back together seamlessly. Input limits: ~4 MP for ×2 (around 2500×1600) and ~2.3 MP for ×4 (Full HD 1920×1080 fits).
- Processing runs on the device itself; large images can take several minutes.

## Why the first run takes longer

The neural network (42–170 MB depending on the model) is downloaded on first use and cached by the browser, so subsequent runs start much faster. All heavy work happens in a background worker — the page stays responsive while the model is thinking.

## Privacy

- Images are **never uploaded to any server** — all processing happens locally in the browser.
- No sign-up and no personal data required.
- Safe to use with personal or work images.

## FAQ

**Is it free?**
Yes, completely free.

**Does the quality suffer?**
Background removal keeps the original resolution — only the background is replaced with transparency. Upscaling increases the resolution ×2 or ×4.

**Nothing happens after selecting an image?**
An internet connection is required on first use to download the model. Reload the page and try again.

**Does it work on a phone?**
Yes, in modern mobile browsers. Large images may take a while.

---

Built with the open-source libraries [@imgly/background-removal](https://github.com/imgly/background-removal-js), [RMBG-1.4](https://huggingface.co/briaai/RMBG-1.4) and [Swin2SR](https://github.com/mv-lab/swin2sr).
