// worker.js — BG Remover processing worker.
// All model downloads and neural-network inference run here, off the main
// thread, so the page stays responsive during processing.
//
// CDN notes (why these exact URLs):
// - Bundled ESM builds (+esm / esm.sh) are required: the raw dist/index.mjs
//   fails in browsers with 'Failed to resolve module specifier "lodash"'.
// - Several mirrors are listed because availability differs between ISPs
//   (cdn.jsdelivr.net can drop connections with ERR_EMPTY_RESPONSE while the
//   fastly/gcore mirrors of the same jsdelivr keep working).
// - Model/engine assets for @imgly cannot be served from jsdelivr: the data
//   package exceeds jsdelivr's 150 MB package limit and returns 403 for every
//   file. Primary source is the vendor CDN (staticimgly.com), fallback unpkg.

const LIB_URLS = [
  'https://cdn.jsdelivr.net/npm/@imgly/background-removal@1.5.8/+esm',
  'https://fastly.jsdelivr.net/npm/@imgly/background-removal@1.5.8/+esm',
  'https://gcore.jsdelivr.net/npm/@imgly/background-removal@1.5.8/+esm',
  'https://esm.sh/@imgly/background-removal@1.5.8'
];
const ASSETS_URLS = [
  'https://staticimgly.com/@imgly/background-removal-data/1.5.8/dist/',
  'https://unpkg.com/@imgly/background-removal-data@1.5.8/dist/'
];
const TJS_URLS = [
  'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2',
  'https://fastly.jsdelivr.net/npm/@xenova/transformers@2.17.2',
  'https://gcore.jsdelivr.net/npm/@xenova/transformers@2.17.2',
  'https://esm.sh/@xenova/transformers@2.17.2'
];

// Tiled upscaling parameters (limits are duplicated in index.html for the
// pre-flight check on the main thread — keep them in sync).
const UP_TILE = 512;       // tile size
const UP_OVERLAP = 32;     // tile overlap, hides seams at tile boundaries
const UP_MAX_X2 = 4200000; // ~4.2 MP for x2 (around 2500x1600)
const UP_MAX_X4 = 2300000; // ~2.3 MP for x4 (enough for Full HD 1920x1080)

let removeBgFn = null; // cached imgly removeBackground()
let assetsUrl = null;  // chosen working assets mirror (kept for the session)
let tjsMod = null;     // cached Transformers.js module (shared by RMBG and upscaling)
let rmbg = null;       // cached { RawImage, model, processor }
const upCache = {};    // cached upscaling pipelines by model id
let gpuChecked = null; // cached real WebGPU adapter availability (avoids noisy "No available adapters" logs)

// navigator.gpu can exist while no adapter is actually available (headless /
// software-only environments). Probing once avoids onnxruntime logging
// "No available adapters." on every run when we blindly request device:'gpu'.
async function hasGpuAdapter() {
  if (gpuChecked !== null) return gpuChecked;
  gpuChecked = false;
  try {
    if (self.navigator && self.navigator.gpu) {
      const adapter = await self.navigator.gpu.requestAdapter();
      gpuChecked = !!adapter;
    }
  } catch (_) { gpuChecked = false; }
  return gpuChecked;
}

function fail(code, message, info) {
  const err = new Error(message || code);
  err.code = code;
  if (info) err.info = info;
  return err;
}

async function pickAssetsUrl() {
  if (assetsUrl) return assetsUrl;
  for (const base of ASSETS_URLS) {
    try {
      const r = await fetch(base + 'resources.json', { cache: 'force-cache' });
      if (r.ok) { assetsUrl = base; return base; }
    } catch (_) { /* mirror unavailable, try the next one */ }
  }
  throw fail('cdn_assets');
}

async function loadImgly() {
  if (removeBgFn) return removeBgFn;
  let lastErr = null;
  for (const url of LIB_URLS) {
    try {
      const mod = await import(url);
      if (typeof mod.removeBackground !== 'function') throw new Error('broken CDN build');
      removeBgFn = mod.removeBackground;
      return removeBgFn;
    } catch (e) { lastErr = e; }
  }
  throw fail('cdn_lib', lastErr && lastErr.message);
}

async function loadTJS() {
  if (tjsMod) return tjsMod;
  let lastErr = null;
  for (const url of TJS_URLS) {
    try {
      const mod = await import(url);
      if (!mod.AutoModel || !mod.AutoProcessor || !mod.RawImage || !mod.pipeline) throw new Error('broken CDN build');
      mod.env.allowLocalModels = false;
      // GitHub Pages does not send the COOP/COEP headers required for WASM
      // multi-threading (crossOriginIsolated). Pinning to 1 thread up front
      // skips the failed multi-thread attempt and its console warning.
      mod.env.backends.onnx.wasm.numThreads = 1;
      // Serve the ONNX WASM engine from the same mirror as the library itself
      if (url.includes('jsdelivr')) mod.env.backends.onnx.wasm.wasmPaths = url + '/dist/';
      tjsMod = mod;
      return mod;
    } catch (e) { lastErr = e; }
  }
  throw fail('cdn_lib', lastErr && lastErr.message);
}

// --- Background removal: imgly isnet models ---
async function runImgly(file, model, report) {
  report({ k: 'engine' }, 4);
  const removeBackground = await loadImgly();
  report({ k: 'cdn' }, 10);
  const publicPath = await pickAssetsUrl();
  report({ k: 'model' }, 14);
  return await removeBackground(file, {
    // Valid models in @imgly/background-removal@1.5.8:
    // 'isnet' (full) | 'isnet_fp16' (medium) | 'isnet_quint8' (compact).
    // Each model is downloaded once and cached by the browser.
    model,
    // WebGPU when a real adapter is available — noticeably faster than WASM/CPU
    device: (await hasGpuAdapter()) ? 'gpu' : 'cpu',
    publicPath,
    progress(key, current, total) {
      if (!total) return;
      // 1.5.x emits keys like 'fetch:/models/isnet', 'compute:inference'
      const k = String(key).startsWith('compute') ? 'infer' : 'model';
      report({ k }, Math.min(95, Math.round(14 + current / total * 80)));
    }
  });
}

// --- Background removal: RMBG-1.4 (BRIA) via Transformers.js ---
// isnet is trained on photos and struggles with renders, anime and game
// graphics; RMBG-1.4 handles game characters, art and complex edges better.
async function loadRmbg(onProgress) {
  if (rmbg) return rmbg;
  const tjs = await loadTJS();
  const progress_callback = d => {
    if (d && d.status === 'progress' && String(d.file || '').endsWith('.onnx')) onProgress(d.progress || 0);
  };
  const model = await tjs.AutoModel.from_pretrained('briaai/RMBG-1.4', {
    config: { model_type: 'custom' },
    progress_callback
  });
  const processor = await tjs.AutoProcessor.from_pretrained('briaai/RMBG-1.4', {
    config: {
      do_normalize: true, do_pad: false, do_rescale: true, do_resize: true,
      feature_extractor_type: 'ImageFeatureExtractor',
      image_mean: [0.5, 0.5, 0.5], image_std: [1, 1, 1],
      resample: 2, rescale_factor: 0.00392156862745098,
      size: { width: 1024, height: 1024 }
    }
  });
  rmbg = { RawImage: tjs.RawImage, model, processor };
  return rmbg;
}

async function runRmbg(file, report) {
  report({ k: 'model' }, 6);
  const { RawImage, model, processor } = await loadRmbg(p => report({ k: 'model' }, Math.min(60, Math.round(6 + p * 0.54))));
  report({ k: 'infer' }, 65);
  const image = await RawImage.fromBlob(file);
  const { pixel_values } = await processor(image);
  const { output } = await model({ input: pixel_values });
  // The 0-255 mask is resized to the original resolution and written into the alpha channel
  const mask = await RawImage.fromTensor(output[0].mul(255).to('uint8')).resize(image.width, image.height);
  report({ k: 'finish' }, 92);
  const canvas = new OffscreenCanvas(image.width, image.height);
  const ctx = canvas.getContext('2d');
  ctx.drawImage(image.toCanvas(), 0, 0);
  const px = ctx.getImageData(0, 0, image.width, image.height);
  for (let i = 0; i < mask.data.length; ++i) px.data[4 * i + 3] = mask.data[i];
  ctx.putImageData(px, 0, 0);
  return await canvas.convertToBlob({ type: 'image/png' });
}

// --- Upscaling: Swin2SR via Transformers.js ---
async function loadUpscaler(modelId, onProgress) {
  if (upCache[modelId]) return upCache[modelId];
  const tjs = await loadTJS();
  const up = await tjs.pipeline('image-to-image', modelId, {
    // Quantized Swin2SR builds produce artifacts, so the full-precision
    // model (~50 MB) is used. Downloaded once and cached by the browser.
    quantized: false,
    progress_callback: d => {
      if (d && d.status === 'progress' && String(d.file || '').endsWith('.onnx')) onProgress(d.progress || 0);
    }
  });
  upCache[modelId] = up;
  return up;
}

// Tile positions along one axis: step = tile minus overlap, last tile pinned to the edge
function tileOffsets(size) {
  if (size <= UP_TILE) return [0];
  const step = UP_TILE - UP_OVERLAP;
  const offs = [];
  for (let o = 0; o + UP_TILE < size; o += step) offs.push(o);
  offs.push(size - UP_TILE);
  return offs;
}

async function runUpscale(file, modelId, report) {
  report({ k: 'engine' }, 3);
  const tjs = await loadTJS();
  const image = await tjs.RawImage.fromBlob(file);
  const scale = modelId.includes('x4') ? 4 : 2;
  const maxPx = scale === 4 ? UP_MAX_X4 : UP_MAX_X2;
  if (image.width * image.height > maxPx) {
    throw fail('too_large', 'image too large', { w: image.width, h: image.height, scale, maxPx });
  }
  report({ k: 'model' }, 6);
  const up = await loadUpscaler(modelId, p => report({ k: 'model' }, Math.min(55, Math.round(6 + p * 0.49))));
  report({ k: 'infer' }, 58);
  const cv = new OffscreenCanvas(image.width * scale, image.height * scale);
  const cctx = cv.getContext('2d');
  if (image.width <= UP_TILE && image.height <= UP_TILE) {
    // Small image — a single pass
    const out = await up(image);
    cctx.drawImage(out.toCanvas(), 0, 0);
  } else {
    // Large image — split into overlapping tiles, upscale one by one, stitch back
    const src = new OffscreenCanvas(image.width, image.height);
    src.getContext('2d').drawImage(image.toCanvas(), 0, 0);
    const xs = tileOffsets(image.width), ys = tileOffsets(image.height);
    const total = xs.length * ys.length;
    let done = 0;
    for (const sy of ys) {
      for (const sx of xs) {
        const tw = Math.min(UP_TILE, image.width - sx), th = Math.min(UP_TILE, image.height - sy);
        const tc = new OffscreenCanvas(tw, th);
        const tctx = tc.getContext('2d', { willReadFrequently: true });
        tctx.drawImage(src, sx, sy, tw, th, 0, 0, tw, th);
        const td = tctx.getImageData(0, 0, tw, th);
        const out = await up(new tjs.RawImage(td.data, tw, th, 4));
        // Inner edges are trimmed by half the overlap so seams stay invisible
        const x0 = sx === 0 ? 0 : sx + UP_OVERLAP / 2;
        const y0 = sy === 0 ? 0 : sy + UP_OVERLAP / 2;
        const x1 = sx + tw >= image.width ? sx + tw : sx + tw - UP_OVERLAP / 2;
        const y1 = sy + th >= image.height ? sy + th : sy + th - UP_OVERLAP / 2;
        cctx.drawImage(out.toCanvas(), (x0 - sx) * scale, (y0 - sy) * scale, (x1 - x0) * scale, (y1 - y0) * scale, x0 * scale, y0 * scale, (x1 - x0) * scale, (y1 - y0) * scale);
        done++;
        report({ k: 'tile', n: done, m: total }, Math.min(92, Math.round(58 + (done / total) * 34)));
      }
    }
  }
  report({ k: 'finish' }, 94);
  const blob = await cv.convertToBlob({ type: 'image/png' });
  return { blob, width: cv.width, height: cv.height, scale };
}

self.onmessage = async (e) => {
  const { id, task, file, model, modelId } = e.data || {};
  const post = m => self.postMessage(Object.assign({ id }, m));
  const report = (label, pct) => post({ type: 'progress', label, pct });
  try {
    let result;
    if (task === 'removeBg') {
      const blob = model === 'rmbg' ? await runRmbg(file, report) : await runImgly(file, model, report);
      result = { blob };
    } else if (task === 'upscale') {
      result = await runUpscale(file, modelId, report);
    } else {
      throw fail('bad_task', String(task));
    }
    post({ type: 'done', result });
  } catch (err) {
    post({
      type: 'error',
      code: (err && err.code) || null,
      message: String((err && err.message) || err),
      info: (err && err.info) || null
    });
  }
};
