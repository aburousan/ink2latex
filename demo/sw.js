const CACHE = 'ink2latex-v9';
const ORT = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/';
const KATEX = 'https://cdn.jsdelivr.net/npm/katex@0.16.22/dist/';
const SHELL = ['./', './index.html', './style.css', './app.js', './features.js', './photo.js'];
const CDN = [ORT + 'ort.wasm.min.js', ORT + 'ort-wasm-simd-threaded.mjs',
  ORT + 'ort-wasm-simd-threaded.wasm', KATEX + 'katex.min.js', KATEX + 'katex.min.css'];

self.addEventListener('install', event => {
  event.waitUntil((async () => {
    const cache = await caches.open(CACHE);
    await cache.addAll([...SHELL, ...CDN]);
    const css = await (await cache.match(KATEX + 'katex.min.css')).text();
    const fonts = [...css.matchAll(/url\(([^)]+)\)/g)].map(match => new URL(match[1].replace(/['"]/g, ''), KATEX).href);
    await cache.addAll([...new Set(fonts)]);
    await self.skipWaiting();
  })());
});
self.addEventListener('activate', event => {
  event.waitUntil((async () => {
    for (const key of await caches.keys()) if (key.startsWith('ink2latex-') && key !== CACHE) await caches.delete(key);
    await self.clients.claim();
  })());
});
self.addEventListener('fetch', event => {
  const url = new URL(event.request.url);
  if (event.request.method !== 'GET' || (url.origin !== self.location.origin && url.hostname !== 'cdn.jsdelivr.net')) return;
  event.respondWith((async () => {
    const cache = await caches.open(CACHE);
    const cached = await cache.match(event.request);
    if (cached) return cached;
    const response = await fetch(event.request);
    if (response.ok) await cache.put(event.request, response.clone());
    return response;
  })());
});
