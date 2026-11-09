import {features} from './features.js';
import {image_to_strokes, split_lines} from './photo.js';

const $ = id => document.getElementById(id);
const canvas = $('ink'), context = canvas.getContext('2d');
const strokes = [];
let current = null, pointer = null, timer, revision = 0, busy = false, queued = false;
let encoder, decoder, vocabulary, offline = false;
let photoRequest = 0;
// Undo history: 'add' for a written stroke, {erased: [[index, stroke], ...]} for an eraser sweep.
const actions = [];
let erasing = false, erased = null;
const ERASE_RADIUS = 12;

function redraw() {
  const rect = canvas.getBoundingClientRect(), ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(rect.width * ratio);
  canvas.height = Math.round(rect.height * ratio);
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.lineWidth = 2.8;
  context.strokeStyle = context.fillStyle = '#243b35';
  context.lineCap = context.lineJoin = 'round';
  for (const stroke of [...strokes, ...(current ? [current] : [])]) {
    if (stroke.length === 1) {
      context.beginPath(); context.arc(stroke[0][0], stroke[0][1], 1.4, 0, 2 * Math.PI); context.fill();
    } else {
      context.beginPath(); context.moveTo(stroke[0][0], stroke[0][1]);
      for (const [x, y] of stroke.slice(1)) context.lineTo(x, y);
      context.stroke();
    }
  }
  $('hint').hidden = strokes.length > 0 || !!current;
}

function point(event) {
  const rect = canvas.getBoundingClientRect();
  return [event.clientX - rect.left, event.clientY - rect.top, event.timeStamp];
}

// Compact LaTeX (\frac{dy}{dx}, \begin{matrix}): a space only where a command name meets a letter (\alpha x).
// Joined with spaces everywhere, \begin{ m a t r i x } failed to typeset and could not be pasted into LaTeX.
// Function names spelled in single letters (l i m, s i n) become their commands, as tidy() in tokenizer.py.
const FUNCTION_NAMES = ['arcsin', 'arccos', 'arctan', 'sinh', 'cosh', 'tanh', 'lim', 'sin', 'cos', 'tan', 'cot', 'sec',
  'csc', 'log', 'exp', 'max', 'min', 'det', 'sup', 'inf', 'ln'];
function tidy(tokens) {
  const out = [], letter = k => k >= 0 && k < tokens.length && /^[A-Za-z]$/.test(tokens[k]);
  for (let i = 0; i < tokens.length;) {
    const name = FUNCTION_NAMES.find(n => n.split('').every((c, j) => tokens[i + j] === c) && !letter(i - 1));
    if (name) { out.push('\\' + name); i += name.length; } else out.push(tokens[i++]);
  }
  return out;
}

function toLatex(tokens) {
  let out = '';
  for (const token of tokens) {
    if (/\\[A-Za-z]+$/.test(out) && /^[A-Za-z]/.test(token)) out += ' ';
    out += token;
  }
  return out;
}

function schedule() {
  clearTimeout(timer);
  if ($('auto').checked && strokes.length && !current) timer = setTimeout(recognise, 600);
}

// A stroke goes when the eraser passes within ERASE_RADIUS of any of its segments.
function near(stroke, x, y) {
  for (let i = 0; i < stroke.length; i++) {
    const [ax, ay] = stroke[i], [bx, by] = stroke[Math.min(i + 1, stroke.length - 1)];
    const dx = bx - ax, dy = by - ay, length = dx * dx + dy * dy;
    const t = length ? Math.max(0, Math.min(1, ((x - ax) * dx + (y - ay) * dy) / length)) : 0;
    if (Math.hypot(ax + t * dx - x, ay + t * dy - y) <= ERASE_RADIUS) return true;
  }
  return false;
}
function eraseAt([x, y]) {
  for (let i = strokes.length - 1; i >= 0; i--) if (near(strokes[i], x, y)) erased.push([i, strokes.splice(i, 1)[0]]);
}
// The eraser button, or a stylus's eraser end (button 5, or the eraser bit in buttons).
const erases = event => erasing || event.button === 5 || (event.pointerType === 'pen' && (event.buttons & 32) !== 0);

canvas.addEventListener('pointerdown', event => {
  if (pointer !== null || (event.button !== 0 && event.button !== 5)) return;
  event.preventDefault(); clearTimeout(timer); revision++;
  pointer = event.pointerId;
  if (erases(event)) { erased = []; eraseAt(point(event)); }
  else current = [point(event)];
  canvas.setPointerCapture(pointer); redraw();
});
canvas.addEventListener('pointermove', event => {
  if (event.pointerId !== pointer) return;
  const samples = event.getCoalescedEvents?.();
  for (const sample of samples?.length ? samples : [event]) erased ? eraseAt(point(sample)) : current.push(point(sample));
  redraw();
});
function endStroke(event) {
  if (event.pointerId !== pointer) return;
  if (erased) {
    if (erased.length) { actions.push({erased}); resetResult(); }
    erased = null;
  } else if (event.type !== 'pointercancel') {
    const last = point(event), previous = current.at(-1);
    if (last[0] !== previous[0] || last[1] !== previous[1]) current.push(last);
    strokes.push(current); actions.push('add');
  }
  current = null; pointer = null; revision++; redraw(); schedule();
}
canvas.addEventListener('pointerup', endStroke);
canvas.addEventListener('pointercancel', endStroke);
canvas.addEventListener('lostpointercapture', event => {
  if (event.pointerId === pointer) {
    if (erased?.length) actions.push({erased});
    current = null; erased = null; pointer = null; revision++; redraw();
  }
});
$('undo').onclick = () => {
  const action = actions.pop();
  // An eraser sweep comes back where its strokes were (in reverse order, so each index is right again).
  if (action?.erased) for (const [i, stroke] of [...action.erased].reverse()) strokes.splice(i, 0, stroke);
  else strokes.pop();
  revision++; redraw(); resetResult(); schedule();
};
// Pen or eraser: one is on, shown pressed.
function setTool(erase) {
  erasing = erase;
  for (const [id, on] of [['pen', !erase], ['eraser', erase]]) { $(id).classList.toggle('on', on); $(id).setAttribute('aria-pressed', String(on)); }
  canvas.classList.toggle('erasing', erase);
}
$('pen').onclick = () => setTool(false);
$('eraser').onclick = () => setTool(!erasing);
$('clear').onclick = () => { photoRequest++; $('photo-note').hidden = true; $('photo-file').value = ''; strokes.length = 0; actions.length = 0; current = null; pointer = null; revision++; clearTimeout(timer); queued = false; redraw(); resetResult(); };
$('photo').onclick = () => $('photo-file').click();
$('photo-file').onchange = () => readPhoto($('photo-file').files[0]);

// A picture from the clipboard (Ctrl+V / ⌘V anywhere, or the Paste button) or dropped on the box reads like a photo.
function pastedImage(items) {
  for (const item of items || []) if (item.kind === 'file' && item.type.startsWith('image/')) return item.getAsFile();
  return null;
}
document.addEventListener('paste', event => {
  const file = pastedImage(event.clipboardData?.items);
  if (!file) return;            // text pastes into the LaTeX box as usual
  event.preventDefault(); readPhoto(file);
});
$('paste').onclick = async () => {
  try {
    for (const item of await navigator.clipboard.read()) {
      const type = item.types.find(t => t.startsWith('image/'));
      if (type) { readPhoto(await item.getType(type)); return; }
    }
    $('status').textContent = 'No picture on the clipboard. Copy an image or take a screenshot first.';
  } catch {
    $('status').textContent = 'This browser will not let the page read the clipboard: press Ctrl+V (⌘V on a Mac) instead.';
  }
};
canvas.addEventListener('dragover', event => { if ([...event.dataTransfer.items].some(i => i.type.startsWith('image/'))) event.preventDefault(); });
canvas.addEventListener('drop', event => {
  const file = [...event.dataTransfer.files].find(f => f.type.startsWith('image/'));
  if (file) { event.preventDefault(); readPhoto(file); }
});

async function readPhoto(file) {
  if (!file) return;
  const request = ++photoRequest, atRevision = revision;
  const url = URL.createObjectURL(file);
  clearTimeout(timer);
  try {
    const picture = new Image();
    picture.src = url;
    await picture.decode();
    if (request !== photoRequest || atRevision !== revision) return;
    const source = document.createElement('canvas');
    source.width = picture.naturalWidth; source.height = picture.naturalHeight;
    const ctx = source.getContext('2d', {willReadFrequently: true});
    ctx.fillStyle = 'white'; ctx.fillRect(0, 0, source.width, source.height);
    ctx.drawImage(picture, 0, 0);
    const rgba = ctx.getImageData(0, 0, source.width, source.height).data;
    const gray = Array.from({length: source.height}, (_, y) => Uint8Array.from({length: source.width}, (_, x) => {
      const i = 4 * (y * source.width + x);
      return Math.floor((299 * rgba[i] + 587 * rgba[i + 1] + 114 * rgba[i + 2] + 500) / 1000);
    }));
    const traced = image_to_strokes(gray), rect = canvas.getBoundingClientRect();
    strokes.length = 0; current = null; pointer = null; revision++; queued = false;
    if (traced.length) {
      const bounds = traced.flat().reduce((b, [x, y]) => [Math.min(b[0], x), Math.min(b[1], y), Math.max(b[2], x), Math.max(b[3], y)], [Infinity, Infinity, -Infinity, -Infinity]);
      const width = bounds[2] - bounds[0], height = bounds[3] - bounds[1];
      const scale = Math.min(Math.max(1, rect.width - 40) / Math.max(1, width), Math.max(1, rect.height - 40) / Math.max(1, height));
      const dx = (rect.width - width * scale) / 2, dy = (rect.height - height * scale) / 2;
      for (const stroke of traced) strokes.push(stroke.map(([x, y, t]) => [(x - bounds[0]) * scale + dx, (y - bounds[1]) * scale + dy, t]));
    }
    actions.length = 0; for (let k = 0; k < strokes.length; k++) actions.push('add');
    $('photo-note').hidden = false;
    redraw(); resetResult();
    if (!traced.length) $('status').textContent = 'No ink found in this photo. Try a closer, clearer picture.';
    schedule();
  } catch (error) {
    if (request === photoRequest) $('status').textContent = `Could not read photo: ${error.message}`;
  } finally {
    URL.revokeObjectURL(url);
    if (request === photoRequest) $('photo-file').value = '';
  }
}
$('auto').onchange = schedule;
$('recognise').onclick = recognise;
$('copy').onclick = async () => {
  if (!$('latex').value) return;
  try { await navigator.clipboard.writeText($('latex').value); $('status').textContent = 'LaTeX copied.'; }
  catch { $('latex').select(); $('status').textContent = 'Select and copy the LaTeX above.'; }
};
new ResizeObserver(redraw).observe(canvas);

function resetResult() {
  $('latex').value = ''; $('rendered').textContent = 'Your equation will appear here.'; $('timing').textContent = '';
  if (encoder) $('status').textContent = 'Ready · Write an expression to begin.';
}

async function recognise() {
  clearTimeout(timer);
  if (!encoder || !strokes.length || current) return;
  if (busy) { queued = true; return; }
  busy = true; queued = false;
  $('recognise').disabled = true;
  const atRevision = revision, start = performance.now();
  $('status').textContent = 'Reading your handwriting…';
  const tensors = [];
  const track = value => { tensors.push(value); return value; };
  // One line of ink to tokens (null when the ink changed meanwhile).
  async function read(lineStrokes) {
    const rows = features(lineStrokes);
    const result = await encoder.run({
      points: track(new ort.Tensor('float32', Float32Array.from(rows.flat()), [1, rows.length, 8])),
      point_mask: track(new ort.Tensor('bool', new Uint8Array(rows.length).fill(1), [1, rows.length]))
    });
    const memory = track(result.memory), memoryMask = track(result.memory_mask);
    const tokens = [1], output = [];
    for (let t = 0; t < 160; t++) {
      if (revision !== atRevision) return null;
      const previous = new ort.Tensor('int64', BigInt64Array.from(tokens, BigInt), [1, tokens.length]);
      let prediction;
      try { prediction = await decoder.run({previous_tokens: previous, memory, memory_mask: memoryMask}); }
      finally { previous.dispose(); }
      const logits = prediction.logits.data;
      let next = 2;
      for (let i = 3; i < logits.length; i++) if (logits[i] > logits[next]) next = i;
      prediction.logits.dispose();
      if (next === 2) break;
      output.push(vocabulary[next]); tokens.push(next);
    }
    return output;
  }
  try {
    // The model reads one line of maths: separate lines (a system of equations) are read one by one.
    const parts = [];
    for (const line of split_lines(strokes)) {
      const output = await read(line);
      if (output === null) return;
      parts.push(toLatex(tidy(output)));
    }
    if (revision !== atRevision) return;
    const latex = parts.length > 1 ? `\\begin{gathered}${parts.join('\\\\')}\\end{gathered}` : parts[0];
    $('latex').value = latex;
    katex.render(latex, $('rendered'), {throwOnError: false, displayMode: true, trust: false});
    $('timing').textContent = `${Math.round(performance.now() - start)} ms`;
    $('status').textContent = offline ? 'Ready · Available offline · Your ink stays on this device.' : 'Ready · Your ink stays on this device.';
  } catch (error) {
    $('status').textContent = `Recognition failed: ${error.message}`;
  } finally {
    for (const tensor of tensors) tensor.dispose();
    busy = false; $('recognise').disabled = !encoder;
    if (queued && !current) { queued = false; schedule(); }
  }
}

async function initialize() {
  if (document.readyState !== 'complete') await new Promise(resolve => window.addEventListener('load', resolve, {once: true}));
  if ('serviceWorker' in navigator) {
    try {
      const registration = await navigator.serviceWorker.register('./sw.js');
      if (!registration.active) {
        const worker = registration.installing || registration.waiting;
        if (!worker) throw new Error('Offline cache unavailable');
        await new Promise((resolve, reject) => {
          const changed = () => {
            if (worker.state === 'activated') resolve();
            if (worker.state === 'redundant') reject(new Error('Offline cache unavailable'));
          };
          worker.addEventListener('statechange', changed); changed();
        });
      }
      if (!navigator.serviceWorker.controller) await new Promise(resolve => navigator.serviceWorker.addEventListener('controllerchange', resolve, {once: true}));
      offline = true;
    } catch { /* Recognition still works when offline storage is unavailable. */ }
  }
  try {
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/';
    const response = await fetch('./models/vocab.json');
    if (!response.ok) throw new Error('Model not published');
    vocabulary = (await response.json()).tokens;
    [encoder, decoder] = await Promise.all([
      ort.InferenceSession.create('./models/encoder.onnx', {executionProviders: ['wasm']}),
      ort.InferenceSession.create('./models/decoder.onnx', {executionProviders: ['wasm']})
    ]);
    $('recognise').disabled = false;
    $('status').textContent = offline ? 'Ready · Available offline · Write an expression to begin.' : 'Ready · Write an expression to begin.';
    schedule();
  } catch {
    $('status').textContent = 'The recognition model is not available yet. You can still try the drawing canvas.';
  }
}
redraw(); initialize();
