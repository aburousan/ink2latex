// Optional integration check: node tests/test_demo.mjs /path/to/demo-with-models
// Needs Node 22+ and Chrome (set CHROME_PATH on Linux). No npm dependencies.
import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {createServer} from 'node:http';
import {mkdtemp, readFile, rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import path from 'node:path';

const root = path.resolve(process.argv[2] || 'demo');
for (const asset of ['encoder.onnx', 'decoder.onnx', 'vocab.json']) {
  await readFile(path.join(root, 'models', asset));
}
const chrome = process.env.CHROME_PATH || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const profile = await mkdtemp(path.join(tmpdir(), 'ink2latex-browser-'));
const mime = {'.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.json': 'application/json', '.onnx': 'application/octet-stream'};
const server = createServer(async (request, response) => {
  try {
    const pathname = decodeURIComponent(new URL(request.url, 'http://localhost').pathname);
    const filename = path.resolve(root, '.' + (pathname.endsWith('/') ? pathname + 'index.html' : pathname));
    if (!filename.startsWith(root + path.sep)) throw new Error('Invalid path');
    const data = await readFile(filename);
    response.writeHead(200, {'Content-Type': mime[path.extname(filename)] || 'application/octet-stream'});
    response.end(data);
  } catch { response.writeHead(404); response.end('Not found'); }
});
await new Promise((resolve, reject) => {
  server.once('error', reject);
  server.listen(0, '127.0.0.1', resolve);
});
const origin = `http://127.0.0.1:${server.address().port}`;
const browser = spawn(chrome, ['--headless=new', '--remote-debugging-port=0', `--user-data-dir=${profile}`,
  '--no-first-run', '--disable-background-networking', 'about:blank'], {stdio: ['ignore', 'ignore', 'pipe']});
let socket;
try {
  const endpoint = await new Promise((resolve, reject) => {
    let stderr = '';
    browser.on('error', reject);
    browser.on('exit', code => reject(new Error(`Chrome exited ${code}: ${stderr}`)));
    browser.stderr.on('data', data => {
      stderr += data;
      const match = stderr.match(/DevTools listening on (ws:\/\/[^\s]+)/);
      if (match) resolve(match[1]);
    });
  });
  socket = new WebSocket(endpoint);
  await new Promise((resolve, reject) => { socket.onopen = resolve; socket.onerror = reject; });
  let nextId = 1;
  const pending = new Map(), errors = [], requests = [];
  socket.onmessage = event => {
    const message = JSON.parse(event.data);
    if (message.id) {
      const promise = pending.get(message.id);
      pending.delete(message.id);
      if (message.error) promise.reject(new Error(JSON.stringify(message.error)));
      else promise.resolve(message.result);
    }
    if (message.method === 'Runtime.exceptionThrown') errors.push(message.params.exceptionDetails);
    if (message.method === 'Network.requestWillBeSent') requests.push(message.params.request.url);
  };
  const send = (method, params = {}, sessionId) => new Promise((resolve, reject) => {
    const id = nextId++;
    pending.set(id, {resolve, reject});
    socket.send(JSON.stringify({id, method, params, sessionId}));
  });
  const {targetId} = await send('Target.createTarget', {url: 'about:blank'});
  const {sessionId} = await send('Target.attachToTarget', {targetId, flatten: true});
  const page = (method, params) => send(method, params, sessionId);
  const evaluate = async expression => {
    const result = await page('Runtime.evaluate', {expression, returnByValue: true, awaitPromise: true});
    if (result.exceptionDetails) throw new Error(JSON.stringify(result.exceptionDetails));
    return result.result.value;
  };
  const waitFor = async expression => {
    const deadline = Date.now() + 45000;
    while (Date.now() < deadline) {
      if (await evaluate(expression)) return;
      await new Promise(resolve => setTimeout(resolve, 100));
    }
    throw new Error(`Timed out: ${expression}; status: ${await evaluate('document.querySelector("#status")?.textContent')}`);
  };
  await page('Runtime.enable');
  await page('Page.enable');
  await page('Emulation.setDeviceMetricsOverride', {width: 1200, height: 1000, deviceScaleFactor: 1, mobile: false});
  await page('Network.enable');
  await page('Network.setCacheDisabled', {cacheDisabled: true});
  await page('Page.navigate', {url: origin + '/'});
  const ready = 'document.querySelector("#status")?.textContent.includes("Available offline") && !document.querySelector("#recognise").disabled';
  await waitFor(ready);
  console.log('Browser models and offline cache loaded.');
  const draw = async () => {
    await evaluate('document.querySelector("canvas").scrollIntoView({block: "center"})');
    const box = await evaluate('(()=>{const r=document.querySelector("canvas").getBoundingClientRect();return {x:r.x,y:r.y}})()');
    await page('Input.dispatchMouseEvent', {type: 'mousePressed', x: box.x + 80, y: box.y + 100, button: 'left', clickCount: 1});
    await page('Input.dispatchMouseEvent', {type: 'mouseMoved', x: box.x + 100, y: box.y + 140, button: 'left', buttons: 1});
    await page('Input.dispatchMouseEvent', {type: 'mouseReleased', x: box.x + 140, y: box.y + 100, button: 'left', clickCount: 1});
  };
  await draw();
  await waitFor('document.querySelector("#timing").textContent.endsWith("ms")');
  assert.ok(await evaluate('!!document.querySelector("#rendered .katex")'));
  const onlineTiming = await evaluate('document.querySelector("#timing").textContent');
  // Eraser: a sweep over the stroke removes it; Undo brings it back.
  await evaluate('document.querySelector("#eraser").click()');
  {
    const box = await evaluate('(()=>{const r=document.querySelector("canvas").getBoundingClientRect();return {x:r.x,y:r.y}})()');
    await page('Input.dispatchMouseEvent', {type: 'mousePressed', x: box.x + 60, y: box.y + 140, button: 'left', clickCount: 1});
    await page('Input.dispatchMouseEvent', {type: 'mouseMoved', x: box.x + 100, y: box.y + 140, button: 'left', buttons: 1});
    await page('Input.dispatchMouseEvent', {type: 'mouseReleased', x: box.x + 110, y: box.y + 140, button: 'left', clickCount: 1});
  }
  assert.equal(await evaluate('document.querySelector("#hint").hidden'), false);
  assert.equal(await evaluate('document.querySelector("#pen").classList.contains("on")'), false);
  await evaluate('document.querySelector("#pen").click()');
  assert.equal(await evaluate('document.querySelector("#eraser").classList.contains("on")'), false);
  assert.equal(await evaluate('document.querySelector("#pen").classList.contains("on")'), true);
  await evaluate('document.querySelector("#undo").click()');
  assert.equal(await evaluate('document.querySelector("#hint").hidden'), true);
  await evaluate('document.querySelector("#undo").click()');
  assert.equal(await evaluate('document.querySelector("#hint").hidden'), false);

  const loadPhoto = async () => {
    const {root: documentNode} = await page('DOM.getDocument');
    const {nodeId} = await page('DOM.querySelector', {nodeId: documentNode.nodeId, selector: '#photo-file'});
    await page('DOM.setFileInputFiles', {nodeId, files: [path.resolve('tests/fixtures/photo_cases/crossed_t.png')]});
    await waitFor('!document.querySelector("#photo-note").hidden');
    assert.equal(await evaluate('document.querySelector("#hint").hidden'), true);
  };
  await evaluate('document.querySelector("#auto").checked = false');
  await loadPhoto();
  await evaluate('document.querySelector("#undo").click()');
  assert.equal(await evaluate('document.querySelector("#hint").hidden'), true);
  await evaluate('document.querySelector("#recognise").click()');
  await waitFor('document.querySelector("#timing").textContent.endsWith("ms")');
  await evaluate('document.querySelector("#clear").click()');
  assert.equal(await evaluate('document.querySelector("#photo-note").hidden'), true);

  // A picture pasted from the clipboard (Ctrl+V) and one dropped on the box read like a photo.
  const picture = `(async () => {
    const c = document.createElement('canvas'); c.width = 240; c.height = 120;
    const x = c.getContext('2d'); x.fillStyle = 'white'; x.fillRect(0, 0, 240, 120);
    x.strokeStyle = 'black'; x.lineWidth = 5; x.lineCap = 'round';
    x.beginPath(); x.moveTo(40, 30); x.lineTo(90, 90); x.moveTo(90, 30); x.lineTo(40, 90); x.stroke();
    x.beginPath(); x.moveTo(130, 60); x.lineTo(200, 60); x.stroke();
    const blob = await new Promise(r => c.toBlob(r, 'image/png'));
    const data = new DataTransfer(); data.items.add(new File([blob], 'shot.png', {type: 'image/png'}));
    return data;
  })()`;
  await evaluate(`${picture}.then(data => document.dispatchEvent(new ClipboardEvent('paste', {clipboardData: data, bubbles: true})))`);
  await waitFor('!document.querySelector("#photo-note").hidden && document.querySelector("#hint").hidden');
  await evaluate('document.querySelector("#clear").click()');
  await evaluate(`${picture}.then(data => document.querySelector("#ink").dispatchEvent(new DragEvent('drop', {dataTransfer: data, bubbles: true, cancelable: true})))`);
  await waitFor('!document.querySelector("#photo-note").hidden && document.querySelector("#hint").hidden');
  await evaluate('document.querySelector("#clear").click()');

  // Disable HTTP cache and stop the origin: reload must use service-worker assets.
  await new Promise(resolve => server.close(resolve));
  await page('Network.emulateNetworkConditions', {offline: true, latency: 0, downloadThroughput: 0, uploadThroughput: 0});
  await page('Page.reload', {ignoreCache: true});
  await waitFor(ready);
  await loadPhoto();
  await waitFor('document.querySelector("#timing").textContent.endsWith("ms")');
  await evaluate('document.querySelector("#clear").click()');
  await evaluate('document.querySelector("#auto").checked = false');
  await draw();
  await evaluate('document.querySelector("#recognise").click()');
  await waitFor('document.querySelector("#timing").textContent.endsWith("ms")');
  const offlineTiming = await evaluate('document.querySelector("#timing").textContent');
  await evaluate('document.querySelector("#clear").click()');
  assert.equal(await evaluate('document.querySelector("#latex").value'), '');
  assert.equal(await evaluate('document.querySelector("#hint").hidden'), false);
  assert.deepEqual(errors, []);
  assert.ok(requests.every(url => url.startsWith(origin) || url.startsWith('https://cdn.jsdelivr.net/') || url.startsWith('data:') || url.startsWith('blob:' + origin)));
  console.log(JSON.stringify({passed: true, onlineTiming, offlineTiming, checks: ['WASM inference', 'auto-recognition', 'KaTeX', 'eraser and undo of an erase', 'undo', 'photo replacement and stroke undo', 'photo note cleared', 'clipboard paste', 'drag and drop', 'offline reload with origin stopped', 'offline photo and auto-recognition', 'manual recognition', 'clear', 'no page errors', 'only local/CDN requests']}));
} finally {
  socket?.close();
  browser.kill('SIGTERM');
  await new Promise(resolve => { if (browser.exitCode !== null || browser.signalCode !== null) resolve(); else browser.once('exit', resolve); });
  if (server.listening) await new Promise(resolve => server.close(resolve));
  await rm(profile, {recursive: true, force: true});
}
