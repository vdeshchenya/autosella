// Execute the exact patched streamText body. No model, credentials, or data.
import assert from "node:assert/strict";
import { heapStats } from "bun:jsc";

const { makeConsumer } = await import(process.argv[2]);
const mode = process.argv[3];
let checks = 0;
const unhandled: unknown[] = [];
process.on("unhandledRejection", error => unhandled.push(error));

function observedController() {
  const controller = new AbortController();
  const signal = controller.signal;
  const add = signal.addEventListener.bind(signal);
  const remove = signal.removeEventListener.bind(signal);
  const listeners = new Set<unknown>();
  signal.addEventListener = ((type: string, listener: any, options: any) => {
    if (type === "abort") listeners.add(listener);
    add(type, listener, options);
  }) as any;
  signal.removeEventListener = ((type: string, listener: any, options: any) => {
    if (type === "abort") listeners.delete(listener);
    remove(type, listener, options);
  }) as any;
  return { controller, signal, listeners };
}

if (mode === "patched") {
  for (const withSignal of [false, true]) {
    const tracked = observedController();
    const lines: unknown[] = [];
    let text = "";
    const consume = makeConsumer(
      { abortSignal: withSignal ? tracked.signal : undefined },
      (line: string, error: boolean) => lines.push([line, error]),
    );
    const stream = new ReadableStream<Uint8Array>({ start(controller) {
      // Bytewise chunks split UTF-8 characters and CRLF boundaries.
      for (const byte of new TextEncoder().encode("€\r\n\nlast")) {
        controller.enqueue(new Uint8Array([byte]));
      }
      controller.close();
    } });
    await consume(stream, (chunk: string) => { text += chunk; }, true);
    assert.equal(text, "€\r\n\nlast");
    assert.deepEqual(lines, [["€", true], ["", true], ["last", true]]);
    assert.equal(stream.locked, false);
    assert.equal(tracked.listeners.size, 0);
    checks++;
  }

  const nullTracked = observedController();
  await makeConsumer({ abortSignal: nullTracked.signal }, () => assert.fail())(
    null, () => assert.fail(), false,
  );
  assert.equal(nullTracked.listeners.size, 0);
  checks++;

  for (const failure of ["read", "onText", "handleLine"]) {
    const tracked = observedController();
    const error = new Error(failure);
    const stream = new ReadableStream<Uint8Array>({ start(controller) {
      if (failure === "read") controller.error(error);
      else {
        controller.enqueue(new TextEncoder().encode("line\n"));
        controller.close();
      }
    } });
    const consume = makeConsumer({ abortSignal: tracked.signal }, () => {
      if (failure === "handleLine") throw error;
    });
    await assert.rejects(consume(stream, () => {
      if (failure === "onText") throw error;
    }, false), candidate => candidate === error);
    assert.equal(stream.locked, false);
    assert.equal(tracked.listeners.size, 0);
    checks++;
  }

  for (const abortMode of ["already", "pending", "cancel-reject", "partial"]) {
    const tracked = observedController();
    let cancelled = 0;
    const lines: string[] = [];
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        if (abortMode === "already") controller.enqueue(new TextEncoder().encode("discard\n"));
        if (abortMode === "partial") controller.enqueue(new TextEncoder().encode("partial"));
      },
      cancel() {
        cancelled++;
        if (abortMode === "cancel-reject") return Promise.reject(new Error("cancel rejected"));
      },
    });
    if (abortMode === "already") tracked.controller.abort();
    const timer = abortMode === "already" ? null : setTimeout(() => tracked.controller.abort(), 10);
    const started = performance.now();
    await makeConsumer({ abortSignal: tracked.signal }, (line: string) => lines.push(line))(
      stream, () => {}, false,
    );
    if (timer) clearTimeout(timer);
    await Bun.sleep(0);
    assert.ok(performance.now() - started < 2000);
    assert.equal(cancelled, 1);
    assert.deepEqual(lines, abortMode === "partial" ? ["partial"] : []);
    assert.equal(stream.locked, false);
    assert.equal(tracked.listeners.size, 0);
    checks++;
  }

  // Exercise actual Bun subprocess stdout, including a read blocked on a pipe.
  for (const pipeMode of ["already", "pending", "eof"]) {
    const tracked = observedController();
    const proc = Bun.spawn([process.execPath, "-e", pipeMode === "eof"
      ? "process.stdout.write('normal\\n');" : "await Bun.sleep(10000);"], {
      stdout: "pipe", stderr: "ignore",
    });
    if (pipeMode === "already") tracked.controller.abort();
    const timer = pipeMode === "pending" ? setTimeout(() => tracked.controller.abort(), 10) : null;
    let text = "";
    try {
      const started = performance.now();
      await makeConsumer({ abortSignal: tracked.signal }, () => {})(
        proc.stdout, (chunk: string) => { text += chunk; }, false,
      );
      assert.ok(performance.now() - started < 2000);
      assert.equal(text, pipeMode === "eof" ? "normal\n" : "");
      assert.equal(proc.stdout.locked, false);
      assert.equal(tracked.listeners.size, 0);
    } finally {
      if (timer) clearTimeout(timer);
      proc.kill();
      await proc.exited;
    }
    checks++;
  }
}

// Keep the signal alive after EOF, as the surrounding iteration does. The
// original listener retains its pending promise and all completed read races.
const memoryController = new AbortController();
(globalThis as any).memoryController = memoryController;
Bun.gc(true);
const before = heapStats().heapSize;
let chunks = 0;
const stream = new ReadableStream<Uint8Array>({ pull(controller) {
  if (chunks++ === 600) { controller.close(); return; }
  const backing = new Uint8Array(256 * 1024);
  backing[0] = 120;
  backing[1] = 10;
  controller.enqueue(backing.subarray(0, 2));
} });
await makeConsumer({ abortSignal: memoryController.signal }, () => {})(stream, () => {}, false);
await Bun.sleep(0);
Bun.gc(true);
const heapGrowthMiB = (heapStats().heapSize - before) / 1024 / 1024;
assert.equal(chunks, 601);
assert.deepEqual(unhandled, []);
console.log(JSON.stringify({ mode, checks, chunks: chunks - 1, heapGrowthMiB, bunVersion: Bun.version }));
