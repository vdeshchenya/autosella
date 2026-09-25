#!/usr/bin/env python3
"""Patch Ralph's verified stream reader without retaining every read result.

The original function was captured from deployed ralph.ts SHA256
bd90aed9d3d3d517fbee019b232ed7000d9b2ff9b60ee7e5579c8784af21bcf8.
Only the complete known function (or its exact patched form) is accepted;
upstream changes require review instead of a best-effort replacement.

For an existing runtime image, copy this file into a derived image and run:
    python3 /path/to/patch_ralph.py /opt/ralph/ralph.ts
No model, provider session, or research data is needed.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


STREAM_START = "  const streamText = async (\n"
STREAM_END = "\n  const heartbeatTimer = setInterval("
ORIGINAL_STREAM_TEXT = r'''  const streamText = async (
    stream: ReadableStream<Uint8Array> | null,
    onText: (chunk: string) => void,
    isError: boolean,
  ) => {
    if (!stream) return;
    const reader = stream.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    
    // Create abort promise if signal provided
    const abortPromise = options.abortSignal
      ? new Promise<{ value: undefined; done: true }>((resolve, reject) => {
          const handler = () => {
            options.abortSignal?.removeEventListener('abort', handler);
            resolve({ value: undefined, done: true });
          };
          options.abortSignal.addEventListener('abort', handler);
        })
      : new Promise<{ value: undefined; done: true }>(() => {});
    
    while (true) {
      const result = options.abortSignal
        ? await Promise.race([reader.read(), abortPromise])
        : await reader.read();
      
      const { value, done } = result;
      if (done) break;
      const text = decoder.decode(value, { stream: true });
      if (text.length > 0) {
        onText(text);
        buffer += text;
        const lines = buffer.split(/\r?\n/);
        buffer = lines.pop() ?? "";
        for (const line of lines) {
          handleLine(line, isError);
        }
      }
    }
    const flushed = decoder.decode();
    if (flushed.length > 0) {
      onText(flushed);
      buffer += flushed;
    }
    if (buffer.length > 0) {
      handleLine(buffer, isError);
    }
  };
'''

PATCHED_STREAM_TEXT = r'''  const streamText = async (
    stream: ReadableStream<Uint8Array> | null,
    onText: (chunk: string) => void,
    isError: boolean,
  ) => {
    if (!stream) return;
    const reader = stream.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    const abortSignal = options.abortSignal;
    // One listener cancels a pending read without retaining earlier chunks.
    const cancelReader = () => {
      void reader.cancel().catch(() => {});
    };
    abortSignal?.addEventListener("abort", cancelReader, { once: true });
    try {
      if (abortSignal?.aborted) cancelReader();
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        const text = decoder.decode(value, { stream: true });
        if (text.length > 0) {
          onText(text);
          buffer += text;
          const lines = buffer.split(/\r?\n/);
          buffer = lines.pop() ?? "";
          for (const line of lines) {
            handleLine(line, isError);
          }
        }
      }
      const flushed = decoder.decode();
      if (flushed.length > 0) {
        onText(flushed);
        buffer += flushed;
      }
      if (buffer.length > 0) {
        handleLine(buffer, isError);
      }
    } finally {
      abortSignal?.removeEventListener("abort", cancelReader);
      reader.releaseLock();
    }
  };
'''


def patched_source(source: str) -> str:
    """Return a patched source, rejecting unknown or ambiguous function shapes."""
    if source.count(STREAM_START) != 1 or source.count(STREAM_END) != 1:
        raise ValueError("Unknown Ralph source: expected one streamText/heartbeat boundary")
    start = source.index(STREAM_START)
    end = source.index(STREAM_END)
    function = source[start:end]
    if function == PATCHED_STREAM_TEXT:
        return source
    if function != ORIGINAL_STREAM_TEXT:
        raise ValueError("Unknown Ralph streamText implementation; refusing to patch")
    return source[:start] + PATCHED_STREAM_TEXT + source[end:]


def patch_file(path: Path) -> bool:
    """Validate before writing; preserve bytes outside the function and file mode."""
    before = path.read_bytes()
    after = patched_source(before.decode("utf-8")).encode("utf-8")
    changed = before != after
    if changed:
        path.write_bytes(after)
    status = "patched" if changed else "already patched"
    print(f"Ralph stream reader {status}: {path}")
    print(f"SHA256 before={hashlib.sha256(before).hexdigest()} after={hashlib.sha256(after).hexdigest()}")
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Path to the pinned Ralph ralph.ts")
    args = parser.parse_args()
    try:
        patch_file(args.path)
    except (OSError, UnicodeError, ValueError) as exc:
        parser.exit(1, f"Ralph stream patch failed: {exc}\n")


if __name__ == "__main__":
    main()
