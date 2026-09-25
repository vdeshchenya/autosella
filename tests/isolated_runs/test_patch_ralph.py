"""Guard the upstream patch and execute its actual TypeScript function with Bun."""
from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from isolated_runs.patch_ralph import (
    ORIGINAL_STREAM_TEXT,
    PATCHED_STREAM_TEXT,
    STREAM_END,
    patch_file,
    patched_source,
)


PREFIX = "// untouched prefix\n"
SUFFIX = STREAM_END + "() => {}, 1000);\n// untouched suffix\n"
ORIGINAL = PREFIX + ORIGINAL_STREAM_TEXT + SUFFIX
BUN = os.environ.get("BUN_BIN") or shutil.which("bun")
if not BUN and (Path.home() / ".bun/bin/bun").is_file():
    BUN = str(Path.home() / ".bun/bin/bun")


class RalphPatchTests(unittest.TestCase):
    def test_exact_replacement_and_idempotence(self):
        patched = patched_source(ORIGINAL)
        self.assertEqual(patched, PREFIX + PATCHED_STREAM_TEXT + SUFFIX)
        self.assertEqual(patched_source(patched), patched)

    def test_unknown_partial_and_duplicate_shapes_fail_without_writing(self):
        cases = [
            "unrelated source",
            ORIGINAL.replace("buffer.split", "buffer.trim().split"),
            ORIGINAL.replace("    let buffer = \"\";", "    let buffer = 'changed';"),
            ORIGINAL + ORIGINAL,
            ORIGINAL + PATCHED_STREAM_TEXT,
            (PREFIX + PATCHED_STREAM_TEXT + SUFFIX).replace("reader.releaseLock();", ""),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "ralph.ts"
            for source in cases:
                with self.subTest(source=source[:60]):
                    target.write_text(source)
                    before = target.read_bytes()
                    with self.assertRaises(ValueError):
                        patch_file(target)
                    self.assertEqual(target.read_bytes(), before)

    def test_file_preserves_mode_and_reports_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "ralph.ts"
            target.write_text(ORIGINAL)
            target.chmod(0o755)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertTrue(patch_file(target))
                self.assertFalse(patch_file(target))
            self.assertEqual(target.stat().st_mode & 0o777, 0o755)
            self.assertIn("SHA256 before=", output.getvalue())
            self.assertIn("already patched", output.getvalue())

    @unittest.skipUnless(BUN, "Bun is required to test the deployed TypeScript reader")
    def test_actual_patched_reader_and_retention_positive_control(self):
        harness = Path(__file__).with_name("ralph_stream_regression.ts")
        with tempfile.TemporaryDirectory() as temporary:
            module = Path(temporary) / "stream.ts"
            # Compile the exact function emitted by the patcher, not a second
            # implementation of the proposed cancellation mechanism.
            patched = patched_source(ORIGINAL)
            function = patched[len(PREFIX):patched.index(STREAM_END)]
            module.write_text(
                "export function makeConsumer(options: any, handleLine: any) {\n"
                + function + "\nreturn streamText;\n}\n"
            )
            checked = subprocess.run(
                [BUN, str(harness), str(module), "patched"],
                check=True, capture_output=True, text=True, timeout=30,
            )
            patched_results = json.loads(checked.stdout)
            self.assertEqual(patched_results["checks"], 13)
            self.assertLess(patched_results["heapGrowthMiB"], 16)
            # The same allocation workload must reveal the original leak,
            # otherwise a low patched measurement would not be meaningful.
            module.write_text(
                "export function makeConsumer(options: any, handleLine: any) {\n"
                + ORIGINAL_STREAM_TEXT + "\nreturn streamText;\n}\n"
            )
            control = subprocess.run(
                [BUN, str(harness), str(module), "original"],
                check=True, capture_output=True, text=True, timeout=30,
            )
            original_results = json.loads(control.stdout)
            self.assertGreater(original_results["heapGrowthMiB"], 96)
            print(json.dumps({"patched": patched_results, "original": original_results}))


if __name__ == "__main__":
    unittest.main()
