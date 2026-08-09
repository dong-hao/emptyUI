import { expect, test } from "vitest";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { PORT_COLORS } from "./catalog";

test("no producer-less port types are advertised (issues.md L2)", () => {
  // the `selection` type returns with its producer (M4-deferred); until
  // then the palette must not advertise a color for it either
  expect("selection" in PORT_COLORS).toBe(false);
});

test("PORT_COLORS mirrors the backend's CORE_PORT_TYPES (issues.md M5)", () => {
  // the frontend table is the pre-catalog fallback (DESIGN §2's handwritten
  // types) — this pin reads the Python source so the mirror cannot drift
  const py = readFileSync(
    resolve(process.cwd(), "../emptyui/src/emptyui/registry.py"),
    "utf8",
  );
  const block = py.match(/CORE_PORT_TYPES[^=]*= \{([\s\S]*?)\n\}/);
  expect(block).toBeTruthy();
  for (const m of block![1].matchAll(/"([\w.]+)": \{"color": "(#[0-9a-fA-F]+)"/g)) {
    const [, tag, color] = m;
    expect(PORT_COLORS[tag], `port type ${tag}`).toBe(color);
  }
});
