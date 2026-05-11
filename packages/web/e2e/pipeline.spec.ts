import { expect, test, type Locator, type Page } from "@playwright/test";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const CSV_PATH = path.resolve(
  here,
  "..",
  "..",
  "..",
  "examples",
  "data",
  "150Hz_synthetic_CH4.csv",
);

async function addNode(page: Page, name: string) {
  await page.getByRole("button", { name, exact: true }).click();
}

function node(page: Page, title: string): Locator {
  return page.locator(".mt-node", { hasText: title }).first();
}

/** Node flow position from the wrapper's translate() (React Flow coords). */
async function flowPosition(page: Page, title: string) {
  return page.evaluate((t) => {
    const nodes = Array.from(document.querySelectorAll(".react-flow__node"));
    const n = nodes.find(
      (n) => n.querySelector(".mt-node-header")?.textContent === t,
    ) as HTMLElement | undefined;
    const m = n?.style.transform.match(/translate\(([-\d.]+)px,\s*([-\d.]+)px\)/);
    return m ? { x: parseFloat(m[1]), y: parseFloat(m[2]) } : { x: 0, y: 0 };
  }, title);
}

/** Current viewport zoom (flow units are scaled by this on screen). */
async function viewportZoom(page: Page): Promise<number> {
  return page.evaluate(() => {
    const vp = document.querySelector(".react-flow__viewport") as HTMLElement | null;
    const m = vp?.style.transform.match(/scale\(([-\d.]+)\)/);
    return m ? parseFloat(m[1]) : 1;
  });
}

/** React Flow's Fit View animates the viewport; wait until the transform
 *  stops changing so placement/zoom reads and handle drags are accurate. */
async function settleViewport(page: Page) {
  const read = () =>
    page.evaluate(() => {
      const vp = document.querySelector(".react-flow__viewport") as HTMLElement | null;
      return vp?.style.transform ?? "";
    });
  let prev = await read();
  for (let i = 0; i < 30; i++) {
    await page.waitForTimeout(100);
    const cur = await read();
    if (cur === prev) return;
    prev = cur;
  }
}

/** Drag a node by a flow-space delta (pointer travel = delta * zoom). */
async function dragNodeBy(page: Page, title: string, dx: number, dy: number) {
  const zoom = await viewportZoom(page);
  const hb = (await node(page, title).locator(".mt-node-header").boundingBox())!;
  const sx = hb.x + hb.width / 2;
  const sy = hb.y + hb.height / 2;
  await page.mouse.move(sx, sy);
  await page.mouse.down();
  await page.mouse.move(sx + dx * zoom, sy + dy * zoom, { steps: 10 });
  await page.mouse.up();
}

/** If another node covers the given node's header, nudge the blocker aside. */
async function ensureHeaderHittable(page: Page, title: string) {
  for (let attempt = 0; attempt < 3; attempt++) {
    const blocked = await page.evaluate((t) => {
      const nodes = Array.from(document.querySelectorAll(".react-flow__node"));
      const n = nodes.find(
        (n) => n.querySelector(".mt-node-header")?.textContent === t,
      );
      const h = n?.querySelector(".mt-node-header") as HTMLElement | null;
      if (!h) return null;
      const r = h.getBoundingClientRect();
      const el = document.elementFromPoint(r.x + r.width / 2, r.y + r.height / 2);
      if (el && (h === el || h.contains(el) || el.contains(h))) return null;
      const block = el?.closest(".react-flow__node");
      return block?.querySelector(".mt-node-header")?.textContent ?? null;
    }, title);
    if (!blocked) return;
    await dragNodeBy(page, blocked, 160, 40);
  }
}

/** New nodes spawn at random positions and may overlap; spread them apart.
 *  Raw mouse drag (locator.dragTo requires the end point to stay over the
 *  target element — the canvas is narrower than the window). Node drags are
 *  relative: pointer travel / zoom = flow travel, so the pointer delta is
 *  computed from the current flow position. We track the position
 *  arithmetically (a drag moves the node by exactly delta/zoom) instead of
 *  re-reading `style.transform` — React commits it asynchronously, and a
 *  stale read makes corrective drags overshoot/oscillate. */
async function placeNode(page: Page, title: string, x: number, y: number) {
  await ensureHeaderHittable(page, title);
  const pos = await flowPosition(page, title);
  let ex = pos.x;
  let ey = pos.y;
  for (let attempt = 0; attempt < 6; attempt++) {
    const dx = x - ex;
    const dy = y - ey;
    if (Math.abs(dx) < 2 && Math.abs(dy) < 2) return;
    await dragNodeBy(page, title, dx, dy);
    ex += dx;
    ey += dy;
  }
}

/** React Flow drags can occasionally stop short of the target flow position
 *  (random-spawn dependent), leaving a handle under a neighbouring node.
 *  Before connecting, verify BOTH handles are hittable; if a node covers one,
 *  nudge that node aside (up to 3 attempts). */
async function ensureHandleHittable(
  page: Page,
  title: string,
  port: string,
  handlepos: "right" | "left",
) {
  for (let attempt = 0; attempt < 3; attempt++) {
    const blocked = await page.evaluate(
      ([t, p, hp]) => {
        const nodes = Array.from(document.querySelectorAll(".react-flow__node"));
        const src = nodes.find(
          (n) => n.querySelector(".mt-node-header")?.textContent === t,
        );
        const handle = src?.querySelector(
          `.mt-port[title="${p}"][data-handlepos="${hp}"]`,
        ) as HTMLElement | null;
        if (!handle) return null;
        const r = handle.getBoundingClientRect();
        const el = document.elementFromPoint(r.x + r.width / 2, r.y + r.height / 2);
        if (el && (handle === el || handle.contains(el) || el.contains(handle))) {
          return null;
        }
        const block = el?.closest(".react-flow__node");
        return block?.querySelector(".mt-node-header")?.textContent ?? null;
      },
      [title, port, handlepos] as const,
    );
    if (!blocked) return;
    const zoom = await viewportZoom(page);
    const hb = (await node(page, blocked).locator(".mt-node-header").boundingBox())!;
    await page.mouse.move(hb.x + hb.width / 2, hb.y + hb.height / 2);
    await page.mouse.down();
    await page.mouse.move(hb.x + hb.width / 2 + 140 * zoom, hb.y + hb.height / 2, {
      steps: 8,
    });
    await page.mouse.up();
  }
}

/** Handles are .mt-port elements; the adjacent label span carries the same
 *  title attribute, and some nodes (e.g. Preview Plot) expose both an input
 *  and an output handle with the same title — disambiguate by position. */
async function connect(
  page: Page,
  from: { node: string; port: string },
  to: { node: string; port: string },
) {
  const source = node(page, from.node).locator(
    `.mt-port[title="${from.port}"][data-handlepos="right"]`,
  );
  const target = node(page, to.node).locator(
    `.mt-port[title="${to.port}"][data-handlepos="left"]`,
  );
  await ensureHandleHittable(page, from.node, from.port, "right");
  await ensureHandleHittable(page, to.node, to.port, "left");
  await source.dragTo(target);
}

/** Build the Load CSV -> Plot Time Series -> Preview Plot workflow. */
async function buildPipeline(page: Page) {
  await page.goto("/");

  // Fit the view so every node is visible and hittable (fitView may zoom the
  // viewport, which the placement math accounts for). Place nodes newest-first:
  // a node is only draggable if its header is not covered by an earlier node
  // (DOM order — later nodes render on top).
  await addNode(page, "Load CSV");
  await addNode(page, "Plot Time Series");
  await addNode(page, "Preview Plot");
  await page.getByRole("button", { name: "Fit View" }).click();
  await settleViewport(page);

  await placeNode(page, "Preview Plot", 1050, 150);
  await placeNode(page, "Plot Time Series", 600, 150);
  await placeNode(page, "Load CSV", 150, 150);

  // Refit so every node (and its handles) is inside the viewport — placement
  // happens at the zoom level set by the first fit, and nodes placed far
  // right may be clipped off-canvas.
  await page.getByRole("button", { name: "Fit View" }).click();
  await settleViewport(page);

  // The CSV path is server-side; fill the absolute path of the repo sample.
  await node(page, "Load CSV").locator(".path-field input").fill(CSV_PATH);
  await node(page, "Load CSV").locator(".path-field input").press("Enter");

  // Load CSV (data: ts.run) -> Plot Time Series (data: ts.run)
  await connect(page, { node: "Load CSV", port: "time series: ts.run" }, {
    node: "Plot Time Series",
    port: "time series: ts.run",
  });
  // Plot Time Series (plot: plot) -> Preview Plot (plot: plot)
  await connect(page, { node: "Plot Time Series", port: "plot: plot" }, {
    node: "Preview Plot",
    port: "plot: plot",
  });
}

test("runs a Load CSV -> Plot Time Series -> Preview pipeline end-to-end", async ({
  page,
}) => {
  await buildPipeline(page);

  await page.getByRole("button", { name: "▶ Run" }).click();
  await expect(page.locator(".run-status.completed")).toBeVisible({
    timeout: 90_000,
  });

  // The preview renders the published plot artifact from the backend.
  const preview = page.locator(".mt-preview img");
  await expect(preview).toBeVisible({ timeout: 30_000 });
  await expect(preview).toHaveAttribute("src", /\/api\/artifacts\//);

  // The viewer node exposes a download link for the full artifact.
  const downloadPromise = page.waitForEvent("download");
  await node(page, "Preview Plot").getByLabel("Download artifact").click();
  const download = await downloadPromise;
  expect(download.suggestedFilename()).toMatch(/\.png$/);
});

test("cancelling a running run stops the pipeline", async ({ page }) => {
  await buildPipeline(page);
  // Cold run: the cross-run cache (§7) would otherwise complete the whole
  // pipeline before the cancel request can land.
  await page.request.delete("/api/cache");

  await page.getByRole("button", { name: "▶ Run" }).click();
  // Cancel is available as soon as the run is queued; clicking immediately
  // lands the cancel request long before the pipeline (~2s) can complete.
  await page.getByRole("button", { name: "✕ Cancel" }).click();

  await expect(page.locator(".run-status.cancelled")).toBeVisible({
    timeout: 30_000,
  });
  // A cancelled run frees the Run button for another attempt.
  await expect(page.getByRole("button", { name: "▶ Run" })).toBeEnabled();
});

test("unchanged rerun reports cached nodes; Run to here executes the ancestry only", async ({
  page,
}) => {
  await buildPipeline(page);

  await page.getByRole("button", { name: "▶ Run" }).click();
  await expect(page.locator(".run-status.completed")).toBeVisible({
    timeout: 90_000,
  });
  await expect(page.locator(".mt-preview img")).toBeVisible({ timeout: 30_000 });

  // Nothing changed ⇒ the structural signatures all match ⇒ every node
  // reports "cached" on the second run (and the artifact is restored into
  // the new run's workspace, so the preview still renders).
  await page.getByRole("button", { name: "▶ Run" }).click();
  await expect(page.locator(".run-status.completed")).toBeVisible({
    timeout: 90_000,
  });
  await expect(page.locator(".mt-cached")).toHaveCount(3);
  await expect(page.locator(".mt-preview img")).toBeVisible({ timeout: 30_000 });

  // Select the middle node and run only it plus its ancestry: Load CSV and
  // Plot Time Series execute (cache hits here), Preview Plot is not in the
  // execution set — no chip for it. The restored plot artifact from Plot
  // Time Series still flows to the viewer via the edge, so the preview keeps
  // rendering the (still valid) figure.
  await node(page, "Plot Time Series").locator(".mt-node-header").click();
  await page.getByRole("button", { name: "▶ Run to here" }).click();
  await expect(page.locator(".run-status.completed")).toBeVisible({
    timeout: 90_000,
  });
  await expect(page.locator(".mt-cached")).toHaveCount(2);
  await expect(page.locator(".mt-preview img")).toBeVisible({ timeout: 30_000 });
});

test("time selector bar: drag updates boxes, box edit moves handle", async ({
  page,
}) => {
  await buildPipeline(page);
  await page.getByRole("button", { name: "▶ Run" }).click();
  await expect(page.locator(".run-status.completed")).toBeVisible({
    timeout: 90_000,
  });

  // The bar appears on Plot Time Series once the upstream streamed its
  // payload extent (§23.4). The sample CSV has no time column ⇒ the bar
  // renders relative seconds.
  const plotNode = node(page, "Plot Time Series");
  const bar = plotNode.locator(".time-bar");
  await expect(bar).toBeVisible({ timeout: 15_000 });
  const ticks = bar.locator(".time-bar-ticks span");
  const t0 = parseFloat((await ticks.nth(0).textContent())!);
  const t1 = parseFloat((await ticks.nth(1).textContent())!);
  expect(t0).toBe(0);
  expect(t1).toBeGreaterThan(1);

  const startBox = plotNode
    .locator(".param-field", { hasText: "Start (UTC)" })
    .locator("input");
  const endBox = plotNode
    .locator(".param-field", { hasText: "End (UTC)" })
    .locator("input");
  await expect(startBox).toHaveValue("");

  // Drag the start handle ~25% into the bar → the Start box updates.
  const handle = bar.locator(".time-bar-handle").nth(0);
  const track = bar.locator(".time-bar-track");
  const hb = (await handle.boundingBox())!;
  const tb = (await track.boundingBox())!;
  await page.mouse.move(hb.x + hb.width / 2, hb.y + hb.height / 2);
  await page.mouse.down();
  await page.mouse.move(tb.x + tb.width * 0.25, hb.y + hb.height / 2, {
    steps: 8,
  });
  await page.mouse.up();
  const dragged = await startBox.inputValue();
  expect(parseFloat(dragged)).toBeGreaterThan(t1 * 0.1);
  expect(parseFloat(dragged)).toBeLessThan(t1 * 0.4);

  // Editing the box moves the handle (bidirectional sync).
  await startBox.fill(String(Math.round(t1 / 2)));
  await startBox.press("Enter");
  const handlePct = (nth: number) =>
    bar.locator(".time-bar-handle").nth(nth).evaluate((el) => {
      const track = el.parentElement!;
      const tr = track.getBoundingClientRect();
      const r = el.getBoundingClientRect();
      return ((r.x + r.width / 2 - tr.x) / tr.width) * 100;
    });
  const leftPct = await handlePct(0);
  expect(leftPct).toBeGreaterThan(45);
  expect(leftPct).toBeLessThan(55);

  // Guards: handles cannot cross or leave the record — dragging the start
  // handle far past the end keeps a > 1 s slot at the record end.
  const endPctBefore = await handlePct(1);
  await page.mouse.move(tb.x + tb.width * 0.25, hb.y + hb.height / 2);
  await page.mouse.down();
  await page.mouse.move(tb.x + tb.width * 0.98, hb.y + hb.height / 2, {
    steps: 8,
  });
  await page.mouse.up();
  const endPctAfter = await handlePct(1);
  expect(endPctAfter).toBeLessThanOrEqual(endPctBefore + 1);
  await expect(bar).toBeVisible(); // never crossed → slot stays valid
});

test("Extract Run offers the collection's sample rates as select suggestions (§20.1)", async ({
  page,
}) => {
  const H5 = path.resolve(
    here, "..", "..", "..", "examples", "data", "phoenix-mtu-test",
    "smoke_2h_phoenix_mtu.mth5",
  );
  await page.goto("/");
  await addNode(page, "Load MTH5");
  await addNode(page, "Extract Run");
  await addNode(page, "Plot Time Series");
  await page.getByRole("button", { name: "Fit View" }).click();
  await settleViewport(page);
  await placeNode(page, "Plot Time Series", 1050, 150);
  await placeNode(page, "Extract Run", 600, 150);
  await placeNode(page, "Load MTH5", 150, 150);
  await page.getByRole("button", { name: "Fit View" }).click();
  await settleViewport(page);

  await node(page, "Load MTH5").locator(".path-field input").fill(H5);
  await node(page, "Load MTH5").locator(".path-field input").press("Enter");
  await connect(page, { node: "Load MTH5", port: "mth5: mth5" }, {
    node: "Extract Run",
    port: "mth5: mth5",
  });
  await connect(page, { node: "Extract Run", port: "time series: ts.run" }, {
    node: "Plot Time Series",
    port: "time series: ts.run",
  });

  await page.getByRole("button", { name: "▶ Run" }).click();
  await expect(page.locator(".run-status.completed")).toBeVisible({
    timeout: 120_000,
  });

  // The sample_rate param's datalist gains the collection's distinct rates
  // once the payload extent arrives — "auto" first, then the rates (§20.1).
  // NB: a <datalist> is display:none by spec — assert on its options.
  const datalist = node(page, "Plot Time Series").locator("datalist");
  await expect(datalist).toHaveCount(1, { timeout: 15_000 });
  const opts = await datalist.locator("option").evaluateAll(
    (os) => os.map((o) => o.getAttribute("value") ?? ""),
  );
  expect(opts[0]).toBe("auto");
  expect(opts).toContain("150"); // smoke archive tiers: 150 / 15 / 2400 Hz
});

test("time selector bar on Extract Run shows the archive run extent (GMT)", async ({
  page,
}) => {
  const H5 = path.resolve(
    here, "..", "..", "..", "examples", "data", "phoenix-mtu-test",
    "smoke_2h_phoenix_mtu.mth5",
  );
  await page.goto("/");
  await addNode(page, "Load MTH5");
  await addNode(page, "Extract Run");
  await page.getByRole("button", { name: "Fit View" }).click();
  await settleViewport(page);
  await placeNode(page, "Extract Run", 600, 150);
  await placeNode(page, "Load MTH5", 150, 150);
  await page.getByRole("button", { name: "Fit View" }).click();
  await settleViewport(page);

  await node(page, "Load MTH5").locator(".path-field input").fill(H5);
  await node(page, "Load MTH5").locator(".path-field input").press("Enter");
  await connect(page, { node: "Load MTH5", port: "mth5: mth5" }, {
    node: "Extract Run",
    port: "mth5: mth5",
  });

  await page.getByRole("button", { name: "▶ Run" }).click();
  await expect(page.locator(".run-status.completed")).toBeVisible({
    timeout: 120_000,
  });

  // Default 'time range' mode: the bar shows the UNION extent over every
  // run of the station (§23.4 — the collection's full coverage). Switch to
  // 'run id' mode so the bar shows one run's extent (the assertions below
  // are tuned to sr150_0001, the first run of the smoke archive's summary);
  // rerun so the payload event reflects the new selection.
  const selection = node(page, "Extract Run")
    .locator(".param-field", { hasText: "Selection" })
    .locator("select");
  await selection.selectOption("run id");
  await page.getByRole("button", { name: "▶ Run" }).click();
  await expect(page.locator(".run-status.completed")).toBeVisible({
    timeout: 120_000,
  });

  // The bar appears on Extract Run from the archive's per-run summary —
  // run sr150_0001 (first run) of the smoke archive, absolute GMT ticks.
  const bar = node(page, "Extract Run").locator(".time-bar");
  await expect(bar).toBeVisible({ timeout: 15_000 });
  const ticks = bar.locator(".time-bar-ticks span");
  await expect(ticks.nth(0)).toHaveText("2026-06-18 20:00:00");
  await expect(ticks.nth(1)).toHaveText("2026-06-18 20:00:15");

  // Select a slot by editing the boxes; rerun crops the extraction.
  const startBox = node(page, "Extract Run")
    .locator(".param-field", { hasText: "Start (UTC)" })
    .locator("input");
  await startBox.fill("2026-06-18 20:00:05");
  await startBox.press("Enter");
  const handlePct = await bar.locator(".time-bar-handle").nth(0).evaluate((el) => {
    const tr = el.parentElement!.getBoundingClientRect();
    const r = el.getBoundingClientRect();
    return ((r.x + r.width / 2 - tr.x) / tr.width) * 100;
  });
  // sr150_0001 is ~16 s long; 5 s in ≈ 31%
  expect(handlePct).toBeGreaterThan(20);
  expect(handlePct).toBeLessThan(40);

  await page.getByRole("button", { name: "▶ Run" }).click();
  await expect(page.locator(".run-status.completed")).toBeVisible({
    timeout: 120_000,
  });
});

test("a node error is shown as copyable text in the properties panel", async ({
  page,
}) => {
  // Deterministic node error: Extract Run with a nonexistent run id.
  const H5 = path.resolve(
    here, "..", "..", "..", "examples", "data", "phoenix-mtu-test",
    "smoke_2h_phoenix_mtu.mth5",
  );
  await page.goto("/");
  await addNode(page, "Load MTH5");
  await addNode(page, "Extract Run");
  await placeNode(page, "Extract Run", 600, 150);
  await placeNode(page, "Load MTH5", 150, 150);
  await page.getByRole("button", { name: "Fit View" }).click();
  await settleViewport(page);

  await node(page, "Load MTH5").locator(".path-field input").fill(H5);
  await node(page, "Load MTH5").locator(".path-field input").press("Enter");
  await connect(page, { node: "Load MTH5", port: "mth5: mth5" }, {
    node: "Extract Run",
    port: "mth5: mth5",
  });

  // Force a deterministic node error: switch to 'run id' mode and pick a
  // nonexistent run (the Run id param is enabled in that mode only).
  const selection = node(page, "Extract Run")
    .locator(".param-field", { hasText: "Selection" })
    .locator("select");
  await selection.selectOption("run id");
  const runBox = node(page, "Extract Run")
    .locator(".param-field", { hasText: "Run id" })
    .locator("input");
  await runBox.fill("no_such_run");
  await runBox.press("Enter");

  await page.getByRole("button", { name: "▶ Run" }).click();
  await expect(page.locator(".run-status.failed")).toBeVisible({
    timeout: 120_000,
  });

  // H3: the toolbar carries the REASON, not just the red status — the run
  // store keeps run_finished.error, so a failed run is never unexplained
  // (the element only mounts when there is a message to show)
  await expect(page.locator("header .error-msg")).toBeVisible();

  // The node card shows an ellipsized badge; the panel must show the full,
  // copyable text. Select the failing node.
  await node(page, "Extract Run").click();
  const block = page.locator(".properties .node-error");
  await expect(block).toBeVisible();
  const text = await block.locator(".node-error-text").textContent();
  expect(text).toContain("run");
  expect(text!.toLowerCase()).toContain("not");

  // Copy affordance: the button acknowledges (clipboard content itself is
  // not asserted — the ZeroTier http origin has no secure-context
  // clipboard; the execCommand fallback path is what runs there). Grant
  // the clipboard so the primary path is deterministic in headless.
  await page.context().grantPermissions(["clipboard-write", "clipboard-read"]);
  await block.locator("button", { hasText: "copy" }).click();
  await expect(block.locator("button")).toHaveText(/copied/);
});
