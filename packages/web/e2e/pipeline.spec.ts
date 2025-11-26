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
    console.log(`HEADER-BLOCK: ${title} covered by "${blocked}"`);
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
    console.log(
      `HIT-BLOCK: ${handlepos} handle ${title}.${port} covered by "${blocked}"`,
      await page.evaluate(() =>
        Array.from(document.querySelectorAll(".react-flow__node")).map((n) => ({
          t: n.querySelector(".mt-node-header")?.textContent,
          x: n.style.transform,
        })),
      ),
    );
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

  await page.getByRole("button", { name: "▶ Run" }).click();
  await expect(page.locator(".run-status.completed")).toBeVisible({
    timeout: 90_000,
  });

  // The preview renders the published plot artifact from the backend.
  const preview = page.locator(".mt-preview img");
  await expect(preview).toBeVisible({ timeout: 30_000 });
  await expect(preview).toHaveAttribute("src", /\/api\/artifacts\//);
});
