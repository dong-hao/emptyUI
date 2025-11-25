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

/** New nodes spawn at random positions and may overlap; spread them apart.
 *  Raw mouse drag (locator.dragTo requires the end point to stay over the
 *  target element — the canvas is narrower than the window). Node drags are
 *  relative: pointer travel / zoom = flow travel, so the pointer delta is
 *  computed from the current flow position and corrected iteratively. A
 *  previously dragged node stays selected (raised z-index) and would intercept
 *  the pointer, so deselect on the pane first. */
async function placeNode(page: Page, title: string, x: number, y: number) {
  const canvas = page.locator(".react-flow");
  const cb = (await canvas.boundingBox())!;
  // deselect: click empty pane area above the nodes
  await page.mouse.click(cb.x + cb.width / 2, cb.y + 30);
  for (let attempt = 0; attempt < 4; attempt++) {
    const pos = await flowPosition(page, title);
    const dx = x - pos.x;
    const dy = y - pos.y;
    if (Math.abs(dx) < 2 && Math.abs(dy) < 2) return;
    const zoom = await viewportZoom(page);
    const hb = (await node(page, title).locator(".mt-node-header").boundingBox())!;
    const sx = hb.x + hb.width / 2;
    const sy = hb.y + hb.height / 2;
    await page.mouse.move(sx, sy);
    await page.mouse.down();
    await page.mouse.move(sx + dx * zoom, sy + dy * zoom, { steps: 10 });
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
  await source.dragTo(target);
}

test("runs a Load CSV -> Plot Time Series -> Preview pipeline end-to-end", async ({
  page,
}) => {
  await page.goto("/");

  // Fit the view so every node is visible and hittable (fitView may zoom the
  // viewport, which the placement math accounts for). Place nodes newest-first:
  // a node is only draggable if its header is not covered by an earlier node
  // (DOM order — later nodes render on top).
  await addNode(page, "Load CSV");
  await addNode(page, "Plot Time Series");
  await addNode(page, "Preview Plot");
  await page.getByRole("button", { name: "Fit View" }).click();

  await placeNode(page, "Preview Plot", 750, 150);
  await placeNode(page, "Plot Time Series", 450, 150);
  await placeNode(page, "Load CSV", 150, 150);

  // Refit so every node (and its handles) is inside the viewport — placement
  // happens at the zoom level set by the first fit, and nodes placed far
  // right may be clipped off-canvas.
  await page.getByRole("button", { name: "Fit View" }).click();

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
