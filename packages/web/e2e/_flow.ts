/** Shared React-Flow interaction helpers for the e2e specs (copied from
 *  pipeline.spec.ts, which keeps its own inline copies — the specs that
 *  need several nodes/edges per test import from here instead). */

import { expect, type Locator, type Page } from "@playwright/test";

export function node(page: Page, title: string): Locator {
  return page.locator(".mt-node", { hasText: title }).first();
}

export async function addNode(page: Page, name: string) {
  await page.getByRole("button", { name, exact: true }).click();
}

/** Current viewport zoom (flow units are scaled by this on screen). */
export async function viewportZoom(page: Page): Promise<number> {
  return page.evaluate(() => {
    const vp = document.querySelector(".react-flow__viewport") as HTMLElement | null;
    const m = vp?.style.transform.match(/scale\(([-\d.]+)\)/);
    return m ? parseFloat(m[1]) : 1;
  });
}

/** React Flow's Fit View animates the viewport; wait until the transform
 *  stops changing so placement/zoom reads and handle drags are accurate. */
export async function settleViewport(page: Page) {
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

/** Node flow position from the wrapper's translate() (React Flow coords). */
export async function flowPosition(page: Page, title: string) {
  return page.evaluate((t) => {
    const nodes = Array.from(document.querySelectorAll(".react-flow__node"));
    const n = nodes.find(
      (n) => n.querySelector(".mt-node-header")?.textContent === t,
    ) as HTMLElement | undefined;
    const m = n?.style.transform.match(/translate\(([-\d.]+)px,\s*([-\d.]+)px\)/);
    return m ? { x: parseFloat(m[1]), y: parseFloat(m[2]) } : { x: 0, y: 0 };
  }, title);
}

/** Drag a node by a flow-space delta (pointer travel = delta * zoom). */
export async function dragNodeBy(page: Page, title: string, dx: number, dy: number) {
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
export async function ensureHeaderHittable(page: Page, title: string) {
  for (let attempt = 0; attempt < 3; attempt++) {
    const blocked = await page.evaluate((t) => {
      const nodes = Array.from(document.querySelectorAll(".react-flow__node"));
      const n = nodes.find(
        (n) => n.querySelector(".mt-node-header")?.textContent === t,
      );
      if (!n) return null;
      const header = n.querySelector(".mt-node-header") as HTMLElement | null;
      if (!header) return null;
      const r = header.getBoundingClientRect();
      const el = document.elementFromPoint(r.x + r.width / 2, r.y + r.height / 2);
      if (el && (header === el || header.contains(el) || el.contains(header))) {
        return null;
      }
      const block = el?.closest(".react-flow__node");
      return block?.querySelector(".mt-node-header")?.textContent ?? null;
    }, title);
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

/** Move a node to an absolute FLOW position (placement retries until the
 *  transform matches — fit view may animate underneath). */
export async function placeNode(page: Page, title: string, x: number, y: number) {
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

/** If another node covers the given handle, nudge the blocker aside (up to
 *  3 attempts) before a connect drag. */
export async function ensureHandleHittable(
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
export async function connect(
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
  await expect(source).toHaveCount(1);
  await expect(target).toHaveCount(1);
  await ensureHandleHittable(page, from.node, from.port, "right");
  await ensureHandleHittable(page, to.node, to.port, "left");
  await source.dragTo(target);
}
