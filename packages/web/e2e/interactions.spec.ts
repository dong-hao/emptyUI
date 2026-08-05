import { expect, test, type Page } from "@playwright/test";

import { addNode, must, node, placeNode, settleViewport } from "./_flow";

/** Deterministic geometry: room for three spread-out nodes on the canvas. */
test.use({ viewport: { width: 1600, height: 1000 } });

/** Palette buttons are grouped by category; click by exact visible name. */


/** Let React Flow's fitView settle before measuring element boxes. */
async function settle(page: Page) {
  await page.waitForTimeout(700);
}

/** Drag a node (by its header) so its top-left lands at (x, y). */
async function moveNodeTo(page: Page, idx: number, x: number, y: number) {
  const h = page.locator(".mt-node").nth(idx).locator(".mt-node-header");
  const b = must(await h.boundingBox(), `node ${idx + 1}'s header`);
  await page.mouse.move(b.x + b.width / 2, b.y + b.height / 2);
  await page.mouse.down();
  await page.mouse.move(x + b.width / 2, y + b.height / 2, { steps: 5 });
  await page.mouse.up();
  await page.waitForTimeout(150);
}

/** Drag from one port handle to another (boxes measured fresh). */
async function dragPortToPort(
  page: Page,
  fromNode: number,
  fromPos: "left" | "right",
  toNode: number,
  toPos: "left" | "right",
) {
  const from = page
    .locator(".mt-node")
    .nth(fromNode)
    .locator(`.mt-port[data-handlepos="${fromPos}"]`)
    .first();
  const to = page
    .locator(".mt-node")
    .nth(toNode)
    .locator(`.mt-port[data-handlepos="${toPos}"]`)
    .first();
  const fb = must(await from.boundingBox(), "the source port handle");
  const tb = must(await to.boundingBox(), "the target port handle");
  await page.mouse.move(fb.x + fb.width / 2, fb.y + fb.height / 2);
  await page.mouse.down();
  await page.mouse.move(tb.x + tb.width / 2, tb.y + tb.height / 2, { steps: 8 });
  await page.mouse.up();
  await page.waitForTimeout(200);
}

/** Add n Coherence Sort nodes at deterministic, non-overlapping positions
 *  (palette placement is random and nodes occlude each other's handles). */
async function addSpreadNodes(page: Page, n: number) {
  for (let i = 0; i < n; i++) await addNode(page, "Coherence Sort");
  await settle(page);
  const spots = [
    [120, 60],
    [400, 300],
    [660, 540],
  ];
  for (let i = 0; i < n; i++) await moveNodeTo(page, i, spots[i][0], spots[i][1]);
  await settle(page);
}

test("self-connections are rejected live during the drag", async ({ page }) => {
  await page.goto("/");
  await addSpreadNodes(page, 1);

  // fc output -> fc input of the SAME node must not create an edge
  await dragPortToPort(page, 0, "right", 0, "left");
  await expect(page.locator(".react-flow__edge")).toHaveCount(0);
});

test("cross-node cycles are rejected, valid chains accepted", async ({ page }) => {
  await page.goto("/");
  await addSpreadNodes(page, 2);

  // valid: node1 fc -> node2 fc
  await dragPortToPort(page, 0, "right", 1, "left");
  await expect(page.locator(".react-flow__edge")).toHaveCount(1);

  // cycle: node2 fc -> node1 fc would close node1 -> node2 -> node1
  await dragPortToPort(page, 1, "right", 0, "left");
  await expect(page.locator(".react-flow__edge")).toHaveCount(1);

  // edge into the already-connected input: n2 -> n1 would close the cycle
  // (n1 -> n2 exists), so it is rejected
  await dragPortToPort(page, 1, "right", 0, "left");
  await expect(page.locator(".react-flow__edge")).toHaveCount(1);
});

test("drag a connected handle to the pane unlinks it (ComfyUI gesture)", async ({ page }) => {
  await page.goto("/");
  await addSpreadNodes(page, 2);
  await dragPortToPort(page, 0, "right", 1, "left");
  await expect(page.locator(".react-flow__edge")).toHaveCount(1);

  // drag node2's connected input handle to the empty pane -> unlink
  const inp = must(await page
    .locator(".mt-node")
    .nth(1)
    .locator('.mt-port[data-handlepos="left"]')
    .first()
    .boundingBox(), "node 2's connected input handle");
  const pane = must(await page.locator(".react-flow__pane").boundingBox(), "the React Flow pane");
  await page.mouse.move(inp.x + inp.width / 2, inp.y + inp.height / 2);
  await page.mouse.down();
  await page.mouse.move(pane.x + pane.width - 60, pane.y + pane.height - 40, { steps: 8 });
  await page.mouse.up();
  await expect(page.locator(".react-flow__edge")).toHaveCount(0);
});

test("a plain click on a connected handle does NOT unlink (hair-trigger guard)", async ({ page }) => {
  // L-F12: the unlink fired on down+up with no pointer movement — a stray
  // click on a connected input silently detached the edge. ComfyUI parity:
  // the link must visibly follow the cursor before it detaches.
  await page.goto("/");
  await addSpreadNodes(page, 2);
  await dragPortToPort(page, 0, "right", 1, "left");
  await expect(page.locator(".react-flow__edge")).toHaveCount(1);

  const inp = must(await page
    .locator(".mt-node")
    .nth(1)
    .locator('.mt-port[data-handlepos="left"]')
    .first()
    .boundingBox(), "node 2's connected input handle");
  // down + up at the same spot (a click, not a drag)
  await page.mouse.move(inp.x + inp.width / 2, inp.y + inp.height / 2);
  await page.mouse.down();
  await page.mouse.up();
  await expect(page.locator(".react-flow__edge")).toHaveCount(1);
});

test("a plain click on an edge endpoint does NOT delete the edge (hair-trigger guard)", async ({ page }) => {
  // The edge-endpoint reconnect gesture had the same trigger class: a
  // click on the endpoint anchor (no movement) deleted the edge. RF renders
  // the anchors as transparent circles (.react-flow__edgeupdater-*); node
  // content can cover them, so the synthetic mousedown dispatch from the
  // endpoint-drag test is reused here.
  await page.goto("/");
  await addSpreadNodes(page, 2);
  await dragPortToPort(page, 0, "right", 1, "left");
  await expect(page.locator(".react-flow__edge")).toHaveCount(1);

  const pt = await page.evaluate(() => {
    const c = document.querySelector(".react-flow__edgeupdater-target");
    if (!c) throw new Error("no .react-flow__edgeupdater-target to grab");
    const r = c.getBoundingClientRect();
    const x = r.x + r.width / 2;
    const y = r.y + r.height / 2;
    c.dispatchEvent(
      new MouseEvent("mousedown", {
        bubbles: true,
        cancelable: true,
        button: 0,
        clientX: x,
        clientY: y,
      }),
    );
    return { x, y };
  });
  // up at the SAME spot (a click, not a drag)
  await page.mouse.move(pt.x, pt.y);
  await page.mouse.up();
  await expect(page.locator(".react-flow__edge")).toHaveCount(1);
});

test("drag an edge endpoint: drop on the pane deletes, drop on a handle re-routes", async ({ page }) => {
  await page.goto("/");
  await addSpreadNodes(page, 3);
  await dragPortToPort(page, 0, "right", 1, "left");
  await expect(page.locator(".react-flow__edge")).toHaveCount(1);

  // hover an exact point ON the path (the <g> bbox center is unreliable for
  // bezier edges) to reveal the endpoint anchors
  const hoverPath = async () => {
    const pt = await page.evaluate(() => {
      const p = document.querySelector<SVGGeometryElement>(".react-flow__edge-path");
      if (!p) throw new Error("no .react-flow__edge-path to hover");
      const q = p.getPointAtLength(p.getTotalLength() * 0.3);
      const ctm = p.getScreenCTM();
      if (!ctm) throw new Error("edge path has no screen transform");
      const s = new DOMPoint(q.x, q.y).matrixTransform(ctm);
      return { x: s.x, y: s.y };
    });
    await page.mouse.move(pt.x, pt.y, { steps: 3 });
    await page.waitForTimeout(250);
  };

  // Start the reconnect drag by dispatching the pointer-down DIRECTLY on the
  // anchor element. The anchor circle sits on the input handle, which is
  // INSIDE the node body (DESIGN §3 port rendering), and RF renders edges
  // below nodes — so node content (inline params) can cover the anchor and a
  // real mousedown on its center would hit the node instead. A bubbled
  // synthetic mousedown on the circle always reaches React's handler; the
  // subsequent real mouse move/up drive the drag (XYHandle listens on
  // document, no pointer capture).
  const grabTargetAnchor = async () => {
    const pt = await page.evaluate(() => {
      const c = document.querySelector(".react-flow__edgeupdater-target");
      if (!c) throw new Error("no .react-flow__edgeupdater-target to grab");
      const r = c.getBoundingClientRect();
      const x = r.x + r.width / 2;
      const y = r.y + r.height / 2;
      c.dispatchEvent(
        new MouseEvent("mousedown", {
          bubbles: true,
          cancelable: true,
          button: 0,
          clientX: x,
          clientY: y,
        }),
      );
      return { x, y };
    });
    await page.mouse.move(pt.x, pt.y);
  };

  // drag the target-end anchor onto node3's input handle -> re-route.
  // The gesture can occasionally miss (a failed attempt detaches the edge),
  // so retry with a freshly re-created edge instead of flaking the suite.
  const ta = page.locator(".react-flow__edgeupdater-target").first();
  const dst = must(
    await page
      .locator(".mt-node")
      .nth(2)
      .locator('.mt-port[data-handlepos="left"]')
      .first()
      .boundingBox(),
    "node 3's left input handle",
  );
  for (let attempt = 0; attempt < 3; attempt++) {
    if ((await page.locator(".react-flow__edge").count()) === 0) {
      await dragPortToPort(page, 0, "right", 1, "left");
      await expect(page.locator(".react-flow__edge")).toHaveCount(1);
    }
    await hoverPath();
    await expect(ta).toBeVisible();
    await grabTargetAnchor();
    await page.mouse.move(dst.x + dst.width / 2, dst.y + dst.height / 2, { steps: 8 });
    await page.mouse.up();
    await page.waitForTimeout(250);
    if ((await page.locator(".react-flow__edge").count()) === 1) break;
  }
  await expect(page.locator(".react-flow__edge")).toHaveCount(1);

  // fresh hover, then drag the target-end anchor onto the empty pane -> delete
  // (retry: a failed grab leaves the edge untouched)
  const pane = must(await page.locator(".react-flow__pane").boundingBox(), "the React Flow pane");
  for (let attempt = 0; attempt < 3; attempt++) {
    await hoverPath();
    await expect(ta).toBeVisible();
    await grabTargetAnchor();
    // drop on a known-empty corner (bottom-right) instead of the pane
    // center — the center can land on a node's body/handles depending
    // on fitView
    await page.mouse.move(pane.x + pane.width - 60, pane.y + pane.height - 40, { steps: 8 });
    await page.mouse.up();
    await page.waitForTimeout(300);
    if ((await page.locator(".react-flow__edge").count()) === 0) break;
  }
  await expect(page.locator(".react-flow__edge")).toHaveCount(0);
});

test("click an edge and press Delete removes it", async ({ page }) => {
  await page.goto("/");
  await addSpreadNodes(page, 2);
  await dragPortToPort(page, 0, "right", 1, "left");
  await expect(page.locator(".react-flow__edge")).toHaveCount(1);

  const eb = must(await page.locator(".react-flow__edge").first().boundingBox(), "the edge");
  await page.mouse.click(eb.x + eb.width / 2, eb.y + eb.height / 2);
  await page.keyboard.press("Delete");
  await expect(page.locator(".react-flow__edge")).toHaveCount(0);
});

function paramField(node: ReturnType<Page["locator"]>, label: string) {
  return node.locator(".param-field", { hasText: label });
}

test("copy-paste duplicates the selected subgraph with fresh ids and params", async ({ page }) => {
  await page.goto("/");
  await addSpreadNodes(page, 2);
  await dragPortToPort(page, 0, "right", 1, "left");
  await expect(page.locator(".react-flow__edge")).toHaveCount(1);

  // a non-default param proves params are deep-copied (not defaulted)
  await paramField(page.locator(".mt-node").nth(0), "Coherence min")
    .locator("input")
    .fill("0.5");
  await paramField(page.locator(".mt-node").nth(0), "Coherence min")
    .locator("input")
    .press("Enter");

  // box-select both nodes deterministically: Shift + drag a selection
  // rectangle spanning the whole canvas (RF's selectionKeyCode is Shift;
  // selectionMode=Full requires nodes fully inside the box)
  const pane = must(await page.locator(".react-flow__pane").boundingBox(), "the React Flow pane");
  await page.keyboard.down("Shift");
  await page.mouse.move(pane.x + 2, pane.y + 2);
  await page.mouse.down();
  await page.mouse.move(pane.x + pane.width - 2, pane.y + pane.height - 2, { steps: 5 });
  await page.mouse.up();
  await page.keyboard.up("Shift");
  await page.waitForTimeout(200);
  await expect(page.locator(".mt-node.selected")).toHaveCount(2);

  await page.keyboard.press("ControlOrMeta+c");
  await page.keyboard.press("ControlOrMeta+v");
  await page.waitForTimeout(300);

  // the full subgraph is duplicated: 4 nodes, 2 edges (both internal —
  // nothing connects the copy to the original)
  await expect(page.locator(".mt-node")).toHaveCount(4);
  await expect(page.locator(".react-flow__edge")).toHaveCount(2);
  // params were deep-copied: two nodes show 0.5 now
  await expect(page.locator('.param-field input[value="0.5"]')).toHaveCount(2);

  // the pasted subgraph is the selection → Delete removes it (the ORIGINAL
  // nodes + edge remain)
  await page.keyboard.press("Delete");
  await expect(page.locator(".mt-node")).toHaveCount(2);
  await expect(page.locator(".react-flow__edge")).toHaveCount(1);
});

test("Ctrl+D duplicates a node and Delete-key removes the duplicate", async ({ page }) => {
  await page.goto("/");
  await addSpreadNodes(page, 1);
  await page.locator(".mt-node").nth(0).locator(".mt-node-header").click();

  await page.keyboard.press("ControlOrMeta+d");
  await expect(page.locator(".mt-node")).toHaveCount(2);
  await expect(page.locator(".react-flow__edge")).toHaveCount(0);

  // the duplicate is selected → Delete removes only it
  await page.keyboard.press("Delete");
  await expect(page.locator(".mt-node")).toHaveCount(1);
});

test("New asks for confirmation and clears the canvas on accept", async ({
  page,
}) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Load CSV", exact: true }).click();
  await expect(page.locator(".mt-node", { hasText: "Load CSV" })).toHaveCount(1);

  // Dismissing the confirm keeps the workflow.
  let keepIt = false;
  page.once("dialog", (d) => {
    keepIt = true;
    void d.dismiss();
  });
  await page.getByRole("button", { name: "New" }).click();
  expect(keepIt, "the confirm dialog fired").toBe(true);
  await expect(page.locator(".mt-node", { hasText: "Load CSV" })).toHaveCount(1);

  // Accepting clears the canvas (and the doc name resets).
  page.once("dialog", (d) => void d.accept());
  await page.getByRole("button", { name: "New" }).click();
  await expect(page.locator(".mt-node")).toHaveCount(0);
});

test("Delete selected removes exactly the highlighted nodes", async ({ page }) => {
  // H4: the button used to delete React Flow's selection UNIONED with the
  // properties panel's stale selectedNodeId — box-select B after clicking A
  // removed A *and* B, and the app has no undo anywhere.
  await page.goto("/");
  await addNode(page, "Load CSV");
  await addNode(page, "Coherence Sort");
  await settle(page);
  await placeNode(page, "Load CSV", 200, 120);
  await placeNode(page, "Coherence Sort", 900, 500);
  await page.getByRole("button", { name: "Fit View" }).click();
  await settleViewport(page);

  // click the first node: React Flow selects it and the panel focuses it
  await node(page, "Load CSV").locator(".mt-node-header").click();
  await settle(page);
  await expect(page.locator(".properties h2")).toHaveText("Load CSV");

  // Shift-box-select ONLY the second node (selectionMode=Full: it must sit
  // fully inside the rectangle, which must not reach the first node)
  const target = (await node(page, "Coherence Sort").boundingBox())!;
  await page.keyboard.down("Shift");
  await page.mouse.move(target.x - 12, target.y - 12);
  await page.mouse.down();
  await page.mouse.move(target.x + target.width + 12, target.y + target.height + 12, {
    steps: 8,
  });
  await page.mouse.up();
  await page.keyboard.up("Shift");
  await settle(page);

  // one highlighted node; the panel keeps ITS focus (Load CSV) — the
  // highlight is what "selected" means, not the panel's display state
  await expect(page.locator(".mt-node.selected")).toHaveCount(1);
  await expect(page.locator(".properties h2")).toHaveText("Load CSV");

  await page.getByRole("button", { name: "Delete selected" }).click();
  await settle(page);

  // only the highlighted node went — the clicked-but-deselected one survives
  await expect(node(page, "Load CSV")).toHaveCount(1);
  await expect(node(page, "Coherence Sort")).toHaveCount(0);
});
