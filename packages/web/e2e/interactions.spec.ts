import { expect, test, type Page } from "@playwright/test";

/** Palette buttons are grouped by category; click by exact visible name. */
async function addNode(page: Page, name: string) {
  await page.getByRole("button", { name, exact: true }).click();
}

/** Drag from one port handle to another. */
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
  const fb = await from.boundingBox();
  const tb = await to.boundingBox();
  await page.mouse.move(fb.x + fb.width / 2, fb.y + fb.height / 2);
  await page.mouse.down();
  await page.mouse.move(tb.x + tb.width / 2, tb.y + tb.height / 2, { steps: 8 });
  await page.mouse.up();
  await page.waitForTimeout(200);
}

test("self-connections are rejected live during the drag", async ({ page }) => {
  await page.goto("/");
  await addNode(page, "Coherence Sort");

  // fc output -> fc input of the SAME node must not create an edge
  await dragPortToPort(page, 0, "right", 0, "left");
  await expect(page.locator(".react-flow__edge")).toHaveCount(0);
});

test("cross-node cycles are rejected, valid chains accepted", async ({ page }) => {
  await page.goto("/");
  await addNode(page, "Coherence Sort");
  await addNode(page, "Coherence Sort");

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
