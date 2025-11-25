import { expect, test, type Locator, type Page } from "@playwright/test";

/** Palette buttons are grouped by category; click by exact visible name. */
async function addNode(page: Page, name: string) {
  await page.getByRole("button", { name, exact: true }).click();
}

/** Params render inline: label span + control inside .param-field (no <label>). */
function param(node: Locator, label: string): Locator {
  return node.locator(".param-field", { hasText: label });
}

test("app boots, catalog loads, and the palette renders all node categories", async ({
  page,
}) => {
  await page.goto("/");
  await expect(page).toHaveTitle("emptyUI");

  // Palette is populated from the live backend catalog (GET /api/nodes).
  await expect(page.getByRole("heading", { name: "Processing" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Data I/O" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Display" })).toBeVisible();
  for (const name of [
    "Compute FC",
    "Estimate TF",
    "Load LEMI-424",
    "Extract Run",
    "Plot AppRes & Phase",
    "View TF",
    "Preview Plot",
  ]) {
    await expect(page.getByRole("button", { name, exact: true })).toBeVisible();
  }
  // Run is disabled until the graph has nodes.
  await expect(page.getByRole("button", { name: "▶ Run" })).toBeDisabled();
});

test("Compute FC renders the channel nomenclature select and cascades channel names", async ({
  page,
}) => {
  await page.goto("/");
  await addNode(page, "Compute FC");

  const node = page.locator(".mt-node", { hasText: "Compute FC" }).first();

  // New nomenclature UI: keyword select (mt_metadata CHANNEL_MAPS keywords)
  // instead of the old manual magnetic-unit radio.
  const nomenclature = param(node, "Channel nomenclature").locator("select");
  await expect(nomenclature).toBeVisible();
  await expect(nomenclature.locator("option")).toHaveText([
    "auto",
    "default",
    "lemi12",
    "lemi34",
    "phoenix123",
    "musgraves",
    "nims",
  ]);
  await expect(nomenclature).toHaveValue("auto");

  // Magnetic input is derived automatically (auto) — no manual B/H switch.
  const magnetic = param(node, "Magnetic input").locator("select");
  await expect(magnetic).toHaveValue("auto");
  await expect(magnetic.locator("option")).toHaveText(["auto", "h", "b"]);

  // Cascade: selecting lemi12 swaps the channel names to the LEMI convention.
  await nomenclature.selectOption("lemi12");
  await expect(param(node, "Input channels").locator("input")).toHaveValue("bx,by");
  await expect(param(node, "Output channels").locator("input")).toHaveValue("e1,e2");
});
