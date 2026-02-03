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

  // Cascade: selecting lemi12 swaps the channel names to the LEMI convention
  // — the vertical magnetic channel (bz) is included so the tipper is on by
  // default (dropped automatically when the run has no vertical channel).
  await nomenclature.selectOption("lemi12");
  await expect(param(node, "Input channels").locator("input")).toHaveValue("bx,by");
  await expect(param(node, "Output channels").locator("input")).toHaveValue("e1,e2,bz");
});

test("Estimate TF's remote reference is enabled only for the robust estimator", async ({
  page,
}) => {
  await page.goto("/");
  await addNode(page, "Estimate TF");

  const node = page.locator(".mt-node", { hasText: "Estimate TF" }).first();
  const engine = param(node, "Estimator").locator("select");
  const remote = param(node, "Remote Reference").locator("input[type=checkbox]");

  // default Robust -> editable; manifest-driven rule (enabled_when)
  await expect(engine).toHaveValue("Robust");
  await expect(remote).toBeEnabled();

  await engine.selectOption("Least Squares");
  await expect(remote).toBeDisabled();
  await engine.selectOption("Robust");
  await expect(remote).toBeEnabled();
});

test("Coherence Sort is band-free and Estimate TF keeps its own banding (§24.9)", async ({
  page,
}) => {
  await page.goto("/");
  await addNode(page, "Coherence Sort");
  await addNode(page, "Estimate TF");

  const sort = page.locator(".mt-node", { hasText: "Coherence Sort" }).first();
  const estimate = page.locator(".mt-node", { hasText: "Estimate TF" }).first();

  // separate the randomly-placed nodes so both handles are hittable
  const eb = (await estimate.locator(".mt-node-header").boundingBox())!;
  await page.mouse.move(eb.x + eb.width / 2, eb.y + eb.height / 2);
  await page.mouse.down();
  await page.mouse.move(eb.x + eb.width / 2 + 420, eb.y + eb.height / 2, {
    steps: 8,
  });
  await page.mouse.up();

  // connect Coherence Sort's fc output -> Estimate TF's fc input (raw mouse
  // drag between handle centers — locator.dragTo undershoots on the canvas)
  const sh = (await sort.locator('.mt-port[data-handlepos="right"]').first().boundingBox())!;
  const th = (await estimate.locator('.mt-port[data-handlepos="left"]').first().boundingBox())!;
  await page.mouse.move(sh.x + sh.width / 2, sh.y + sh.height / 2);
  await page.mouse.down();
  await page.mouse.move(th.x + th.width / 2, th.y + th.height / 2, { steps: 8 });
  await page.mouse.up();

  // §24.9: the sort is band-free — no banding widget on its surface; the
  // estimator owns banding and keeps its own defaults (the old sort→TF
  // banding inherit was deliberately removed and must stay removed)
  for (const label of ["Band style", "Periods per octave", "Min bins per band"]) {
    await expect(param(sort, label)).toHaveCount(0);
  }
  await expect(param(sort, "Apply below period")).toHaveCount(1);
  await expect(param(estimate, "Band style").locator("select")).toHaveValue(
    "l2u",
  );
});
