/** Artifact display e2e: the View Table node fetches its input table's
 *  JSON artifact and renders it inline; the download link points at the
 *  artifact URL (the browser handles the fetch — no server round trip is
 *  asserted here). */

import { expect, test } from "@playwright/test";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { addNode, connect, node, placeNode, settleViewport } from "./_flow";

const here = path.dirname(fileURLToPath(import.meta.url));

test("View Table renders the incoming table artifact inline with a download link", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1920, height: 1080 });
  const H5 = path.resolve(
    here, "..", "..", "..", "examples", "data", "lims-usarray-test", "4P_SDG31.h5",
  );
  await page.goto("/");
  await addNode(page, "Load MTH5");
  await addNode(page, "Extract Run");
  await addNode(page, "Calibrate");
  await addNode(page, "Compute FC");
  await addNode(page, "Coherence Sort");
  await addNode(page, "View Table");
  await page.getByRole("button", { name: "Fit View" }).click();
  await settleViewport(page);
  for (const [title, x] of [
    ["View Table", 1650], ["Coherence Sort", 1350], ["Compute FC", 1050],
    ["Calibrate", 850], ["Extract Run", 500], ["Load MTH5", 150],
  ] as const) {
    await placeNode(page, title, x, 150);
  }
  await page.getByRole("button", { name: "Fit View" }).click();
  await settleViewport(page);

  await node(page, "Load MTH5").locator(".path-field input").fill(H5);
  await node(page, "Load MTH5").locator(".path-field input").press("Enter");
  await connect(page, { node: "Load MTH5", port: "mth5: mth5" }, {
    node: "Extract Run", port: "mth5: mth5",
  });
  await connect(page, { node: "Extract Run", port: "time series: ts.run" }, {
    node: "Calibrate", port: "data: ts.run",
  });
  await connect(page, { node: "Calibrate", port: "calibrated: ts.run" }, {
    node: "Compute FC", port: "time series: ts.run",
  });
  await connect(page, { node: "Compute FC", port: "fc: fc" }, {
    node: "Coherence Sort", port: "fc: fc",
  });
  await connect(page, { node: "Coherence Sort", port: "coherence diagnostics: table" }, {
    node: "View Table", port: "table: table",
  });

  await page.getByRole("button", { name: "▶ Run" }).click();
  await expect(page.locator(".run-status.completed")).toBeVisible({
    timeout: 120_000,
  });

  // The inline table renders the artifact's rows (one per frequency bin per
  // decimation level — non-empty regardless of QC outcomes).
  const body = node(page, "View Table").locator(".mt-table");
  await expect(body).toBeVisible({ timeout: 15_000 });
  const text = await body.textContent();
  expect(text).toContain("period");

  // The download link targets the artifact URL.
  const href = await body.locator("a.mt-download").getAttribute("href");
  expect(href).toContain("/api/artifacts/");
});
