/** Chain-walking inheritance e2e (docs/DESIGN.md §4): identities (survey/station/
 *  run) flow through the upstream chain — auto-filled into empty params and
 *  rendered as editable combobox suggestions — across pass-through nodes
 *  (filters with no inherit declaration walk every input port with the same
 *  name). Fan-in unions are not exercisable through real wiring (validation
 *  rejects duplicate edges into one port); the walk's union is implicitly
 *  covered by the multi-port pass-through seek here. */

import { expect, test } from "@playwright/test";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { addNode, connect, node, placeNode, settleViewport } from "./_flow";

const here = path.dirname(fileURLToPath(import.meta.url));

test("identities auto-fill and render as comboboxes through a pass-through filter", async ({
  page,
}) => {
  // Wider viewport: the 4-node row must clear the minimap (bottom-right)
  // after fitView — a drop point under the minimap never lands.
  await page.setViewportSize({ width: 1920, height: 1080 });
  // Load MTH5 -> Extract Run -> Bandpass (pass-through) -> Compute FC:
  // Extract Run's survey/station/run are set MANUALLY (Load MTH5 has no
  // survey_id/station_id params to inherit); Compute FC sits BEHIND the
  // pass-through Bandpass, so its inherit walk must transitively seek the
  // same-named params on Extract Run.
  const H5 = path.resolve(
    here, "..", "..", "..", "examples", "data", "phoenix-mtu-test",
    "smoke_phoenix_mtu.mth5",
  );
  await page.goto("/");
  await addNode(page, "Load MTH5");
  await addNode(page, "Extract Run");
  await addNode(page, "Bandpass Filter");
  await addNode(page, "Compute FC");
  await page.getByRole("button", { name: "Fit View" }).click();
  await settleViewport(page);
  await placeNode(page, "Compute FC", 1200, 150);
  await placeNode(page, "Bandpass Filter", 850, 150);
  await placeNode(page, "Extract Run", 500, 150);
  await placeNode(page, "Load MTH5", 150, 150);
  await page.getByRole("button", { name: "Fit View" }).click();
  await settleViewport(page);

  await node(page, "Load MTH5").locator(".path-field input").fill(H5);
  await node(page, "Load MTH5").locator(".path-field input").press("Enter");

  // Set the identities on Extract Run manually (Load MTH5 carries no
  // survey_id/station_id params to inherit — the Run id param is enabled
  // in 'run id' mode only).
  const er = node(page, "Extract Run");
  const surveyIn = er.locator(".param-field", { hasText: "Survey id" }).locator("input");
  await surveyIn.fill("survey");
  await surveyIn.press("Enter");
  const stationIn = er.locator(".param-field", { hasText: "Station id" }).locator("input");
  await stationIn.fill("B0940330");
  await stationIn.press("Enter");
  const selIn = er.locator(".param-field", { hasText: "Selection" }).locator("select");
  await selIn.selectOption("run id");
  const runIn = er.locator(".param-field", { hasText: "Run id" }).locator("input");
  await runIn.fill("sr150_0001");
  await runIn.press("Enter");

  await connect(page, { node: "Load MTH5", port: "mth5: mth5" }, {
    node: "Extract Run",
    port: "mth5: mth5",
  });
  await connect(page, { node: "Extract Run", port: "time series: ts.run" }, {
    node: "Bandpass Filter",
    port: "time series: ts.run",
  });
  await connect(page, { node: "Bandpass Filter", port: "filtered time series: ts.run" }, {
    node: "Compute FC",
    port: "time series: ts.run",
  });

  // Auto-fill: Compute FC's identity params filled from Extract Run through
  // the pass-through filter — WITHOUT the user typing anything.
  const fc = node(page, "Compute FC");
  const surveyBox = fc.locator(".param-field", { hasText: "Survey id" }).locator("input");
  const stationBox = fc.locator(".param-field", { hasText: "Station id" }).locator("input");
  const runBox = fc.locator(".param-field", { hasText: "Run id" }).locator("input");
  await expect(surveyBox).toHaveValue("survey", { timeout: 15_000 });
  await expect(stationBox).toHaveValue("B0940330");
  await expect(runBox).toHaveValue("sr150_0001");

  // Combobox: the inherited params render as editable comboboxes whose
  // datalist suggestions carry the upstream values (the <datalist> is
  // display:none — assert its options, not visibility).
  for (const [box, value] of [
    [surveyBox, "survey"], [stationBox, "B0940330"], [runBox, "sr150_0001"],
  ] as const) {
    const listId = await box.getAttribute("list");
    expect(listId, "inherited param renders as a combobox").toBeTruthy();
    const opts = await fc.locator(`datalist[id="${listId}"] option`)
      .evaluateAll((els) => els.map((e) => (e as HTMLOptionElement).value));
    expect(opts).toContain(value);
  }
});
