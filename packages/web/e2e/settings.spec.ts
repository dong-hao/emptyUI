import { expect, test } from "@playwright/test";

/**
 * Plugin enable/disable settings dialog (DESIGN.md §5.1).
 *
 * Full-stack: the backend is the shared dev/CI instance and runtime toggles
 * are in-memory only, so the test first ensures a known state (example
 * enabled) and always leaves the plugin enabled for the rest of the suite.
 * The example plugin is installed in the dev/CI backend (README setup).
 */
test("settings dialog toggles third-party plugins live", async ({ page }) => {
  await page.goto("/");

  // Palette nodes provided by the third-party example plugin (installed in
  // the dev/CI backend) — used as the visible/not-visible probe.
  const rms = page.getByRole("button", { name: "RMS Meter", exact: true });

  await page.getByRole("button", { name: "⚙ Settings" }).click();
  const dialog = page.locator(".settings-dialog");
  await expect(dialog).toBeVisible();

  // Bundled plugins: read-only info rows, always on (§5.1).
  for (const name of ["core", "mth5", "aurora"]) {
    const row = dialog.locator("tr", { hasText: name }).first();
    await expect(row.locator("input[type=checkbox]")).toBeDisabled();
    await expect(row.locator("input[type=checkbox]")).toBeChecked();
  }

  // Third-party plugin: toggleable. Self-heal to a known enabled state.
  // The checkbox is controlled: the checked state follows the server
  // response, so click and wait (never use check()/uncheck() — their
  // immediate post-click check races the async toggle).
  const example = dialog.locator("tr", { hasText: "example" }).first();
  const exampleToggle = example.locator("input[type=checkbox]");
  await expect(exampleToggle).toBeEnabled();
  if (!(await exampleToggle.isChecked())) {
    await exampleToggle.click();
    await expect(exampleToggle).toBeChecked();
  }
  await expect(rms).toBeVisible();

  // Disable: the catalog refetches and the palette hides the plugin's nodes.
  await exampleToggle.click();
  await expect(exampleToggle).not.toBeChecked();
  await expect(rms).toBeHidden();
  // The row stays listed (re-enableable) and reports disabled.
  await expect(example).toContainText("disabled");

  // Enable: the nodes return.
  await exampleToggle.click();
  await expect(exampleToggle).toBeChecked();
  await expect(rms).toBeVisible();
  await expect(example).not.toContainText("disabled");

  await page
    .locator(".settings-dialog .modal-foot")
    .getByRole("button", { name: "Close" })
    .click();
  await expect(dialog).toBeHidden();
});

test("settings dialog: live knobs editable, deployment rows read-only", async ({
  page,
}) => {
  await page.goto("/");
  await page.getByRole("button", { name: "⚙ Settings" }).click();
  const dialog = page.locator(".settings-dialog");
  await expect(dialog).toBeVisible();

  // Server & retention: the cache-size knob is editable and applies live.
  // The value must MOVE relative to whatever the (possibly warm, reused)
  // server already holds — filling a value equal to the current one is not
  // dirty and the Save button stays disabled.
  const cacheRow = dialog.locator("tr", { hasText: "cache max entries" });
  const input = cacheRow.locator("input[type=number]");
  await expect(input).toBeEnabled();
  const current = await input.inputValue();
  await input.fill(String((parseInt(current, 10) || 0) + 1));
  await dialog.locator("button", { hasText: "Save" }).click();
  // after save the server view reflects the edit (input value + ui source)
  await expect(cacheRow.locator("input")).toHaveValue(
    String((parseInt(current, 10) || 0) + 1),
  );
  await expect(cacheRow).toContainText("ui");

  // Deployment rows are read-only code values (no inputs)
  const deployRow = dialog.locator("tr", { hasText: "cors origins" });
  await expect(deployRow.locator("code")).toContainText("*");
  await expect(deployRow.locator("input")).toHaveCount(0);

  // Planned (history §21) rows are disabled with a planned tag
  const plannedRow = dialog.locator("tr", { hasText: "project dir" });
  await expect(plannedRow.locator("input")).toBeDisabled();
  await expect(plannedRow).toContainText("planned");
});
