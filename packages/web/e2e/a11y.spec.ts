/** Accessibility assertions (docs/issues.md L1): named param controls, live
 *  run-status regions, and dialogs that trap focus and name their close
 *  control. */
import { expect, test } from "@playwright/test";
import { addNode, node } from "./_flow";

test("param controls are labeled by their visible label (L1)", async ({
  page,
}) => {
  await page.goto("/");
  await addNode(page, "Load CSV");
  await addNode(page, "Compute FC");
  // the visible label resolves its control via htmlFor — previously only
  // the default widget branch carried an aria-label
  await expect(
    node(page, "Load CSV").getByLabel("CSV file path"),
  ).toBeVisible();
  await expect(node(page, "Compute FC").getByLabel("Prewhitening")).toBeVisible();
  await expect(
    node(page, "Compute FC").getByLabel("Per-window detrend"),
  ).toBeVisible();
});

test("run status is a live region (L1)", async ({ page }) => {
  await page.goto("/");
  await expect(page.locator('.run-status[role="status"]')).toHaveCount(1);
});

test("the settings dialog traps focus and names its close control (L1)", async ({
  page,
}) => {
  await page.goto("/");
  await page.getByRole("button", { name: "⚙ Settings" }).click();
  const dialog = page.getByRole("dialog", { name: "Settings" });
  await expect(dialog).toBeVisible();
  // focus moved into the dialog on open
  const inside = () =>
    page.evaluate(() => !!document.activeElement?.closest(".modal"));
  await expect.poll(inside).toBe(true);
  await expect(
    dialog.locator(".modal-head").getByRole("button", { name: "Close" }),
  ).toBeVisible();
  // Shift+Tab from the first control cycles back INSIDE (was: straight
  // through the overlay into the canvas)
  await page.keyboard.press("Shift+Tab");
  await expect.poll(inside).toBe(true);
  await page.keyboard.press("Escape");
  await expect(dialog).not.toBeVisible();
  // focus restored to the invoker
  await expect(page.getByRole("button", { name: "⚙ Settings" })).toBeFocused();
});
