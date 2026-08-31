/** Failure-surface e2e (DESIGN §8.1, docs/issues.md H6): a catalog request that
 *  never comes back, and a render that throws, must both land somewhere the
 *  user can act on — a message plus a retry, or the crash card — instead of
 *  the two silent outcomes the register found: an empty palette with a dead
 *  toolbar, or a white page. */

import { expect, test } from "@playwright/test";

const CATALOG = "**/api/nodes";

test("a failed catalog fetch surfaces in palette and toolbar; Retry recovers", async ({
  page,
}) => {
  let served = 0;
  await page.route(CATALOG, async (route) => {
    served += 1;
    await route.fulfill({
      status: 500,
      contentType: "text/plain",
      body: "injected catalog failure",
    });
  });

  await page.goto("/");

  // the reason is where the user is looking, not a silently empty sidebar
  const paletteError = page.locator(".palette-error");
  await expect(paletteError).toBeVisible({ timeout: 20_000 });
  await expect(paletteError).toContainText("Node catalog unavailable");
  await expect(paletteError).toContainText("GET /nodes: 500");
  await expect(page.locator("header .error-msg")).toContainText(
    "catalog: GET /nodes: 500",
  );
  await expect(
    page.getByRole("button", { name: "Retry catalog", exact: true }),
  ).toBeVisible();

  // the same button must actually recover the palette (route restored)
  await page.unroute(CATALOG);
  await page.getByRole("button", { name: "Retry catalog", exact: true }).click();
  await expect(paletteError).toHaveCount(0, { timeout: 15_000 });
  await expect(page.locator("header .error-msg")).toHaveCount(0, {
    timeout: 15_000,
  });
  await expect(
    page.getByRole("button", { name: "Load CSV", exact: true }),
  ).toBeVisible();
  expect(served).toBeGreaterThanOrEqual(1);
});

test("a render failure lands on the crash card, not a white screen", async ({
  page,
}) => {
  // DEV-only fault injection (main.tsx): throws inside the React tree, where
  // only an error boundary can recover — production builds compile it out.
  await page.goto("/?fault=render");

  const card = page.getByRole("alert");
  await expect(card).toBeVisible();
  await expect(card).toContainText("emptyUI crashed");
  await expect(card).toContainText("injected render failure");
  await expect(
    page.getByRole("button", { name: "Reload emptyUI", exact: true }),
  ).toBeVisible();

  // and the app is not wedged: without the fault the same page boots
  await page.goto("/");
  await expect(
    page.getByRole("button", { name: "Load CSV", exact: true }),
  ).toBeVisible();
});