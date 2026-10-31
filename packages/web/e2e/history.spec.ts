/** Session-only run history (docs/issues.md L32).
 *
 * The backend is the shared dev/CI instance, so each test clears the
 * in-memory history first (it is session-only and no other spec depends on
 * it). Runs are seeded through the API to keep the test focused on the
 * dialog; the label is the input-data basename derived server-side.
 */
import { expect, test, type Page } from "@playwright/test";

const CSV = "examples/data/150Hz_synthetic_CH4.csv";
const CSV_LABEL = "150Hz_synthetic_CH4.csv";

async function seedRun(page: Page, path: string): Promise<string> {
  const workflow = {
    format: "emptyui/workflow",
    version: 1,
    meta: { name: "history-e2e" },
    nodes: [{ id: "n1", type: "core.io.load_csv", params: { path } }],
    edges: [],
  };
  const resp = await page.request.post("/api/runs", { data: { workflow } });
  expect(resp.ok()).toBeTruthy();
  const { run_id } = (await resp.json()) as { run_id: string };
  await expect
    .poll(
      async () => {
        const r = await page.request.get(`/api/runs/${run_id}`);
        return ((await r.json()) as { status: string }).status;
      },
      { timeout: 30_000 },
    )
    .toMatch(/completed|failed|cancelled|timeout/);
  return run_id;
}

test("history lists a finished run, opens it, exports and deletes it (L32)", async ({
  page,
}) => {
  await page.goto("/");
  // wait for the catalog so Open restores the manifest title
  await expect(page.getByRole("button", { name: "Load CSV", exact: true })).toBeVisible();
  expect((await page.request.delete("/api/history")).ok()).toBeTruthy();
  await seedRun(page, CSV);

  await page.getByRole("button", { name: "🕘 History" }).click();
  const dialog = page.locator(".history-dialog");
  await expect(dialog).toBeVisible();
  const row = dialog.locator(".history-row").first();
  await expect(row.locator(".history-label")).toHaveText(CSV_LABEL);
  await expect(row.locator(".history-status")).toHaveText("completed");
  await expect(row.locator(".history-row-meta")).toContainText("1 node");

  // Export downloads the exact submitted snapshot.
  const [download] = await Promise.all([
    page.waitForEvent("download"),
    row.getByRole("button", { name: "Export" }).click(),
  ]);
  expect(download.suggestedFilename()).toBe(`${CSV_LABEL}.emptyui.json`);

  // Open restores the graph and the historical run state.
  await row.getByRole("button", { name: "Open" }).click();
  await expect(dialog).toBeHidden();
  await expect(page.locator(".mt-node", { hasText: "Load CSV" })).toBeVisible();
  await expect(page.locator(".run-status")).toHaveText("completed");

  // Delete removes the entry (confirm accepted).
  page.once("dialog", (d) => d.accept());
  await page.getByRole("button", { name: "🕘 History" }).click();
  await expect(dialog).toBeVisible();
  await dialog.locator(".history-row").first().getByRole("button", { name: "Delete" }).click();
  await expect(dialog.getByText(/No runs yet/)).toBeVisible();
  await expect(dialog.locator(".history-label")).toHaveCount(0);
});

test("a failed run is listed as failed and Clear all empties the table (L32)", async ({
  page,
}) => {
  await page.goto("/");
  expect((await page.request.delete("/api/history")).ok()).toBeTruthy();
  await seedRun(page, "examples/data/does-not-exist.csv");

  await page.getByRole("button", { name: "🕘 History" }).click();
  const dialog = page.locator(".history-dialog");
  await expect(dialog).toBeVisible();
  const row = dialog.locator(".history-row").first();
  await expect(row.locator(".history-status")).toHaveText("failed");
  await expect(row.locator(".history-label")).toHaveText("does-not-exist.csv");

  page.once("dialog", (d) => d.accept());
  await dialog.getByRole("button", { name: "Clear all" }).click();
  await expect(dialog.getByText(/No runs yet/)).toBeVisible();
});
