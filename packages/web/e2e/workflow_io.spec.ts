import { expect, test } from "@playwright/test";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const DOC = path.resolve(here, "..", "..", "..", "day_12h.emptyui.json");
const doc = JSON.parse(await readFile(DOC, "utf-8"));

test("loading the shipped example migrates the legacy one-port Extract Run edge (§4)", async ({
  page,
}) => {
  await page.goto("/");
  await expect(page.locator(".mt-node")).toHaveCount(0);

  await page.locator('input[type="file"]').setInputFiles(DOC);
  await expect(page.locator(".mt-node")).toHaveCount(doc.nodes.length);
  // every edge survives: the legacy runs-port edge is migrated (source
  // handle runs -> data) BEFORE stale-edge pruning runs — the retired
  // port is no longer declared by the manifest, so pruning first would
  // eat it and the Extract Run -> Compute FC wiring would vanish on load
  await expect(page.locator(".react-flow__edge")).toHaveCount(doc.edges.length);

  // round-trip: the re-saved document carries the migrated port name
  const downloadPromise = page.waitForEvent("download");
  await page.getByRole("button", { name: "Save", exact: true }).click();
  const download = await downloadPromise;
  const saved = JSON.parse(
    await readFile((await download.path())!, "utf-8"),
  );
  expect(saved.edges).toHaveLength(doc.edges.length);
  expect(saved.edges.some((e: { source_port: string }) => e.source_port === "runs")).toBe(false);
  // the migrated edge's source is the Extract Run node, now on `data`
  const extractIds = new Set(
    saved.nodes
      .filter((n: { type: string }) => n.type === "mth5.io.extract_run")
      .map((n: { id: string }) => n.id),
  );
  const migrated = saved.edges.find(
    (e: { id: string }) => e.id === "efb0b2d92-3a44-4c59-9f4c-dc127e4d97ca",
  );
  expect(migrated.source_port).toBe("data");
  expect(extractIds.has(migrated.source)).toBe(true);
});
