import { expect, test } from "@playwright/test";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const DOC = path.resolve(
  here,
  "..",
  "..",
  "..",
  "examples",
  "workflow",
  "phoenix-mtu5a-day-12h.emptyui.json",
);
const doc = JSON.parse(await readFile(DOC, "utf-8"));

// The shipped example carries already-migrated Extract Run edges (the
// legacy two-port documents no longer exist in the repo), so the legacy
// one-port migration is exercised by reverting the first Extract Run
// output edge to the retired `runs` source port BEFORE the UI sees the
// document. Everything else is the real shipped file — node/edge counts
// and ids round-trip unchanged.
const extractIds = new Set(
  doc.nodes
    .filter((n: { type: string }) => n.type === "mth5.io.extract_run")
    .map((n: { id: string }) => n.id),
);
const legacy = JSON.parse(JSON.stringify(doc));
const legacyEdge = legacy.edges.find((e: { source: string }) =>
  extractIds.has(e.source),
);

test("loading the shipped example migrates the legacy one-port Extract Run edge (§4)", async ({
  page,
}) => {
  expect(extractIds.size, "fixture must contain an Extract Run node").toBeGreaterThan(0);
  expect(legacyEdge, "fixture must wire Extract Run downstream").toBeTruthy();
  legacyEdge.source_port = "runs";

  await page.goto("/");
  await expect(page.locator(".mt-node")).toHaveCount(0);

  await page.locator('input[type="file"]').setInputFiles({
    name: "phoenix-mtu5a-day-12h-legacy.emptyui.json",
    mimeType: "application/json",
    buffer: Buffer.from(JSON.stringify(legacy)),
  });
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
  // the migrated edge is the one we downgraded; its source is the
  // Extract Run node and it is back on `data`
  const migrated = saved.edges.find(
    (e: { id: string }) => e.id === legacyEdge.id,
  );
  expect(migrated.source_port).toBe("data");
  expect(extractIds.has(migrated.source)).toBe(true);
});