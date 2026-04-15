/** WS outage e2e (DESIGN §8.1, L-F3): severing the events WebSocket must
 *  surface the outage loudly (the toolbar badge) while the run keeps
 *  working — the polling fallback delivers the same events and is
 *  deduped against the WS replay in the run store. */

import { expect, test } from "@playwright/test";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { addNode, connect, node, placeNode, settleViewport } from "./_flow";

const here = path.dirname(fileURLToPath(import.meta.url));

test("a severed WebSocket shows the badge; the run completes via polling", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1920, height: 1080 });
  // Every WebSocket connection attempt is closed by the server side.
  await page.routeWebSocket(/\/api\/ws$/, (ws) => {
    ws.close();
  });

  await page.goto("/");
  // The badge appears once the first close fires (amber reconnecting).
  await expect(page.locator(".socket-badge.reconnecting")).toBeVisible({
    timeout: 15_000,
  });

  // The run still works — the polling fallback delivers the events.
  const CSV_PATH = path.resolve(
    here, "..", "..", "..", "examples", "data", "150Hz_synthetic_CH4.csv",
  );
  await addNode(page, "Load CSV");
  await addNode(page, "Plot Time Series");
  await addNode(page, "Preview Plot");
  await page.getByRole("button", { name: "Fit View" }).click();
  await settleViewport(page);
  await placeNode(page, "Preview Plot", 1050, 150);
  await placeNode(page, "Plot Time Series", 600, 150);
  await placeNode(page, "Load CSV", 150, 150);
  await page.getByRole("button", { name: "Fit View" }).click();
  await settleViewport(page);

  await node(page, "Load CSV").locator(".path-field input").fill(CSV_PATH);
  await node(page, "Load CSV").locator(".path-field input").press("Enter");
  await connect(page, { node: "Load CSV", port: "time series: ts.run" }, {
    node: "Plot Time Series",
    port: "time series: ts.run",
  });
  await connect(page, { node: "Plot Time Series", port: "plot: plot" }, {
    node: "Preview Plot",
    port: "plot: plot",
  });

  await page.getByRole("button", { name: "▶ Run" }).click();
  await expect(page.locator(".run-status.completed")).toBeVisible({
    timeout: 120_000,
  });
});
