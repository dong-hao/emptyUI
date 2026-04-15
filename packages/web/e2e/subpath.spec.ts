/** Sub-path mount e2e (M-F5): the app must work when served under a
 *  sub-path (e.g. /tools/emptyui/) — apiBase, wsUrl and artifact URLs all
 *  derive from document.baseURI. The proxy is a tiny path-stripping
 *  reverse proxy in front of the vite dev server (which proxies /api to
 *  the backend): HTTP requests stream through; WebSocket upgrades are
 *  piped socket-to-socket with the prefix stripped. */

import { expect, test } from "@playwright/test";
import http from "node:http";
import net from "node:net";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { addNode, connect, node, placeNode, settleViewport } from "./_flow";

const here = path.dirname(fileURLToPath(import.meta.url));
const MOUNT = "/tools/emptyui";
const PROXY_PORT = 5199;
const TARGET_PORT = 5173;

let server: http.Server;

function stripPrefix(url: string): string {
  return url.startsWith(`${MOUNT}/`) || url === MOUNT
    ? url.slice(MOUNT.length) || "/"
    : url;
}

test.beforeAll(async () => {
  server = http.createServer((req, res) => {
    const url = stripPrefix(req.url ?? "/");
    const headers = { ...req.headers, host: `localhost:${TARGET_PORT}` };
    const proxied = http.request(
      { host: "localhost", port: TARGET_PORT, path: url, method: req.method, headers },
      (up) => {
        res.writeHead(up.statusCode ?? 502, up.headers);
        up.pipe(res);
      },
    );
    proxied.on("error", () => {
      res.statusCode = 502;
      res.end();
    });
    req.pipe(proxied);
  });
  server.on("upgrade", (req, socket, head) => {
    const url = stripPrefix(req.url ?? "/");
    const upstream = net.connect(TARGET_PORT, "localhost", () => {
      const lines = [
        `${req.method} ${url} HTTP/1.1`,
        ...Object.entries(req.headers)
          .map(([k, v]) => (k.toLowerCase() === "host"
            ? `host: localhost:${TARGET_PORT}`
            : `${k}: ${v}`)),
      ];
      upstream.write(lines.join("\r\n") + "\r\n\r\n");
      if (head.length) upstream.write(head);
      upstream.pipe(socket);
      socket.pipe(upstream);
    });
    upstream.on("error", () => socket.destroy());
  });
  await new Promise<void>((r) => server.listen(PROXY_PORT, r));
});

test.afterAll(async () => {
  await new Promise<void>((r) => server.close(() => r()));
});

test("the app boots, runs, and derives artifact URLs under a sub-path", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1920, height: 1080 });
  await page.goto(`http://localhost:${PROXY_PORT}${MOUNT}/`);

  // The catalog loads through the prefixed API path.
  await expect(page.locator(".palette h3").first()).toHaveText("Data I/O");

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

  // Artifact URLs derive from the SUB-PATH base (M-F5).
  const src = await node(page, "Preview Plot").locator("img").getAttribute("src");
  expect(src).toContain(`${MOUNT}/api/artifacts/`);
});
