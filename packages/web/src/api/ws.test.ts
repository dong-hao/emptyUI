import { afterEach, beforeEach, expect, test, vi } from "vitest";
// wsUrl comes from client.ts, which reads document.baseURI — no DOM here
vi.mock("./client", () => ({ wsUrl: () => "ws://test/api/events" }));
import { ensureSocket, useSocketState } from "./ws";

beforeEach(() => {
  vi.useFakeTimers();
  useSocketState.setState({ status: "online" });
});

afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
});

class FakeSocket {
  static instances: FakeSocket[] = [];
  onopen: (() => void) | null = null;
  onclose: (() => void) | null = null;
  onmessage: ((msg: unknown) => void) | null = null;
  readyState: number = WebSocket.CONNECTING;
  constructor() {
    FakeSocket.instances.push(this);
  }
}

vi.stubGlobal("WebSocket", FakeSocket);

beforeEach(() => {
  FakeSocket.instances = [];
});

test("the socket keeps retrying after MAX_RETRIES instead of giving up (M15)", async () => {
  ensureSocket();
  expect(FakeSocket.instances).toHaveLength(1);
  // five failed connections: the old code scheduled exactly 5 retries then
  // gave up forever (status offline, no more attempts)
  for (let i = 0; i < 8; i++) {
    FakeSocket.instances[FakeSocket.instances.length - 1].onclose?.();
    await vi.runAllTimersAsync();
  }
  // the outage is loud ...
  expect(useSocketState.getState().status).toBe("offline");
  // ... but the socket KEEPS retrying: 8 closures => 8 attempts total
  expect(FakeSocket.instances.length).toBeGreaterThanOrEqual(8);
  // and a late success recovers the badge without a page reload
  const last = FakeSocket.instances[FakeSocket.instances.length - 1];
  last.readyState = WebSocket.OPEN;
  last.onopen?.();
  expect(useSocketState.getState().status).toBe("online");
});
