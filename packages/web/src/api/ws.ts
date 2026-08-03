import { create } from "zustand";
import { wsUrl } from "./client";
import { useRun } from "../stores/run";

let socket: WebSocket | null = null;
let retries = 0;

/** Socket liveness for the toolbar indicator (L-F3): the events WebSocket
 *  is a convenience (the polling fallback delivers the same events), but a
 *  silent give-up hid the outage — the status is surfaced loudly instead. */
type SocketStatus = "online" | "reconnecting" | "offline";
export const useSocketState = create<{ status: SocketStatus }>(() => ({
  status: "online",
}));

const MAX_RETRIES = 5;

function setStatus(s: SocketStatus) {
  useSocketState.setState({ status: s });
}

/** Lazily connects the events WebSocket; reconnects with backoff. */
export function ensureSocket(): WebSocket {
  if (socket && socket.readyState <= WebSocket.OPEN) return socket;
  try {
    socket = new WebSocket(wsUrl());
  } catch {
    // no WebSocket support (non-browser embedding) — polling covers it
    setStatus("offline");
    return null as unknown as WebSocket;
  }
  socket.onmessage = (msg) => {
    try {
      useRun.getState().handleEvent(JSON.parse(msg.data));
    } catch {
      // ignore malformed frames
    }
  };
  socket.onclose = () => {
    socket = null;
    retries += 1;
    // M15: never give up. The old code stopped after MAX_RETRIES and only a
    // page reload recovered (nothing re-called ensureSocket). The outage
    // stays LOUD — the badge shows "offline" once the fast retries are
    // exhausted — but a slow capped retry keeps running underneath, so the
    // connection self-heals when the backend comes back (no reload).
    setStatus(retries > MAX_RETRIES ? "offline" : "reconnecting");
    setTimeout(ensureSocket, Math.min(500 * 2 ** retries, 15_000));
  };
  socket.onopen = () => {
    retries = 0;
    setStatus("online");
  };
  return socket;
}
