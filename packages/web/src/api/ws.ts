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
    if (retries < MAX_RETRIES) {
      setStatus("reconnecting");
      setTimeout(() => {
        retries += 1;
        ensureSocket();
      }, 500 * 2 ** retries);
    } else {
      // L-F3: the give-up is LOUD — the toolbar shows the outage; rerunning
      // (or a page reload) calls ensureSocket() again and resets the count
      // via onopen
      setStatus("offline");
    }
  };
  socket.onopen = () => {
    retries = 0;
    setStatus("online");
  };
  return socket;
}
