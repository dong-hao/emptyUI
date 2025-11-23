import { wsUrl } from "./client";
import { useRun } from "../stores/run";

let socket: WebSocket | null = null;
let retries = 0;

/** Lazily connects the events WebSocket; reconnects with backoff. */
export function ensureSocket(): WebSocket {
  if (socket && socket.readyState <= WebSocket.OPEN) return socket;

  socket = new WebSocket(wsUrl());
  socket.onmessage = (msg) => {
    try {
      useRun.getState().handleEvent(JSON.parse(msg.data));
    } catch {
      // ignore malformed frames
    }
  };
  socket.onclose = () => {
    socket = null;
    if (retries < 5) {
      setTimeout(() => {
        retries += 1;
        ensureSocket();
      }, 500 * 2 ** retries);
    }
  };
  socket.onopen = () => {
    retries = 0;
  };
  return socket;
}
