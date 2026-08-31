import React from "react";
import ReactDOM from "react-dom/client";
import { QueryClientProvider } from "@tanstack/react-query";
import { queryClient } from "./api/queryClient";
import { AppErrorBoundary } from "./AppErrorBoundary";
import App from "./App";
import "./styles.css";

/** Dev-only fault injection (docs/issues.md H6, DESIGN §8.1): `?fault=render`
 *  throws during the render pass so the boundary above it can be exercised
 *  end-to-end. `import.meta.env.DEV` compiles to `false` in every production
 *  bundle — the check, and the throw behind it, are dropped by the build. */
function FaultInjector() {
  if (
    import.meta.env.DEV &&
    new URLSearchParams(window.location.search).get("fault") === "render"
  ) {
    throw new Error("injected render failure (?fault=render)");
  }
  return null;
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <AppErrorBoundary>
        <FaultInjector />
        <App />
      </AppErrorBoundary>
    </QueryClientProvider>
  </React.StrictMode>,
);
