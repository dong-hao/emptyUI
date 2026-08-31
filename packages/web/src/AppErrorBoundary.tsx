import { Component, type ErrorInfo, type ReactNode } from "react";

interface Props {
  children: ReactNode;
}

interface State {
  error: Error | null;
  copied: boolean;
}

/**
 * Top-level React error boundary (DESIGN §8.1, docs/issues.md H6).
 *
 * React only recovers from a render throw if an error boundary sits ABOVE
 * the component that threw, so without this the whole tree unmounts into a
 * white page — every later failure looks exactly like "the app never
 * started". The fallback swaps that for a readable card: the failure, a way
 * to copy the full chain, and a reload back into a clean state (the app
 * keeps no browser-side document, so reload == new workflow).
 */
export class AppErrorBoundary extends Component<Props, State> {
  state: State = { error: null, copied: false };

  static getDerivedStateFromError(error: Error): State {
    return { error, copied: false };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    // console keeps the whole chain (component stacks are dev-only)
    console.error("emptyUI crashed while rendering:", error, info.componentStack);
  }

  private readonly copyDetails = () => {
    const { error } = this.state;
    if (!error || !navigator.clipboard) return;
    const details = [error.message, error.stack ?? "", navigator.userAgent]
      .filter(Boolean)
      .join("\n\n");
    void navigator.clipboard.writeText(details).then(() => {
      this.setState({ copied: true });
    });
  };

  render() {
    const { error, copied } = this.state;
    if (!error) return this.props.children;
    return (
      <div className="crash-overlay" role="alert">
        <div className="crash-card">
          <h1>emptyUI crashed</h1>
          <p className="crash-msg">{error.message}</p>
          <p className="muted">
            The workflow could not be rendered. Nothing is stored in this
            browser, so reloading starts from a clean state.
          </p>
          {error.stack && (
            <details className="crash-details">
              <summary>Technical details</summary>
              <pre>{error.stack}</pre>
            </details>
          )}
          <div className="crash-actions">
            <button className="primary" onClick={() => window.location.reload()}>
              Reload emptyUI
            </button>
            <button onClick={this.copyDetails} disabled={copied}>
              {copied ? "Copied" : "Copy details"}
            </button>
          </div>
        </div>
      </div>
    );
  }
}