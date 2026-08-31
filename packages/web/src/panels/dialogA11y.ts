import { useEffect, type RefObject } from "react";

/**
 * Dialog focus management (docs/issues.md L1): aria-modal alone traps nothing —
 * keyboard focus walked straight through the overlay into the canvas. On
 * mount the first focusable control is focused; Tab/Shift+Tab cycle within
 * the dialog; on unmount focus returns to the invoker. Escape closes
 * (stopping before any outer Escape handler).
 */
export function useDialogFocus(
  ref: RefObject<HTMLElement>,
  onClose: () => void,
) {
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const prev = document.activeElement as HTMLElement | null;
    const focusables = () =>
      Array.from(
        el.querySelectorAll<HTMLElement>(
          'a[href], button:not([disabled]), input:not([disabled]), '
          + 'select:not([disabled]), textarea:not([disabled]), '
          + '[tabindex]:not([tabindex="-1"])',
        ),
      ).filter((f) => f.offsetParent !== null);
    (focusables()[0] ?? el).focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.stopPropagation();
        onClose();
        return;
      }
      if (e.key !== "Tab") return;
      const f = focusables();
      if (f.length === 0) return;
      const first = f[0];
      const last = f[f.length - 1];
      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault();
        first.focus();
      }
    };
    el.addEventListener("keydown", onKey);
    return () => {
      el.removeEventListener("keydown", onKey);
      prev?.focus?.();
    };
    // the invoker re-renders with a fresh closure; the dialog's lifetime
    // is what the ref bound to, so the handler identity is irrelevant here
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
}
