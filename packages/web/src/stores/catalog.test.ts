import { expect, test } from "vitest";
import { PORT_COLORS } from "./catalog";

test("no producer-less port types are advertised (issues.md L2)", () => {
  // the `selection` type returns with its producer (M4-deferred); until
  // then the palette must not advertise a color for it either
  expect("selection" in PORT_COLORS).toBe(false);
});
