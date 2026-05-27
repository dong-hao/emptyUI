// ESLint flat config (ESLint 9). `npm run lint` fails on errors; warnings are
// reported but do not fail the gate. Scope: the frontend sources, the
// Playwright suite and the config files (P0.1, P0.3).
import js from "@eslint/js";
import globals from "globals";
import reactHooks from "eslint-plugin-react-hooks";
import tseslint from "typescript-eslint";

export default tseslint.config(
  { ignores: ["dist", "node_modules", "test-results", "playwright-report"] },
  js.configs.recommended,
  ...tseslint.configs.recommended,
  {
    files: ["**/*.{ts,tsx}"],
    languageOptions: { ecmaVersion: 2022, globals: globals.browser },
    plugins: { "react-hooks": reactHooks },
    rules: {
      "react-hooks/rules-of-hooks": "error",
      "react-hooks/exhaustive-deps": "warn",
    },
  },
  {
    files: ["e2e/**/*.ts", "*.config.ts", "vite.config.ts", "eslint.config.js"],
    languageOptions: { globals: globals.node },
  },
);
