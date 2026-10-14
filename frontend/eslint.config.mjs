/**
 * ESLint flat config.
 *
 * `package.json` previously declared `"lint": "next lint"`. Next.js 16 removed
 * that subcommand, so the string was parsed as a project directory and the
 * script exited 1 with "Invalid project directory provided, no such directory:
 * .../frontend/lint". Meanwhile `make lint` quietly ran `npm run typecheck`
 * instead — so the frontend was never linted, and the declared script was
 * guaranteed to fail for anyone who ran it.
 *
 * Per the Next.js 16 upgrade guide, `next lint` is replaced by the ESLint CLI
 * and `@next/eslint-plugin-next` now defaults to flat config.
 * `core-web-vitals` is the recommended set: it adds the Next.js and React
 * Hooks rules and raises the rules that affect Core Web Vitals from warnings
 * to errors.
 */

import { defineConfig, globalIgnores } from "eslint/config";
import nextVitals from "eslint-config-next/core-web-vitals";

export default defineConfig([
  ...nextVitals,
  globalIgnores([
    ".next/**",
    "out/**",
    "build/**",
    "next-env.d.ts",
  ]),
]);
