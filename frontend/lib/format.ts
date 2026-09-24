/**
 * Formatting helpers.
 *
 * Presentation rules that recur across the interface live here so they cannot
 * drift between surfaces: a probability rendered as "68%" on one page and
 * "0.684" on another makes two numbers look like two findings.
 */

import type { ConfidenceLevel } from "./types";

/** Probabilities are shown as whole percentages. Sub-percent precision on a
 *  quantity with a ±20-point interval is false precision. */
export const pct = (v: number | null | undefined, digits = 0): string =>
  v === null || v === undefined || Number.isNaN(v) ? "—" : `${(v * 100).toFixed(digits)}%`;

export const num = (v: number | null | undefined, digits = 2): string =>
  v === null || v === undefined || Number.isNaN(v) ? "—" : v.toFixed(digits);

export const signed = (v: number | null | undefined, digits = 2): string =>
  v === null || v === undefined || Number.isNaN(v)
    ? "—"
    : `${v >= 0 ? "+" : ""}${v.toFixed(digits)}`;

export const signedPct = (v: number | null | undefined, digits = 1): string =>
  v === null || v === undefined || Number.isNaN(v)
    ? "—"
    : `${v >= 0 ? "+" : ""}${(v * 100).toFixed(digits)}%`;

/** Interval rendered as a range, never as "± x" — the posterior is not
 *  symmetric on the probability scale and "±" would imply that it is. */
export const interval = (lo: number, hi: number, digits = 0): string =>
  `${pct(lo, digits)} – ${pct(hi, digits)}`;

export const ci = (
  e: { value: number | null; ci_low: number | null; ci_high: number | null } | undefined,
  digits = 3,
): string => {
  if (!e || e.value === null) return "—";
  if (e.ci_low === null || e.ci_high === null) return num(e.value, digits);
  return `${num(e.value, digits)} (${num(e.ci_low, digits)}–${num(e.ci_high, digits)})`;
};

/** snake_case → Sentence case, with acronyms preserved. */
const ACRONYMS = new Set([
  "pk", "pd", "pkpd", "adc", "gpcr", "cta", "ind", "auroc", "auprc", "ece",
  "gemm", "pdx", "cdx", "mss", "tnbc", "hgsoc", "mcrpc", "ccrcc", "aml", "cll",
  "dlbcl", "nhp", "coi", "soc", "qd", "bid", "iv", "po", "ip",
]);

export const humanise = (value: string | null | undefined): string => {
  if (!value) return "—";
  return value
    .split(/[_\s]+/)
    .map((w, i) => {
      const lower = w.toLowerCase();
      if (ACRONYMS.has(lower)) return lower.toUpperCase();
      return i === 0 ? w.charAt(0).toUpperCase() + w.slice(1).toLowerCase() : lower;
    })
    .join(" ");
};

export const shortDate = (iso: string | null | undefined): string => {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "—";
  return d.toLocaleDateString("en-GB", { year: "numeric", month: "short", day: "numeric" });
};

export const year = (iso: string | null | undefined): string =>
  iso ? String(new Date(iso).getFullYear()) : "—";

export const CONFIDENCE_LABEL: Record<ConfidenceLevel, string> = {
  high: "High",
  moderate: "Moderate",
  low: "Low",
  very_low: "Very low",
  not_assessed: "Not assessed",
};

/** Position on the 5-step risk ramp. Colour is a redundant channel; the
 *  number and the bar position carry the meaning. */
export const riskStep = (p: number | null | undefined): 1 | 2 | 3 | 4 | 5 => {
  if (p === null || p === undefined) return 3;
  if (p < 0.25) return 1;
  if (p < 0.45) return 2;
  if (p < 0.6) return 3;
  if (p < 0.78) return 4;
  return 5;
};

export const riskColor = (p: number | null | undefined): string =>
  `var(--color-risk-${riskStep(p)})`;

/** A written band, so risk is legible without reading the colour at all. */
export const riskBand = (p: number | null | undefined): string => {
  if (p === null || p === undefined) return "Not estimated";
  return (["Lower", "Moderate-low", "Moderate", "Elevated", "High"] as const)[riskStep(p) - 1];
};

export const OUTCOME_TONE: Record<string, "failure" | "success" | "neutral"> = {
  phase_2_efficacy_failure: "failure",
  phase_2_safety_failure: "failure",
  phase_1_safety_failure: "failure",
  phase_1_pkpd_failure: "failure",
  failed_before_human_dosing: "failure",
  phase_3_failure: "failure",
  progressed_past_phase_2: "success",
  regulatory_approval: "success",
  discontinued_through_phase_2: "neutral",
  active_no_outcome_yet: "neutral",
  censored_insufficient_followup: "neutral",
  unknown: "neutral",
};
