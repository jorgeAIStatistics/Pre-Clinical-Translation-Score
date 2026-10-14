/**
 * Publication-quality charts, hand-built on SVG.
 *
 * No charting library. Scientific charts need exact control over axes,
 * reference lines, interval rendering and annotation, and a general-purpose
 * library spends most of its API surface on things this product never needs
 * while making the three things it does need awkward. d3's scale functions do
 * the arithmetic; the marks are written out.
 *
 * Conventions across every chart here:
 *   · Reference lines (the null, perfect calibration, the cohort average) are
 *     drawn first and dashed, so a reader sees what the data is being compared
 *     against before seeing the data.
 *   · Uncertainty is drawn wherever it exists. A point estimate without its
 *     interval is not plotted.
 *   · Axes are labelled with units. Nothing is left to a caption.
 */

"use client";

import { scaleLinear, scalePoint } from "d3-scale";
import { line as d3line } from "d3-shape";
import type { ReactNode } from "react";

import { pct, riskColor } from "@/lib/format";

const AXIS = "var(--color-ink-300)";
const TEXT = "var(--color-ink-500)";
const GRID = "var(--color-ink-200)";

function AxisLabel({ x, y, children, anchor = "middle", rotate }: {
  x: number; y: number; children: ReactNode; anchor?: "start" | "middle" | "end"; rotate?: number;
}) {
  return (
    <text
      x={x} y={y} textAnchor={anchor} fill={TEXT}
      fontSize={9.5} fontFamily="var(--font-sans)"
      transform={rotate ? `rotate(${rotate} ${x} ${y})` : undefined}
    >
      {children}
    </text>
  );
}

/* ───────────────────────────────────────────────────────────────────────────
   Driver waterfall: the exact additive decomposition, drawn as it is computed.

   Bars run from a shared zero line. Because the contributions sum exactly to
   the log-odds difference from the cohort average, the chart *is* the
   arithmetic rather than an illustration of it.
   ─────────────────────────────────────────────────────────────────────────── */

export function DriverWaterfall({
  drivers, height = 22, maxBars = 10,
}: {
  drivers: Array<{ label: string; contribution: number; domain_label: string; is_missing: boolean; probability_delta: number }>;
  height?: number;
  maxBars?: number;
}) {
  const shown = drivers.slice(0, maxBars);
  if (!shown.length) return null;

  const max = Math.max(...shown.map((d) => Math.abs(d.contribution)), 0.05);
  const W = 420;
  const labelW = 210;
  const plotW = W - labelW - 56;
  const mid = labelW + plotW / 2;
  const x = scaleLinear().domain([-max, max]).range([labelW, labelW + plotW]);
  const H = shown.length * height + 26;

  return (
    <svg width="100%" viewBox={`0 0 ${W} ${H}`} role="img"
         aria-label="Per-feature contributions to the log-odds of failure">
      <line x1={mid} x2={mid} y1={16} y2={H - 8} stroke={AXIS} strokeWidth={1} />
      <AxisLabel x={x(-max * 0.55)} y={10}>← reduces risk</AxisLabel>
      <AxisLabel x={x(max * 0.55)} y={10}>increases risk →</AxisLabel>

      {shown.map((d, i) => {
        const y = 22 + i * height;
        const positive = d.contribution > 0;
        const x0 = positive ? mid : x(d.contribution);
        const w = Math.abs(x(d.contribution) - mid);
        const fill = positive ? "var(--color-risk-5)" : "var(--color-risk-1)";
        return (
          <g key={d.label + i}>
            <text
              x={labelW - 10} y={y + height / 2 + 3} textAnchor="end"
              fontSize={10.5} fill="var(--color-ink-700)" fontFamily="var(--font-sans)"
            >
              {d.label.length > 34 ? `${d.label.slice(0, 33)}…` : d.label}
              {d.is_missing ? <tspan fill={TEXT} fontSize={9}> (not reported)</tspan> : null}
            </text>
            <rect
              x={x0} y={y + 3} width={Math.max(w, 1)} height={height - 9}
              fill={fill} opacity={d.is_missing ? 0.45 : 0.85} rx={1}
            />
            {/* A hatch marks contributions that come from *absence* of evidence,
                so "we don't know this" never looks like a measured finding. */}
            {d.is_missing ? (
              <rect
                x={x0} y={y + 3} width={Math.max(w, 1)} height={height - 9}
                fill="url(#missing-hatch)" rx={1}
              />
            ) : null}
            <text
              x={labelW + plotW + 8} y={y + height / 2 + 3}
              fontSize={10} fill={TEXT} fontFamily="var(--font-mono)"
            >
              {d.probability_delta >= 0 ? "+" : ""}{(d.probability_delta * 100).toFixed(1)}%
            </text>
          </g>
        );
      })}

      <defs>
        <pattern id="missing-hatch" width="4" height="4" patternTransform="rotate(45)"
                 patternUnits="userSpaceOnUse">
          <line x1="0" y1="0" x2="0" y2="4" stroke="#fff" strokeWidth="1.4" opacity="0.8" />
        </pattern>
      </defs>
    </svg>
  );
}

/* ───────────────────────────────────────────────────────────────────────────
   Calibration plot. The diagonal is drawn first: it is the claim being tested.
   Wilson intervals on the observed rate, because calibration bins frequently
   sit at 0 or 1 where a normal approximation is nonsense.
   ─────────────────────────────────────────────────────────────────────────── */

export function CalibrationPlot({
  bins, slope, intercept, size = 300,
}: {
  bins: Array<{ mean_predicted: number; observed_rate: number; n: number; observed_ci_low: number; observed_ci_high: number }>;
  slope?: number | null;
  intercept?: number | null;
  size?: number;
}) {
  const m = { t: 12, r: 12, b: 34, l: 40 };
  const W = size, H = size;
  const x = scaleLinear().domain([0, 1]).range([m.l, W - m.r]);
  const y = scaleLinear().domain([0, 1]).range([H - m.b, m.t]);
  const maxN = Math.max(...bins.map((b) => b.n), 1);
  const r = scaleLinear().domain([1, maxN]).range([2.5, 7]);

  return (
    <svg width="100%" viewBox={`0 0 ${W} ${H}`} role="img"
         aria-label="Calibration: observed outcome rate against predicted probability">
      {[0, 0.25, 0.5, 0.75, 1].map((t) => (
        <g key={t}>
          <line x1={x(t)} x2={x(t)} y1={m.t} y2={H - m.b} stroke={GRID} strokeWidth={0.5} />
          <line x1={m.l} x2={W - m.r} y1={y(t)} y2={y(t)} stroke={GRID} strokeWidth={0.5} />
        </g>
      ))}

      {/* Perfect calibration */}
      <line x1={x(0)} y1={y(0)} x2={x(1)} y2={y(1)}
            stroke="var(--color-ink-400)" strokeWidth={1} strokeDasharray="3 3" />

      {bins.map((b, i) => (
        <g key={i}>
          <line
            x1={x(b.mean_predicted)} x2={x(b.mean_predicted)}
            y1={y(b.observed_ci_low)} y2={y(b.observed_ci_high)}
            stroke="var(--color-accent-600)" strokeWidth={1.25} opacity={0.65}
          />
          <circle
            cx={x(b.mean_predicted)} cy={y(b.observed_rate)} r={r(b.n)}
            fill="var(--color-accent-600)" opacity={0.85}
          />
        </g>
      ))}

      <line x1={m.l} x2={W - m.r} y1={H - m.b} y2={H - m.b} stroke={AXIS} />
      <line x1={m.l} x2={m.l} y1={m.t} y2={H - m.b} stroke={AXIS} />
      {[0, 0.5, 1].map((t) => (
        <g key={t}>
          <AxisLabel x={x(t)} y={H - m.b + 13}>{pct(t)}</AxisLabel>
          <AxisLabel x={m.l - 8} y={y(t) + 3} anchor="end">{pct(t)}</AxisLabel>
        </g>
      ))}
      <AxisLabel x={(m.l + W - m.r) / 2} y={H - 4}>Predicted probability</AxisLabel>
      <AxisLabel x={11} y={(m.t + H - m.b) / 2} rotate={-90}>Observed rate</AxisLabel>

      {slope != null && intercept != null ? (
        <text x={W - m.r - 4} y={m.t + 12} textAnchor="end" fontSize={9.5}
              fill={TEXT} fontFamily="var(--font-mono)">
          slope {slope.toFixed(2)} · intercept {intercept.toFixed(2)}
        </text>
      ) : null}
    </svg>
  );
}

/* ───────────────────────────────────────────────────────────────────────────
   Decision curve. Net benefit against the two reference strategies.
   ─────────────────────────────────────────────────────────────────────────── */

export function DecisionCurve({
  data, width = 620, height = 260,
}: {
  data: { thresholds: number[]; model: number[]; treat_all: number[]; treat_none: number[] };
  width?: number; height?: number;
}) {
  const m = { t: 14, r: 104, b: 36, l: 52 };

  // The y-axis is clipped, and it has to be.
  //
  // Net benefit for "flag all" is `prevalence − (1 − prevalence)·odds`, which
  // diverges to −∞ as the threshold approaches 1. Auto-scaling to include it
  // squashes every curve that matters into a few pixels at the top of the
  // plot — which is exactly what happened before this clip existed: the model
  // curve was visually indistinguishable from zero.
  //
  // Clipping the axis to the region where the decision is actually live is
  // standard practice in decision-curve analysis. The lower bound is a small
  // negative value rather than 0, because a model that is *worse* than doing
  // nothing must remain visible; it is just not allowed to set the scale.
  const modelMax = Math.max(...data.model, 0);
  const yMax = Math.max(modelMax * 1.25, 0.05);
  const yMin = Math.max(Math.min(...data.model, 0) - 0.02, -Math.max(yMax * 0.6, 0.06));

  const x = scaleLinear().domain([Math.min(...data.thresholds), Math.max(...data.thresholds)])
    .range([m.l, width - m.r]);
  const y = scaleLinear().domain([yMin, yMax]).nice().range([height - m.b, m.t]);
  const path = (vals: number[]) =>
    d3line<number>().x((_, i) => x(data.thresholds[i])).y((v) => y(v))(vals) ?? "";

  // Reference strategies leave the plot area; a clip path stops them drawing
  // over the axes instead of simply vanishing at the boundary.
  const clipId = "dca-clip";

  const series = [
    { key: "model", label: "Model", color: "var(--color-accent-600)", width: 2, dash: undefined },
    { key: "treat_all", label: "Flag all", color: "var(--color-ink-400)", width: 1.25, dash: "4 3" },
    { key: "treat_none", label: "Flag none", color: "var(--color-ink-300)", width: 1.25, dash: "2 3" },
  ] as const;

  return (
    <svg width="100%" viewBox={`0 0 ${width} ${height}`} role="img"
         aria-label="Decision curve: net benefit across risk thresholds">
      {y.ticks(5).map((t) => (
        <line key={t} x1={m.l} x2={width - m.r} y1={y(t)} y2={y(t)} stroke={GRID} strokeWidth={0.5} />
      ))}
      <line x1={m.l} x2={width - m.r} y1={y(0)} y2={y(0)} stroke={AXIS} strokeWidth={1} />

      <defs>
        <clipPath id={clipId}>
          <rect x={m.l} y={m.t} width={width - m.r - m.l} height={height - m.b - m.t} />
        </clipPath>
      </defs>
      <g clipPath={`url(#${clipId})`}>
        {series.map((s) => (
          <path key={s.key} d={path(data[s.key])} fill="none"
                stroke={s.color} strokeWidth={s.width} strokeDasharray={s.dash} />
        ))}
      </g>

      <line x1={m.l} x2={m.l} y1={m.t} y2={height - m.b} stroke={AXIS} />
      {y.ticks(5).map((t) => (
        <AxisLabel key={t} x={m.l - 7} y={y(t) + 3} anchor="end">{t.toFixed(2)}</AxisLabel>
      ))}
      {[0.1, 0.3, 0.5, 0.7, 0.9].map((t) => (
        <AxisLabel key={t} x={x(t)} y={height - m.b + 13}>{pct(t)}</AxisLabel>
      ))}
      <AxisLabel x={(m.l + width - m.r) / 2} y={height - 4}>Risk threshold</AxisLabel>
      <AxisLabel x={12} y={(m.t + height - m.b) / 2} rotate={-90}>Net benefit</AxisLabel>
      <text x={width - m.r + 8} y={height - m.b} fontSize={8.5} fill={TEXT}
            fontFamily="var(--font-sans)">
        <tspan x={width - m.r + 8} dy={0}>y-axis clipped;</tspan>
        <tspan x={width - m.r + 8} dy={10}>&ldquo;flag all&rdquo; diverges</tspan>
      </text>

      {series.map((s, i) => (
        <g key={s.key} transform={`translate(${width - m.r + 8}, ${m.t + 6 + i * 15})`}>
          <line x1={0} x2={14} y1={0} y2={0} stroke={s.color} strokeWidth={s.width}
                strokeDasharray={s.dash} />
          <text x={18} y={3} fontSize={9.5} fill={TEXT} fontFamily="var(--font-sans)">{s.label}</text>
        </g>
      ))}
    </svg>
  );
}

/* ───────────────────────────────────────────────────────────────────────────
   Scenario comparison: two intervals on one scale.
   ─────────────────────────────────────────────────────────────────────────── */

type ComparedEstimate = {
  probability: number | null;
  interval_low: number | null;
  interval_high: number | null;
};

export function ScenarioComparisonChart({
  current, scenario, width = 460,
}: {
  current: ComparedEstimate;
  scenario: ComparedEstimate;
  width?: number;
}) {
  // The scenario endpoint returns 409 rather than a comparison when either
  // side has no estimate, so this should be unreachable — but accepting
  // nullable inputs and drawing nothing is cheaper than an assertion that
  // renders NaN geometry if the contract ever changes.
  if (
    current.probability === null || current.interval_low === null ||
    current.interval_high === null || scenario.probability === null ||
    scenario.interval_low === null || scenario.interval_high === null
  ) {
    return (
      <p className="text-[12.5px] text-ink-500">
        No comparison to draw: at least one side has no estimate.
      </p>
    );
  }
  const lo = { ...current } as Required<Record<keyof ComparedEstimate, number>>;
  const hi = { ...scenario } as Required<Record<keyof ComparedEstimate, number>>;
  const m = { t: 22, r: 20, b: 30, l: 78 };
  const H = 112;
  const x = scaleLinear().domain([0, 1]).range([m.l, width - m.r]);
  const rows = [
    { label: "Current", d: lo, y: m.t + 12 },
    { label: "Scenario", d: hi, y: m.t + 48 },
  ];

  return (
    <svg width="100%" viewBox={`0 0 ${width} ${H}`} role="img"
         aria-label="Current versus scenario estimate with uncertainty intervals">
      {[0, 0.25, 0.5, 0.75, 1].map((t) => (
        <line key={t} x1={x(t)} x2={x(t)} y1={m.t} y2={H - m.b} stroke={GRID} strokeWidth={0.5} />
      ))}

      {rows.map((r) => (
        <g key={r.label}>
          <text x={m.l - 10} y={r.y + 4} textAnchor="end" fontSize={10.5}
                fill="var(--color-ink-600)" fontFamily="var(--font-sans)">{r.label}</text>
          <rect
            x={x(r.d.interval_low)} y={r.y - 6}
            width={Math.max(x(r.d.interval_high) - x(r.d.interval_low), 1)} height={12}
            fill={riskColor(r.d.probability)} opacity={0.32} rx={2}
          />
          <rect x={x(r.d.probability) - 1} y={r.y - 9} width={2} height={18}
                fill={riskColor(r.d.probability)} />
          <text x={x(r.d.probability)} y={r.y - 13} textAnchor="middle" fontSize={10}
                fill="var(--color-ink-700)" fontFamily="var(--font-mono)">
            {pct(r.d.probability)}
          </text>
        </g>
      ))}

      <line x1={m.l} x2={width - m.r} y1={H - m.b} y2={H - m.b} stroke={AXIS} />
      {[0, 0.25, 0.5, 0.75, 1].map((t) => (
        <AxisLabel key={t} x={x(t)} y={H - m.b + 13}>{pct(t)}</AxisLabel>
      ))}
    </svg>
  );
}

/* ───────────────────────────────────────────────────────────────────────────
   Uncertainty composition: which source of ignorance is driving the width.
   ─────────────────────────────────────────────────────────────────────────── */

const SOURCE_COPY: Record<string, { label: string; blurb: string }> = {
  parameter: {
    label: "Model parameters",
    blurb: "Uncertainty about the coefficients given the training cohort.",
  },
  imputation: {
    label: "Missing evidence",
    blurb: "Cost of features this program does not report.",
  },
  calibration: {
    label: "Calibration",
    blurb: "Uncertainty in mapping the model score onto a probability.",
  },
};

export function UncertaintyComposition({
  shares, sds,
}: { shares: Record<string, number>; sds: Record<string, number> }) {
  const entries = Object.entries(shares)
    .filter(([, v]) => v > 0.0005)
    .sort((a, b) => b[1] - a[1]);
  if (!entries.length) return null;
  const colors: Record<string, string> = {
    parameter: "var(--color-accent-600)",
    imputation: "var(--color-risk-4)",
    calibration: "var(--color-ink-400)",
  };

  return (
    <div>
      <div className="flex h-2.5 w-full overflow-hidden rounded-[2px]" role="img"
           aria-label="Share of prediction uncertainty by source">
        {entries.map(([k, v]) => (
          <div key={k} style={{ width: `${v * 100}%`, background: colors[k] ?? "var(--color-ink-300)" }} />
        ))}
      </div>
      <dl className="mt-3 space-y-2">
        {entries.map(([k, v]) => (
          <div key={k} className="flex items-baseline gap-2">
            <span aria-hidden className="mt-[3px] inline-block h-2 w-2 shrink-0 rounded-[1px]"
                  style={{ background: colors[k] ?? "var(--color-ink-300)" }} />
            <dt className="text-[12px] font-medium text-ink-700">
              {SOURCE_COPY[k]?.label ?? k}
            </dt>
            <dd className="tnum font-mono text-[11.5px] text-ink-500">
              {(v * 100).toFixed(0)}% · SD {sds[k]?.toFixed(2) ?? "—"}
            </dd>
          </div>
        ))}
      </dl>
      <p className="mt-2 text-[11.5px] leading-[1.55] text-ink-500">
        {SOURCE_COPY[entries[0][0]]?.blurb}
      </p>
    </div>
  );
}

/* ───────────────────────────────────────────────────────────────────────────
   Domain contribution strip: the scorecard's per-domain roll-up.
   ─────────────────────────────────────────────────────────────────────────── */

export function DomainContributions({
  domains,
}: { domains: Array<{ domain_label: string; contribution: number; n_drivers: number }> }) {
  if (!domains.length) return null;
  const max = Math.max(...domains.map((d) => Math.abs(d.contribution)), 0.05);
  return (
    <ul className="space-y-2">
      {domains.map((d) => {
        const w = (Math.abs(d.contribution) / max) * 50;
        const positive = d.contribution > 0;
        return (
          <li key={d.domain_label} className="grid grid-cols-[minmax(0,1fr)_112px_54px] items-center gap-3">
            <span className="truncate text-[12px] text-ink-700">{d.domain_label}</span>
            <span className="relative flex h-3 items-center">
              <span aria-hidden className="absolute left-1/2 h-full w-px bg-ink-300" />
              <span
                className="absolute h-[7px] rounded-[1px]"
                style={{
                  background: positive ? "var(--color-risk-5)" : "var(--color-risk-1)",
                  opacity: 0.8,
                  width: `${w}%`,
                  left: positive ? "50%" : `${50 - w}%`,
                }}
              />
            </span>
            <span className="tnum text-right font-mono text-[11px] text-ink-500">
              {d.contribution >= 0 ? "+" : ""}{d.contribution.toFixed(2)}
            </span>
          </li>
        );
      })}
    </ul>
  );
}
