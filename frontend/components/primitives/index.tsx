/**
 * Reusable interface primitives.
 *
 * Kept deliberately small. A scientific tool earns trust through consistency
 * and legibility, not through a large component inventory; every element here
 * exists because it appears on at least two surfaces.
 */

import clsx from "clsx";
import Link from "next/link";
import type { ReactNode } from "react";

import { CONFIDENCE_LABEL, humanise, riskBand, riskColor } from "@/lib/format";
import type { ConfidenceLevel } from "@/lib/types";

/* ── Layout ─────────────────────────────────────────────────────────────── */

export function Card({
  children, className, as: Tag = "section",
}: { children: ReactNode; className?: string; as?: "section" | "div" | "article" }) {
  return (
    <Tag
      className={clsx(
        "rounded-[var(--radius-card)] border border-[var(--border-subtle)]",
        "bg-[var(--surface-raised)]",
        className,
      )}
    >
      {children}
    </Tag>
  );
}

export function CardHeader({
  title, subtitle, action, id,
}: { title: ReactNode; subtitle?: ReactNode; action?: ReactNode; id?: string }) {
  return (
    <div className="flex items-start justify-between gap-6 border-b border-[var(--border-subtle)] px-5 py-4">
      <div className="min-w-0">
        <h2 id={id} className="text-[13px] font-semibold tracking-[0.02em] text-ink-800">
          {title}
        </h2>
        {subtitle ? (
          <p className="mt-1 max-w-prose text-[12.5px] leading-[1.55] text-ink-500">{subtitle}</p>
        ) : null}
      </div>
      {action ? <div className="shrink-0">{action}</div> : null}
    </div>
  );
}

export function SectionLabel({ children }: { children: ReactNode }) {
  return (
    <div className="text-[10.5px] font-semibold uppercase tracking-[0.11em] text-ink-400">
      {children}
    </div>
  );
}

/* ── Text ───────────────────────────────────────────────────────────────── */

export function Prose({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <p className={clsx("max-w-[68ch] text-[13px] leading-[1.65] text-ink-600", className)}>
      {children}
    </p>
  );
}

export function Stat({
  label, value, hint, mono = true,
}: { label: ReactNode; value: ReactNode; hint?: ReactNode; mono?: boolean }) {
  return (
    <div>
      <div className="text-[10.5px] font-medium uppercase tracking-[0.09em] text-ink-400">
        {label}
      </div>
      <div className={clsx("mt-1 text-[15px] text-ink-800", mono && "tnum font-mono")}>{value}</div>
      {hint ? <div className="mt-0.5 text-[11.5px] text-ink-500">{hint}</div> : null}
    </div>
  );
}

/* ── Badges ─────────────────────────────────────────────────────────────── */

export function Badge({
  children, tone = "neutral", title,
}: {
  children: ReactNode;
  tone?: "neutral" | "accent" | "support" | "refute" | "warn" | "muted";
  title?: string;
}) {
  const tones = {
    neutral: "bg-ink-100 text-ink-700 border-ink-200",
    accent: "bg-accent-50 text-accent-700 border-accent-100",
    support: "bg-[#eaf4f1] text-[#1f5c50] border-[#cfe5df]",
    refute: "bg-[#f8ece8] text-[#8c4633] border-[#eed6cd]",
    warn: "bg-[#fdf4e3] text-[#8a6320] border-[#f0e0bd]",
    muted: "bg-transparent text-ink-500 border-ink-200",
  } as const;
  return (
    <span
      title={title}
      className={clsx(
        "inline-flex items-center gap-1 rounded-[3px] border px-1.5 py-[1px]",
        "text-[11px] font-medium leading-[1.5] whitespace-nowrap",
        tones[tone],
      )}
    >
      {children}
    </span>
  );
}

/**
 * Evidence direction. Carries a glyph as well as a colour so the distinction
 * survives greyscale and colour-vision deficiency.
 */
export function DirectionBadge({ direction }: { direction: string }) {
  const map: Record<string, { tone: "support" | "refute" | "muted"; glyph: string; label: string }> = {
    supports: { tone: "support", glyph: "▲", label: "Supports" },
    refutes: { tone: "refute", glyph: "▼", label: "Refutes" },
    mixed: { tone: "muted", glyph: "◆", label: "Mixed" },
    neutral: { tone: "muted", glyph: "—", label: "Neutral" },
    unknown: { tone: "muted", glyph: "?", label: "Unknown" },
  };
  const m = map[direction] ?? map.unknown;
  return (
    <Badge tone={m.tone}>
      <span aria-hidden className="text-[9px]">{m.glyph}</span>
      {m.label}
    </Badge>
  );
}

export function ConfidenceBadge({ level }: { level: ConfidenceLevel | null | undefined }) {
  if (!level) return <Badge tone="muted">Not assessed</Badge>;
  const tone = level === "high" || level === "moderate" ? "neutral" : "warn";
  // Filled pips make the ordering readable without relying on the word alone.
  const filled = { high: 4, moderate: 3, low: 2, very_low: 1, not_assessed: 0 }[level];
  return (
    <Badge tone={tone} title={`Evidence confidence: ${CONFIDENCE_LABEL[level]}`}>
      <span aria-hidden className="tracking-[-1px] text-[8px]">
        {"●".repeat(filled)}
        <span className="text-ink-300">{"○".repeat(4 - filled)}</span>
      </span>
      {CONFIDENCE_LABEL[level]}
    </Badge>
  );
}

/* ── Risk display ───────────────────────────────────────────────────────── */

/**
 * The interval bar.
 *
 * The full 0–100% track is always drawn, so two programs' intervals are
 * comparable at a glance and a wide interval *looks* wide. The point estimate
 * is a tick, not a fill, because a filled bar reads as "amount of risk" rather
 * than "location of the estimate".
 */
export function IntervalBar({
  low, high, point, height = 8, showScale = false,
}: { low: number; high: number; point: number; height?: number; showScale?: boolean }) {
  const color = riskColor(point);
  return (
    <div className="w-full">
      <div
        className="relative w-full overflow-hidden rounded-[2px] bg-ink-100"
        style={{ height }}
        role="img"
        aria-label={`Estimate ${(point * 100).toFixed(0)} percent, interval ${(low * 100).toFixed(0)} to ${(high * 100).toFixed(0)} percent`}
      >
        <div
          className="absolute inset-y-0 rounded-[2px] opacity-40"
          style={{ left: `${low * 100}%`, width: `${Math.max(high - low, 0.004) * 100}%`, background: color }}
        />
        <div
          className="absolute inset-y-0 w-[2px]"
          style={{ left: `calc(${point * 100}% - 1px)`, background: color }}
        />
      </div>
      {showScale ? (
        <div className="mt-1 flex justify-between text-[9.5px] tnum font-mono text-ink-400">
          <span>0%</span><span>25%</span><span>50%</span><span>75%</span><span>100%</span>
        </div>
      ) : null}
    </div>
  );
}

export function RiskBandLabel({ p }: { p: number | null | undefined }) {
  return (
    <span className="inline-flex items-center gap-1.5 text-[11.5px] text-ink-600">
      <span
        aria-hidden
        className="inline-block h-2 w-2 rounded-[1px]"
        style={{ background: riskColor(p) }}
      />
      {riskBand(p)}
    </span>
  );
}

/* ── Tables ─────────────────────────────────────────────────────────────── */

export function Th({
  children, className, align = "left", scope = "col", ariaSort,
}: {
  children: ReactNode;
  className?: string;
  align?: "left" | "right" | "center";
  scope?: "col" | "row";
  /**
   * Sort state for a sortable column.
   *
   * `aria-sort` is only valid on an element with the `columnheader` role, so it
   * belongs here on the `<th>` — not on a `<button>` inside it, where it was
   * previously set and where assistive technology ignores it. That meant a
   * screen-reader user could operate the sort control and never be told which
   * column was sorted or in which direction.
   */
  ariaSort?: "ascending" | "descending" | "none";
}) {
  return (
    <th
      scope={scope}
      aria-sort={ariaSort}
      className={clsx(
        "whitespace-nowrap border-b border-[var(--border-strong)] px-3 py-2",
        "text-[10.5px] font-semibold uppercase tracking-[0.08em] text-ink-500",
        align === "right" && "text-right",
        align === "center" && "text-center",
        className,
      )}
    >
      {children}
    </th>
  );
}

export function Td({
  children, className, align = "left", mono = false, colSpan,
}: {
  children: ReactNode;
  className?: string;
  align?: "left" | "right" | "center";
  mono?: boolean;
  colSpan?: number;
}) {
  return (
    <td
      colSpan={colSpan}
      className={clsx(
        "border-b border-[var(--border-subtle)] px-3 py-2 text-[12.5px] text-ink-700",
        align === "right" && "text-right",
        align === "center" && "text-center",
        mono && "tnum font-mono",
        className,
      )}
    >
      {children}
    </td>
  );
}

/* ── Notices ────────────────────────────────────────────────────────────── */

/**
 * The demonstration-data banner.
 *
 * Deliberately not dismissible and deliberately prominent. A user who forgets
 * that a number came from a simulation is the failure mode this whole platform
 * is built to avoid, so the reminder sits above the content rather than in a
 * footnote.
 */
export function DemoBanner({ note }: { note?: string | null }) {
  return (
    <div
      role="note"
      className="flex items-start gap-3 border-b border-[#e6d9b8] bg-[#fdf8ea] px-6 py-2.5"
    >
      <span aria-hidden className="mt-[1px] text-[11px] text-[#8a6320]">◆</span>
      <p className="text-[12px] leading-[1.55] text-[#6d4f1a]">
        <strong className="font-semibold">Demonstration data.</strong>{" "}
        {note ??
          "Programs, evidence and outcomes on this deployment are synthetic. Estimates describe a documented simulation and carry no scientific interpretation."}
      </p>
    </div>
  );
}

export function Callout({
  children, tone = "neutral", title,
}: { children: ReactNode; tone?: "neutral" | "warn" | "accent"; title?: ReactNode }) {
  const tones = {
    neutral: "border-ink-200 bg-ink-50",
    warn: "border-[#f0e0bd] bg-[#fdf8ea]",
    accent: "border-accent-100 bg-accent-50",
  } as const;
  return (
    <div className={clsx("rounded-[var(--radius-card)] border px-4 py-3", tones[tone])}>
      {title ? (
        <div className="mb-1 text-[11.5px] font-semibold text-ink-700">{title}</div>
      ) : null}
      <div className="text-[12.5px] leading-[1.6] text-ink-600">{children}</div>
    </div>
  );
}

export function EmptyState({ title, detail }: { title: string; detail?: string }) {
  return (
    <div className="px-5 py-10 text-center">
      <p className="text-[13px] font-medium text-ink-600">{title}</p>
      {detail ? <p className="mx-auto mt-1 max-w-[46ch] text-[12px] text-ink-500">{detail}</p> : null}
    </div>
  );
}

/* ── Misc ───────────────────────────────────────────────────────────────── */

export function MonoId({ value, href }: { value: string; href?: string }) {
  const content = (
    <span className="font-mono text-[11px] tracking-tight text-ink-500">{value}</span>
  );
  return href ? (
    <Link href={href as never} className="hover:text-accent-700 hover:underline">
      {content}
    </Link>
  ) : (
    content
  );
}

export function KeyValue({ items }: { items: Array<[ReactNode, ReactNode]> }) {
  return (
    <dl className="grid grid-cols-[minmax(0,auto)_minmax(0,1fr)] gap-x-5 gap-y-2">
      {items.map(([k, v], i) => (
        <div key={i} className="contents">
          <dt className="text-[11.5px] text-ink-500">{k}</dt>
          <dd className="text-[12.5px] text-ink-800">{v}</dd>
        </div>
      ))}
    </dl>
  );
}

export function Pill({ children }: { children: ReactNode }) {
  return (
    <span className="rounded-full border border-ink-200 bg-ink-50 px-2 py-[1px] text-[11px] text-ink-600">
      {children}
    </span>
  );
}

export { humanise };
