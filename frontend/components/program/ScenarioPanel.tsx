"use client";

/**
 * Scenario analysis.
 *
 * Levers are grouped by evidence domain and split into "additional evidence"
 * and "adverse developments". The adverse group is not an afterthought: a
 * sensitivity tool that only lets a user imagine good news is a tool for
 * building a case, not for testing one, and the split makes both directions
 * equally reachable.
 *
 * The disclaimer sits with the result, not in a tooltip, because the
 * misreading it guards against ("running this study would cut our risk by six
 * points") is the natural one.
 */

import { useState, useTransition } from "react";

import { ScenarioComparisonChart } from "@/components/charts";
import { Badge, Callout, Card, CardHeader, Prose, SectionLabel } from "@/components/primitives";
import { pct, signedPct } from "@/lib/format";
import type { ScenarioComparison, ScenarioLever, ScoreOut } from "@/lib/types";

export function ScenarioPanel({
  trackId, levers, current,
}: { trackId: string; levers: ScenarioLever[]; current: ScoreOut }) {
  const [selected, setSelected] = useState<string[]>([]);
  const [result, setResult] = useState<ScenarioComparison | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pending, startTransition] = useTransition();

  const improvement = levers.filter((l) => l.direction !== "adverse");
  const adverse = levers.filter((l) => l.direction === "adverse");

  const toggle = (key: string) =>
    setSelected((s) => (s.includes(key) ? s.filter((k) => k !== key) : [...s, key]));

  const run = () => {
    setError(null);
    startTransition(async () => {
      try {
        const res = await fetch(
          `${process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000"}/api/v1/programs/${trackId}/scenarios`,
          {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify({ lever_keys: selected, overrides: {} }),
          },
        );
        if (!res.ok) {
          const body = await res.json().catch(() => ({}));
          throw new Error(body.detail ?? `Request failed (${res.status})`);
        }
        setResult(await res.json());
      } catch (e) {
        setError(e instanceof Error ? e.message : "Scenario failed.");
        setResult(null);
      }
    });
  };

  const reset = () => { setSelected([]); setResult(null); setError(null); };

  const LeverGroup = ({ items, label }: { items: ScenarioLever[]; label: string }) => (
    <div>
      <SectionLabel>{label}</SectionLabel>
      <ul className="mt-2 space-y-1.5">
        {items.map((lv) => {
          const on = selected.includes(lv.key);
          return (
            <li key={lv.key}>
              <label
                className={`flex cursor-pointer gap-2.5 rounded-[3px] border px-2.5 py-2 transition-colors ${
                  on
                    ? "border-accent-300 bg-accent-50"
                    : "border-transparent hover:border-ink-200 hover:bg-ink-50"
                }`}
              >
                <input
                  type="checkbox"
                  checked={on}
                  onChange={() => toggle(lv.key)}
                  className="mt-[3px] h-3.5 w-3.5 shrink-0 accent-[var(--color-accent-600)]"
                />
                <span className="min-w-0">
                  <span className="block text-[12.5px] font-medium text-ink-800">{lv.label}</span>
                  <span className="mt-0.5 block text-[11.5px] leading-[1.5] text-ink-500">
                    {lv.description}
                  </span>
                  <span className="mt-1 block text-[10.5px] text-ink-400">{lv.domain_label}</span>
                </span>
              </label>
            </li>
          );
        })}
      </ul>
    </div>
  );

  return (
    <Card>
      <CardHeader
        title="Scenario analysis"
        subtitle="Recompute the estimate under a hypothetical evidence state, to see how much it depends on evidence the program does not yet have."
        action={
          selected.length ? (
            <button
              onClick={reset}
              className="text-[12px] text-ink-500 hover:text-ink-800 hover:underline"
            >
              Clear
            </button>
          ) : null
        }
      />

      <div className="grid gap-6 p-5 lg:grid-cols-[minmax(0,320px)_minmax(0,1fr)]">
        <div className="space-y-5">
          <LeverGroup items={improvement} label="If additional evidence arrives" />
          <LeverGroup items={adverse} label="If the evidence turns against it" />
          <button
            onClick={run}
            disabled={!selected.length || pending}
            className="w-full rounded-[3px] bg-accent-600 px-3 py-2 text-[12.5px] font-medium text-white transition-colors hover:bg-accent-700 disabled:cursor-not-allowed disabled:bg-ink-300"
          >
            {pending
              ? "Recomputing…"
              : `Recompute with ${selected.length || "no"} change${selected.length === 1 ? "" : "s"}`}
          </button>
        </div>

        <div>
          {error ? (
            <Callout tone="warn" title="Scenario failed">{error}</Callout>
          ) : result ? (
            <div className="space-y-5">
              <div className="flex flex-wrap items-baseline gap-x-8 gap-y-2">
                <div>
                  <SectionLabel>Current</SectionLabel>
                  <div className="tnum mt-1 font-mono text-[22px] text-ink-800">
                    {pct(result.current.probability)}
                  </div>
                  <div className="tnum font-mono text-[11px] text-ink-400">
                    {pct(result.current.interval_low)}–{pct(result.current.interval_high)}
                  </div>
                </div>
                <span aria-hidden className="text-[18px] text-ink-300">→</span>
                <div>
                  <SectionLabel>Under this scenario</SectionLabel>
                  <div className="tnum mt-1 font-mono text-[22px] text-ink-800">
                    {pct(result.scenario.probability)}
                  </div>
                  <div className="tnum font-mono text-[11px] text-ink-400">
                    {pct(result.scenario.interval_low)}–{pct(result.scenario.interval_high)}
                  </div>
                </div>
                <div className="ml-auto text-right">
                  <SectionLabel>Change</SectionLabel>
                  <div
                    className="tnum mt-1 font-mono text-[22px]"
                    style={{
                      color:
                        result.delta_probability < 0
                          ? "var(--color-risk-1)"
                          : "var(--color-risk-5)",
                    }}
                  >
                    {signedPct(result.delta_probability)}
                  </div>
                  <div className="tnum font-mono text-[11px] text-ink-400">
                    interval width {signedPct(result.delta_interval_width)}
                  </div>
                </div>
              </div>

              <ScenarioComparisonChart current={result.current} scenario={result.scenario} />

              <div>
                <SectionLabel>Assumptions changed</SectionLabel>
                <ul className="mt-2 space-y-1">
                  {result.changes.map((c) => (
                    <li key={c.feature} className="flex flex-wrap items-baseline gap-2 text-[12px]">
                      <span className="text-ink-700">{c.label}</span>
                      <span className="tnum font-mono text-[11.5px] text-ink-500">
                        {c.was_missing ? "not reported" : c.from_value} → {c.to_value}
                      </span>
                      <Badge tone="muted">{c.domain_label}</Badge>
                    </li>
                  ))}
                </ul>
              </div>

              <Callout tone="warn" title="How to read this">
                {result.disclaimer}
              </Callout>
            </div>
          ) : (
            <div className="flex h-full min-h-[220px] flex-col justify-center">
              <Prose>
                Select one or more hypothetical changes and recompute. The result
                shows what this model would return if that evidence existed —
                it is a sensitivity analysis of the estimate, not a forecast of
                what the experiment would find.
              </Prose>
              <p className="mt-3 text-[12px] text-ink-500">
                Current estimate:{" "}
                <span className="tnum font-mono text-ink-800">{pct(current.probability)}</span>{" "}
                <span className="tnum font-mono text-ink-400">
                  ({pct(current.interval_low)}–{pct(current.interval_high)})
                </span>
              </p>
            </div>
          )}
        </div>
      </div>
    </Card>
  );
}
