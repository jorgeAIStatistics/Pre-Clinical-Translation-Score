"use client";

import Link from "next/link";
import { useMemo, useState } from "react";

import { Badge, DirectionBadge, Td, Th } from "@/components/primitives";
import { humanise, num, shortDate } from "@/lib/format";
import type { EvidenceOut } from "@/lib/types";

export function EvidenceTable({ items, cutoff }: { items: EvidenceOut[]; cutoff: string | null }) {
  const [cls, setCls] = useState("all");
  const [dir, setDir] = useState("all");

  const classes = useMemo(
    () => Array.from(new Set(items.map((i) => i.evidence_class))).sort(),
    [items],
  );

  const visible = items.filter(
    (i) => (cls === "all" || i.evidence_class === cls) && (dir === "all" || i.direction === dir),
  );

  return (
    <div>
      <div className="flex flex-wrap items-center gap-2 border-b border-[var(--border-subtle)] px-5 py-3">
        <select
          value={cls} onChange={(e) => setCls(e.target.value)} aria-label="Evidence class"
          className="h-8 rounded-[3px] border border-ink-200 bg-white px-2 text-[12.5px] text-ink-700"
        >
          <option value="all">All evidence classes</option>
          {classes.map((c) => <option key={c} value={c}>{humanise(c)}</option>)}
        </select>
        <select
          value={dir} onChange={(e) => setDir(e.target.value)} aria-label="Direction"
          className="h-8 rounded-[3px] border border-ink-200 bg-white px-2 text-[12.5px] text-ink-700"
        >
          <option value="all">Any direction</option>
          <option value="supports">Supports</option>
          <option value="refutes">Refutes</option>
          <option value="neutral">Neutral</option>
        </select>
        <span className="ml-auto tnum font-mono text-[11.5px] text-ink-500">
          {visible.length} of {items.length}
          {cutoff ? ` · available at ${shortDate(cutoff)}` : ""}
        </span>
      </div>

      <div className="scroll-thin max-h-[560px] overflow-auto">
        <table className="w-full border-collapse">
          <caption className="sr-only">Evidence items contributing to this estimate</caption>
          <thead className="sticky top-0 z-10 bg-[var(--surface-raised)]">
            <tr>
              <Th>Evidence</Th>
              <Th>Class</Th>
              <Th>System</Th>
              <Th>Direction</Th>
              <Th align="right">Effect</Th>
              <Th align="right">n/arm</Th>
              <Th>Quality</Th>
              <Th>Available</Th>
              <Th>Source</Th>
            </tr>
          </thead>
          <tbody>
            {visible.map((e) => (
              <tr key={e.id} className="align-top hover:bg-ink-50">
                <Td className="max-w-[300px]">
                  <span className="block text-[12.5px] text-ink-800">
                    {e.title ?? humanise(e.endpoint_category)}
                  </span>
                  {e.model_type !== "not_applicable" ? (
                    <span className="mt-0.5 block text-[11px] text-ink-500">
                      {humanise(e.model_type)}
                      {e.model_subtype !== "not_applicable" ? ` · ${humanise(e.model_subtype)}` : ""}
                      {e.immune_competence !== "not_applicable"
                        ? ` · ${humanise(e.immune_competence)}`
                        : ""}
                    </span>
                  ) : null}
                </Td>
                <Td><Badge tone="muted">{humanise(e.evidence_class)}</Badge></Td>
                <Td className="text-[11.5px]">
                  {humanise(e.species)}
                  <span className="block text-[11px] text-ink-400">
                    {humanise(e.biological_system)}
                  </span>
                </Td>
                <Td><DirectionBadge direction={e.direction} /></Td>
                <Td align="right" mono>
                  {e.effect_value === null ? "—" : num(e.effect_value, 1)}
                  <span className="block text-[10px] text-ink-400">
                    {e.effect_metric ? humanise(e.effect_metric) : ""}
                  </span>
                  {e.standardised_effect !== null ? (
                    <span className="block text-[10px] text-ink-400" title="Mapped onto the common standardised scale">
                      z {num(e.standardised_effect, 2)}
                    </span>
                  ) : null}
                </Td>
                <Td align="right" mono>{e.n_per_arm ?? "—"}</Td>
                <Td className="text-[11px] text-ink-500">
                  <span className="block">{humanise(e.randomization)}</span>
                  <span className="block">{humanise(e.blinding)}</span>
                  <span className="block">{humanise(e.replication_scope)}</span>
                </Td>
                <Td mono className="text-[11.5px]">{shortDate(e.available_at)}</Td>
                <Td className="max-w-[170px] text-[11px]">
                  {e.source ? (
                    <Link
                      href={`/provenance?evidence=${e.id}`}
                      className="text-accent-700 hover:underline"
                      title={e.source.title ?? ""}
                    >
                      {e.source.pmid
                        ? `PMID ${e.source.pmid}`
                        : e.source.doi
                          ? e.source.doi
                          : humanise(e.source.source_type)}
                    </Link>
                  ) : (
                    <span className="text-ink-400">—</span>
                  )}
                  <span className="mt-0.5 block text-ink-400">
                    {humanise(e.provenance_method)}
                  </span>
                </Td>
              </tr>
            ))}
          </tbody>
        </table>
        {visible.length === 0 ? (
          <p className="px-5 py-8 text-center text-[13px] text-ink-500">
            No evidence matches these filters.
          </p>
        ) : null}
      </div>
    </div>
  );
}
