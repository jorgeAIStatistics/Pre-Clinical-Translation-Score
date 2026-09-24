"use client";

/**
 * The portfolio table.
 *
 * Sorting defaults to program name, not to risk. A table that opens sorted by
 * probability invites the reader to treat the top row as "the one to cut",
 * which is exactly the decision the interval and the confidence grade exist to
 * complicate. Risk is one sortable column among several, and the interval and
 * evidence confidence sit immediately beside it so they cannot be read apart
 * from it.
 */

import Link from "next/link";
import { useMemo, useState } from "react";

import { Badge, ConfidenceBadge, IntervalBar, Td, Th } from "@/components/primitives";
import { humanise, pct, shortDate } from "@/lib/format";
import type { PortfolioRow } from "@/lib/types";

type SortKey = "program_name" | "probability" | "completeness" | "as_of_date" | "indication";

export function PortfolioTable({ rows }: { rows: PortfolioRow[] }) {
  const [sort, setSort] = useState<SortKey>("program_name");
  const [dir, setDir] = useState<1 | -1>(1);
  const [filter, setFilter] = useState("");
  const [area, setArea] = useState<string>("all");
  const [modality, setModality] = useState<string>("all");

  const areas = useMemo(
    () => Array.from(new Set(rows.map((r) => r.therapeutic_area))).sort(),
    [rows],
  );
  const modalities = useMemo(
    () => Array.from(new Set(rows.map((r) => r.modality))).sort(),
    [rows],
  );

  const visible = useMemo(() => {
    const q = filter.trim().toLowerCase();
    const out = rows.filter((r) => {
      if (area !== "all" && r.therapeutic_area !== area) return false;
      if (modality !== "all" && r.modality !== modality) return false;
      if (!q) return true;
      return (
        r.program_name.toLowerCase().includes(q) ||
        r.indication.toLowerCase().includes(q) ||
        (r.target_symbol ?? "").toLowerCase().includes(q) ||
        (r.sponsor ?? "").toLowerCase().includes(q)
      );
    });
    return out.sort((a, b) => {
      const av = a[sort], bv = b[sort];
      if (av === null || av === undefined) return 1;
      if (bv === null || bv === undefined) return -1;
      if (typeof av === "number" && typeof bv === "number") return (av - bv) * dir;
      return String(av).localeCompare(String(bv)) * dir;
    });
  }, [rows, filter, area, modality, sort, dir]);

  const toggle = (key: SortKey) => {
    if (sort === key) setDir((d) => (d === 1 ? -1 : 1));
    else { setSort(key); setDir(key === "probability" ? -1 : 1); }
  };

  const SortHead = ({ k, children, align = "left" }: {
    k: SortKey; children: React.ReactNode; align?: "left" | "right";
  }) => (
    <Th align={align}>
      <button
        onClick={() => toggle(k)}
        className="inline-flex items-center gap-1 hover:text-ink-800"
        aria-sort={sort === k ? (dir === 1 ? "ascending" : "descending") : "none"}
      >
        {children}
        <span aria-hidden className={sort === k ? "text-accent-600" : "text-ink-300"}>
          {sort === k ? (dir === 1 ? "↑" : "↓") : "↕"}
        </span>
      </button>
    </Th>
  );

  return (
    <div>
      <div className="flex flex-wrap items-center gap-2 border-b border-[var(--border-subtle)] px-5 py-3">
        <input
          type="search"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          placeholder="Filter by program, target, indication or sponsor…"
          aria-label="Filter programs"
          className="h-8 w-[300px] rounded-[3px] border border-ink-200 bg-white px-2.5 text-[12.5px] text-ink-800 placeholder:text-ink-400"
        />
        <select
          value={area} onChange={(e) => setArea(e.target.value)} aria-label="Therapeutic area"
          className="h-8 rounded-[3px] border border-ink-200 bg-white px-2 text-[12.5px] text-ink-700"
        >
          <option value="all">All therapeutic areas</option>
          {areas.map((a) => <option key={a} value={a}>{humanise(a)}</option>)}
        </select>
        <select
          value={modality} onChange={(e) => setModality(e.target.value)} aria-label="Modality"
          className="h-8 rounded-[3px] border border-ink-200 bg-white px-2 text-[12.5px] text-ink-700"
        >
          <option value="all">All modalities</option>
          {modalities.map((m) => <option key={m} value={m}>{humanise(m)}</option>)}
        </select>
        <span className="ml-auto tnum font-mono text-[11.5px] text-ink-500">
          {visible.length} of {rows.length}
        </span>
      </div>

      <div className="scroll-thin overflow-x-auto">
        <table className="w-full border-collapse">
          <caption className="sr-only">
            Development programs with translational-risk estimates, uncertainty
            intervals and evidence confidence.
          </caption>
          <thead>
            <tr>
              <SortHead k="program_name">Program</SortHead>
              <Th>Target</Th>
              <SortHead k="indication">Indication</SortHead>
              <Th>Modality</Th>
              <SortHead k="probability" align="right">Risk</SortHead>
              <Th className="w-[170px]">95% interval</Th>
              <Th>Confidence</Th>
              <SortHead k="completeness" align="right">Evidence</SortHead>
              <Th>Leading risk driver</Th>
              <SortHead k="as_of_date">Cutoff</SortHead>
            </tr>
          </thead>
          <tbody>
            {visible.map((r) => (
              <tr key={r.track_id} className="group hover:bg-ink-50">
                <Td className="max-w-[190px]">
                  <Link
                    href={`/programs/${r.track_id}`}
                    className="font-medium text-ink-800 hover:text-accent-700 hover:underline"
                  >
                    {r.program_name}
                  </Link>
                  {r.sponsor ? (
                    <div className="truncate text-[11px] text-ink-400">{r.sponsor}</div>
                  ) : null}
                </Td>
                <Td mono>{r.target_symbol ?? "—"}</Td>
                <Td className="max-w-[210px]">
                  <span className="block truncate" title={r.indication}>{r.indication}</span>
                </Td>
                <Td>
                  <Badge tone="muted">{humanise(r.modality)}</Badge>
                </Td>
                <Td align="right" mono className="font-semibold">
                  {r.probability === null ? "—" : pct(r.probability)}
                </Td>
                <Td>
                  {r.probability === null ? (
                    <span className="text-ink-400">—</span>
                  ) : (
                    <div>
                      <IntervalBar
                        low={r.interval_low ?? 0}
                        high={r.interval_high ?? 1}
                        point={r.probability}
                        height={6}
                      />
                      <div className="mt-1 tnum font-mono text-[10px] text-ink-400">
                        {pct(r.interval_low)}–{pct(r.interval_high)}
                      </div>
                    </div>
                  )}
                </Td>
                <Td><ConfidenceBadge level={r.evidence_confidence} /></Td>
                <Td align="right" mono>
                  {r.completeness === null ? "—" : pct(r.completeness)}
                  <span className="ml-1 text-[10px] text-ink-400">
                    ({r.n_evidence_items})
                  </span>
                </Td>
                <Td className="max-w-[180px]">
                  {r.top_risk_driver ? (
                    <span className="block truncate text-[12px]" title={`${r.top_risk_driver} — ${r.top_risk_driver_domain}`}>
                      {r.top_risk_driver}
                    </span>
                  ) : (
                    <span className="text-ink-400">—</span>
                  )}
                </Td>
                <Td mono className="text-[11.5px] text-ink-500">{shortDate(r.as_of_date)}</Td>
              </tr>
            ))}
          </tbody>
        </table>
        {visible.length === 0 ? (
          <p className="px-5 py-10 text-center text-[13px] text-ink-500">
            No programs match these filters.
          </p>
        ) : null}
      </div>
    </div>
  );
}
