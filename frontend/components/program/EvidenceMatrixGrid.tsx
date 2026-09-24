/**
 * Evidence matrix: evidence class × biological system.
 *
 * The point of this view is negative space. A cell with no study is drawn with
 * a hatch rather than left blank or filled with a zero, because "no study of
 * this kind exists" and "a study of this kind found nothing" are different
 * facts and the interface must not blur them.
 */

import { Card, CardHeader } from "@/components/primitives";
import { humanise } from "@/lib/format";
import type { EvidenceMatrix } from "@/lib/types";

const SHORT_SYSTEM: Record<string, string> = {
  cell_free_biochemical: "Cell-free",
  immortalised_cell_line: "Cell line",
  primary_cell: "Primary cell",
  organoid: "Organoid",
  ex_vivo_tissue: "Ex vivo",
  in_vivo_whole_animal: "In vivo",
  human_subject: "Human subject",
  human_population: "Population",
};

export function EvidenceMatrixGrid({ matrix }: { matrix: EvidenceMatrix }) {
  // Aggregate across endpoints: the grid is class × system, and endpoint
  // detail belongs in the tooltip rather than as duplicate cells.
  const byCell = new Map<string, { total: number; sup: number; ref: number; endpoints: Set<string> }>();
  for (const c of matrix.cells) {
    const key = `${c.evidence_class}|${c.system}`;
    const e = byCell.get(key) ?? { total: 0, sup: 0, ref: 0, endpoints: new Set<string>() };
    e.total += c.n_total;
    e.sup += c.n_supporting;
    e.ref += c.n_refuting;
    e.endpoints.add(c.endpoint);
    byCell.set(key, e);
  }

  // Only show rows that have any evidence, plus the human-derived rows, which
  // are the ones whose emptiness is most informative.
  const HUMAN_ROWS = new Set([
    "organoid", "primary_human_cell", "human_ex_vivo",
    "human_tissue_molecular", "human_genetic",
  ]);
  const rows = matrix.rows.filter(
    (r) => HUMAN_ROWS.has(r) || matrix.cells.some((c) => c.evidence_class === r),
  );

  const maxTotal = Math.max(...Array.from(byCell.values(), (v) => v.total), 1);

  return (
    <Card>
      <CardHeader
        title="Evidence matrix"
        subtitle={matrix.note}
        action={
          <span className="tnum font-mono text-[11px] text-ink-400">
            {matrix.n_evidence} items · {Math.round(matrix.coverage * 100)}% cell coverage
          </span>
        }
      />
      <div className="scroll-thin overflow-x-auto p-5">
        <table className="border-collapse">
          <caption className="sr-only">
            Evidence count by evidence class and biological system. Hatched cells
            indicate no study on record.
          </caption>
          <thead>
            <tr>
              <th className="sticky left-0 z-10 bg-[var(--surface-raised)] pr-3" />
              {matrix.columns.map((c) => (
                <th
                  key={c}
                  scope="col"
                  className="px-1 pb-2 align-bottom text-[10px] font-medium text-ink-500"
                >
                  <span className="block w-[62px] leading-[1.25]">
                    {SHORT_SYSTEM[c] ?? humanise(c)}
                  </span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r}>
                <th
                  scope="row"
                  className="sticky left-0 z-10 whitespace-nowrap bg-[var(--surface-raised)] pr-3 text-right text-[11.5px] font-normal text-ink-600"
                >
                  {humanise(r)}
                </th>
                {matrix.columns.map((c) => {
                  const cell = byCell.get(`${r}|${c}`);
                  if (!cell) {
                    return (
                      <td key={c} className="p-[2px]">
                        <div
                          className="hatch h-[30px] w-[62px] rounded-[2px] border border-ink-100"
                          title={`${humanise(r)} / ${SHORT_SYSTEM[c] ?? humanise(c)} — no study on record`}
                        />
                      </td>
                    );
                  }
                  const intensity = 0.18 + 0.72 * (cell.total / maxTotal);
                  const conflicted = cell.sup > 0 && cell.ref > 0;
                  return (
                    <td key={c} className="p-[2px]">
                      <div
                        className="relative flex h-[30px] w-[62px] items-center justify-center rounded-[2px] border"
                        style={{
                          background: `color-mix(in srgb, var(--color-accent-500) ${intensity * 100}%, white)`,
                          borderColor: conflicted ? "var(--color-refute)" : "var(--border-subtle)",
                          borderWidth: conflicted ? 1.5 : 1,
                        }}
                        title={[
                          `${humanise(r)} / ${SHORT_SYSTEM[c] ?? humanise(c)}`,
                          `${cell.total} item(s): ${cell.sup} supporting, ${cell.ref} refuting`,
                          `Endpoints: ${Array.from(cell.endpoints).map(humanise).join(", ")}`,
                          conflicted ? "Contains conflicting results" : "",
                        ].filter(Boolean).join("\n")}
                      >
                        <span className="tnum font-mono text-[11.5px] text-ink-800">
                          {cell.total}
                        </span>
                        {conflicted ? (
                          <span
                            aria-hidden
                            className="absolute right-[3px] top-[2px] text-[8px] text-[var(--color-refute)]"
                          >
                            ▼
                          </span>
                        ) : null}
                      </div>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>

        <div className="mt-4 flex flex-wrap items-center gap-x-5 gap-y-2 text-[11px] text-ink-500">
          <span className="inline-flex items-center gap-1.5">
            <span className="hatch inline-block h-3 w-5 rounded-[1px] border border-ink-100" />
            No study on record
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span
              className="inline-block h-3 w-5 rounded-[1px] border border-[var(--border-subtle)]"
              style={{ background: "color-mix(in srgb, var(--color-accent-500) 30%, white)" }}
            />
            Shade indicates count
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span
              className="inline-block h-3 w-5 rounded-[1px]"
              style={{ border: "1.5px solid var(--color-refute)" }}
            />
            Contains conflicting results
          </span>
        </div>
      </div>
    </Card>
  );
}
