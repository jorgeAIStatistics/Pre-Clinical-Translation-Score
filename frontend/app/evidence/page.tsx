import Link from "next/link";

import {
  Badge, Card, CardHeader, Prose, SectionLabel, Stat, Td, Th,
} from "@/components/primitives";
import { api } from "@/lib/api";
import { humanise } from "@/lib/format";

export const dynamic = "force-dynamic";

/**
 * Evidence Explorer.
 *
 * Entry point for the evidence base as a whole rather than one program's slice
 * of it: the feature schema (what the platform is able to see), the source
 * vocabulary, and a route into any program's evidence.
 */
export default async function EvidencePage() {
  const [features, programs, endpoints] = await Promise.all([
    api.featureSchema(),
    api.programs({ limit: 40 }),
    api.endpoints(),
  ]);

  const byDomain = features.reduce<Record<string, typeof features>>((acc, f) => {
    (acc[f.domain_label] ??= []).push(f);
    return acc;
  }, {});

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-[20px] font-semibold tracking-[-0.01em] text-ink-900">
          Evidence explorer
        </h1>
        <Prose className="mt-2">
          The feature schema is the contract between evidence and model: a fact
          that has no feature here cannot influence an estimate, and every
          feature states exactly which evidence produces it.
        </Prose>
      </header>

      <div className="flex gap-8">
        <Stat label="Declared features" value={features.length} />
        <Stat label="Evidence domains" value={Object.keys(byDomain).length} />
        <Stat
          label="Scenario-adjustable"
          value={features.filter((f) => f.scenario_adjustable).length}
          hint="Can be varied in scenario analysis"
        />
        <Stat label="Supported endpoints" value={endpoints.endpoints.length} />
      </div>

      <Card>
        <CardHeader
          title="Supervised endpoints"
          subtitle={endpoints.note}
        />
        <div className="scroll-thin overflow-x-auto">
          <table className="w-full border-collapse">
            <thead>
              <tr>
                <Th>Endpoint</Th><Th>Estimand</Th>
                <Th align="right">Min. follow-up</Th><Th>Primary</Th>
              </tr>
            </thead>
            <tbody>
              {endpoints.endpoints.map((e: any) => (
                <tr key={e.name} className="align-top hover:bg-ink-50">
                  <Td className="whitespace-nowrap font-medium">{e.display_name}</Td>
                  <Td className="max-w-[72ch] text-[12px] leading-[1.6]">{e.estimand}</Td>
                  <Td align="right" mono>{e.min_followup_years}y</Td>
                  <Td>{e.is_primary ? <Badge tone="accent">Primary</Badge> : null}</Td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Card>
        <CardHeader
          title="Feature schema"
          subtitle="Grouped by evidence domain. The derivation column states the aggregation rule that turns evidence rows into each value."
        />
        <div className="divide-y divide-[var(--border-subtle)]">
          {Object.entries(byDomain).map(([domain, items]) => (
            <div key={domain} className="p-5">
              <div className="flex items-baseline gap-3">
                <SectionLabel>{domain}</SectionLabel>
                <span className="tnum font-mono text-[11px] text-ink-400">
                  {items.length} feature{items.length === 1 ? "" : "s"}
                </span>
              </div>
              <div className="scroll-thin mt-3 overflow-x-auto">
                <table className="w-full border-collapse">
                  <thead>
                    <tr>
                      <Th>Feature</Th><Th>Description</Th><Th>Derivation</Th>
                      <Th>Prior direction</Th><Th>Missing policy</Th>
                    </tr>
                  </thead>
                  <tbody>
                    {items.map((f) => (
                      <tr key={f.name} className="align-top hover:bg-ink-50">
                        <Td className="whitespace-nowrap">
                          <span className="font-medium text-ink-800">{f.label}</span>
                          <span className="mt-0.5 block font-mono text-[10.5px] text-ink-400">
                            {f.name}
                          </span>
                          {f.scenario_adjustable ? (
                            <span className="mt-1 inline-block">
                              <Badge tone="muted">adjustable</Badge>
                            </span>
                          ) : null}
                        </Td>
                        <Td className="max-w-[44ch] text-[12px] leading-[1.55]">{f.description}</Td>
                        <Td className="max-w-[40ch] font-mono text-[10.5px] leading-[1.5] text-ink-500">
                          {f.derivation}
                        </Td>
                        <Td className="whitespace-nowrap text-[11.5px]">
                          {f.expected_direction === "uncertain" ? (
                            <span className="text-ink-400">Estimated, not assumed</span>
                          ) : (
                            <span>
                              {f.expected_direction === "increases_risk" ? "↑ risk" : "↓ risk"}
                            </span>
                          )}
                        </Td>
                        <Td className="text-[11.5px] text-ink-500">{humanise(f.missing_policy)}</Td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ))}
        </div>
      </Card>

      <Card>
        <CardHeader
          title="Browse program evidence"
          subtitle="Open a program to see its evidence matrix, individual studies and their sources."
        />
        <div className="scroll-thin max-h-[420px] overflow-auto">
          <table className="w-full border-collapse">
            <thead className="sticky top-0 bg-[var(--surface-raised)]">
              <tr>
                <Th>Program</Th><Th>Target</Th><Th>Indication</Th>
                <Th align="right">Evidence items</Th><Th>Outcome</Th>
              </tr>
            </thead>
            <tbody>
              {programs.map((p) => (
                <tr key={p.id} className="hover:bg-ink-50">
                  <Td>
                    <Link
                      href={`/programs/${p.id}`}
                      className="font-medium text-ink-800 hover:text-accent-700 hover:underline"
                    >
                      {p.program_name}
                    </Link>
                  </Td>
                  <Td mono>{p.target?.symbol ?? "—"}</Td>
                  <Td className="max-w-[260px]">
                    <span className="block truncate">{p.indication.name}</span>
                  </Td>
                  <Td align="right" mono>{p.n_evidence_items}</Td>
                  <Td className="text-[11.5px] text-ink-500">{humanise(p.outcome_label)}</Td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}
