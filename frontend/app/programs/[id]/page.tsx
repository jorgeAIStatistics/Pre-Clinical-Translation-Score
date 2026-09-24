import Link from "next/link";
import { notFound } from "next/navigation";

import {
  DomainContributions, DriverWaterfall, UncertaintyComposition,
} from "@/components/charts";
import { EvidenceMatrixGrid } from "@/components/program/EvidenceMatrixGrid";
import { EvidenceTable } from "@/components/program/EvidenceTable";
import { ScenarioPanel } from "@/components/program/ScenarioPanel";
import {
  Badge, Callout, Card, CardHeader, ConfidenceBadge, EmptyState, IntervalBar,
  KeyValue, MonoId, Pill, Prose, RiskBandLabel, SectionLabel, Stat, Td, Th,
} from "@/components/primitives";
import { api } from "@/lib/api";
import { humanise, num, pct, shortDate } from "@/lib/format";

export const dynamic = "force-dynamic";

export default async function ProgramPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;

  const program = await api.program(id).catch(() => null);
  if (!program) notFound();

  // Fetched in parallel: the page is useless without the score, and the rest
  // are independent panels.
  const [score, evidence, matrix, comparators, levers, pkpd, targetAssessment, safety] =
    await Promise.all([
      api.score(id),
      api.evidence(id),
      api.evidenceMatrix(id),
      api.comparators(id, 8),
      api.levers(),
      api.pkpd(id),
      api.targetAssessment(id),
      api.safety(id),
    ]);

  const explanation = score.explanation;
  const cutoff = score.as_of_date;

  return (
    <div className="space-y-6">
      {/* ── Identity ───────────────────────────────────────────────────── */}
      <header>
        <div className="flex items-center gap-2 text-[12px] text-ink-500">
          <Link href="/" className="hover:text-accent-700 hover:underline">Portfolio</Link>
          <span aria-hidden>/</span>
          <span>{program.indication.name}</span>
        </div>

        <div className="mt-2 flex flex-wrap items-start justify-between gap-6">
          <div className="min-w-0">
            <h1 className="text-[22px] font-semibold tracking-[-0.015em] text-ink-900">
              {program.program_name}
            </h1>
            <p className="mt-1.5 max-w-[76ch] text-[13.5px] leading-[1.6] text-ink-600">
              {program.mechanism_of_action ?? "Mechanism not recorded."}
            </p>
            <div className="mt-3 flex flex-wrap items-center gap-2">
              {program.target ? (
                <Badge tone="accent">
                  <span className="font-mono">{program.target.symbol}</span>
                  <span className="text-accent-600/70">· {humanise(program.target.target_class)}</span>
                </Badge>
              ) : null}
              <Badge tone="neutral">{humanise(program.modality)}</Badge>
              <Badge tone="muted">{humanise(program.disease_stage)}</Badge>
              <Badge tone="muted">{humanise(program.line_of_therapy)}</Badge>
              {program.biomarker_selection !== "all_comers" ? (
                <Badge tone="muted">{humanise(program.biomarker_selection)}</Badge>
              ) : null}
              {program.combination_partner ? (
                <Badge tone="muted">+ {program.combination_partner}</Badge>
              ) : null}
            </div>
          </div>

          <dl className="flex shrink-0 gap-7">
            <Stat label="Sponsor" value={program.sponsor?.name ?? "—"} mono={false} />
            <Stat label="IND / CTA" value={shortDate(program.ind_or_cta_on)} />
            <Stat label="Evidence items" value={program.n_evidence_items} />
          </dl>
        </div>

        {program.all_tracks.length > 1 ? (
          <div className="mt-4 flex flex-wrap items-center gap-2">
            <SectionLabel>Also developed in</SectionLabel>
            {program.all_tracks.filter((t) => !t.is_current).map((t) => (
              <Link key={t.track_id} href={`/programs/${t.track_id}`}>
                <Pill>{t.indication}</Pill>
              </Link>
            ))}
          </div>
        ) : null}
      </header>

      {/* ── The estimate ───────────────────────────────────────────────── */}
      <Card>
        <CardHeader
          title="Translational risk estimate"
          subtitle={score.estimand}
          action={
            <div className="text-right">
              <div className="font-mono text-[11px] text-ink-400">{score.model_version}</div>
              <div className="font-mono text-[11px] text-ink-400">
                as of {shortDate(cutoff)}
              </div>
            </div>
          }
        />

        <div className="grid gap-8 p-5 lg:grid-cols-[minmax(0,380px)_minmax(0,1fr)]">
          <div>
            <SectionLabel>Estimated probability of Phase II efficacy failure</SectionLabel>
            <div className="mt-2 flex items-baseline gap-3">
              <span className="tnum font-mono text-[44px] leading-none text-ink-900">
                {pct(score.probability)}
              </span>
              <RiskBandLabel p={score.probability} />
            </div>

            <div className="mt-4">
              <IntervalBar
                low={score.interval_low}
                high={score.interval_high}
                point={score.probability}
                height={10}
                showScale
              />
              <p className="mt-2 text-[12.5px] text-ink-600">
                <span className="tnum font-mono">
                  {Math.round(score.interval_level * 100)}% uncertainty interval:{" "}
                  {pct(score.interval_low)} – {pct(score.interval_high)}
                </span>
              </p>
            </div>

            <div className="mt-5 flex flex-wrap items-center gap-3">
              <ConfidenceBadge level={score.evidence_confidence} />
              <span className="text-[12px] text-ink-500">
                evidence completeness{" "}
                <span className="tnum font-mono text-ink-700">
                  {pct(score.completeness.overall)}
                </span>
              </span>
            </div>

            <ul className="mt-3 space-y-1.5">
              {score.confidence_reasons.map((r, i) => (
                <li key={i} className="flex gap-2 text-[11.5px] leading-[1.55] text-ink-500">
                  <span aria-hidden className="mt-[5px] h-1 w-1 shrink-0 rounded-full bg-ink-300" />
                  {r}
                </li>
              ))}
            </ul>

            {score.completeness.critical_gaps.length ? (
              <div className="mt-4">
                <Callout tone="warn" title="Critical evidence gaps">
                  This estimate rests on an evidence base missing most of:{" "}
                  {score.completeness.critical_gaps.map(humanise).join(", ")}.
                </Callout>
              </div>
            ) : null}
          </div>

          <div className="space-y-6">
            <div>
              <SectionLabel>Where the uncertainty comes from</SectionLabel>
              <div className="mt-3">
                <UncertaintyComposition
                  shares={score.uncertainty.variance_share}
                  sds={score.uncertainty.components_sd_logit}
                />
              </div>
            </div>

            {score.uncertainty.caveats.length ? (
              <ul className="space-y-1.5">
                {score.uncertainty.caveats.map((c, i) => (
                  <li key={i} className="flex gap-2 text-[11.5px] leading-[1.55] text-ink-500">
                    <span aria-hidden className="mt-[2px] text-[10px] text-ink-400">△</span>
                    {c}
                  </li>
                ))}
              </ul>
            ) : null}

            {score.specification_sensitivity?.available ? (
              <div>
                <SectionLabel>Model-specification sensitivity</SectionLabel>
                <p className="mt-1.5 text-[12px] text-ink-600">
                  Scoring the same evidence with other candidate model families moves
                  the estimate by at most{" "}
                  <span className="tnum font-mono">
                    {pct(score.specification_sensitivity.max_absolute_difference)}
                  </span>
                  . Reported separately from the interval, never folded into it.
                </p>
              </div>
            ) : null}
          </div>
        </div>

        {score.model_usefulness?.headline ? (
          <div className="border-t border-[var(--border-subtle)] px-5 py-4">
            <Callout
              tone={score.model_usefulness.verdict === "beats_base_rate" ? "accent" : "warn"}
              title="How much weight this estimate can carry"
            >
              <p>{score.model_usefulness.headline}</p>
              <p className="mt-2">
                <Link href="/methods" className="text-accent-700 underline underline-offset-2">
                  Full validation report
                </Link>
              </p>
            </Callout>
          </div>
        ) : null}
      </Card>

      {/* ── Drivers ────────────────────────────────────────────────────── */}
      {explanation ? (
        <div className="grid gap-6 lg:grid-cols-[minmax(0,1.25fr)_minmax(0,1fr)]">
          <Card>
            <CardHeader
              title="Risk drivers"
              subtitle={explanation.statement}
            />
            <div className="p-5">
              <div className="mb-4 flex flex-wrap items-baseline gap-x-6 gap-y-1 text-[12px] text-ink-600">
                <span>
                  Cohort-average program:{" "}
                  <span className="tnum font-mono text-ink-800">
                    {pct(explanation.baseline_probability)}
                  </span>
                </span>
                <span>
                  This program:{" "}
                  <span className="tnum font-mono text-ink-800">{pct(explanation.probability)}</span>
                </span>
                <span className="text-ink-400">
                  reconstruction error {explanation.reconstruction_error.toExponential(1)}
                </span>
              </div>

              <DriverWaterfall
                drivers={[...explanation.increasing_risk, ...explanation.decreasing_risk]
                  .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))
                  .slice(0, 12)}
              />

              <div className="mt-6 grid gap-6 sm:grid-cols-2">
                <div>
                  <SectionLabel>Increasing risk</SectionLabel>
                  <ul className="mt-2 space-y-2.5">
                    {explanation.increasing_risk.slice(0, 5).map((d) => (
                      <li key={d.feature}>
                        <div className="text-[12.5px] font-medium text-ink-800">{d.label}</div>
                        <div className="mt-0.5 text-[11.5px] leading-[1.5] text-ink-500">
                          {d.is_missing
                            ? `Not reported (${humanise(d.missing_reason)}). The model prices the absence itself.`
                            : `${d.value !== null ? `Value ${typeof d.value === "number" ? num(d.value, 2) : humanise(String(d.value))}` : ""}${
                                d.z_score !== null ? ` · ${num(Math.abs(d.z_score), 1)} SD ${d.z_score > 0 ? "above" : "below"} cohort mean` : ""
                              }`}
                        </div>
                        <div className="mt-1 flex items-center gap-2">
                          <Badge tone="muted">{d.domain_label}</Badge>
                          <span className="tnum font-mono text-[11px] text-ink-500">
                            {d.probability_delta >= 0 ? "+" : ""}
                            {(d.probability_delta * 100).toFixed(1)}%
                          </span>
                          {d.evidence_ids.length ? (
                            <span className="text-[10.5px] text-ink-400">
                              {d.evidence_ids.length} evidence item
                              {d.evidence_ids.length === 1 ? "" : "s"}
                            </span>
                          ) : null}
                        </div>
                      </li>
                    ))}
                  </ul>
                </div>
                <div>
                  <SectionLabel>Decreasing risk</SectionLabel>
                  <ul className="mt-2 space-y-2.5">
                    {explanation.decreasing_risk.slice(0, 5).map((d) => (
                      <li key={d.feature}>
                        <div className="text-[12.5px] font-medium text-ink-800">{d.label}</div>
                        <div className="mt-0.5 text-[11.5px] leading-[1.5] text-ink-500">
                          {d.is_missing
                            ? `Not reported (${humanise(d.missing_reason)}).`
                            : `${d.value !== null ? `Value ${typeof d.value === "number" ? num(d.value, 2) : humanise(String(d.value))}` : ""}${
                                d.z_score !== null ? ` · ${num(Math.abs(d.z_score), 1)} SD ${d.z_score > 0 ? "above" : "below"} cohort mean` : ""
                              }`}
                        </div>
                        <div className="mt-1 flex items-center gap-2">
                          <Badge tone="muted">{d.domain_label}</Badge>
                          <span className="tnum font-mono text-[11px] text-ink-500">
                            {(d.probability_delta * 100).toFixed(1)}%
                          </span>
                        </div>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
          </Card>

          <Card>
            <CardHeader
              title="Translational scorecard"
              subtitle="Net contribution of each evidence domain to the log-odds, and how completely each domain is documented."
            />
            <div className="p-5">
              <DomainContributions domains={explanation.by_domain} />

              <div className="mt-6">
                <SectionLabel>Domain completeness</SectionLabel>
                <ul className="mt-2.5 space-y-1.5">
                  {Object.entries(score.completeness.by_domain)
                    .sort((a, b) => a[1] - b[1])
                    .map(([dom, v]) => (
                      <li key={dom} className="grid grid-cols-[minmax(0,1fr)_84px_40px] items-center gap-3">
                        <span className="truncate text-[12px] text-ink-700">{humanise(dom)}</span>
                        <span className="h-1.5 overflow-hidden rounded-[1px] bg-ink-100">
                          <span
                            className="block h-full rounded-[1px]"
                            style={{
                              width: `${v * 100}%`,
                              background: v < 0.5 ? "var(--color-risk-4)" : "var(--color-accent-500)",
                            }}
                          />
                        </span>
                        <span className="tnum text-right font-mono text-[11px] text-ink-500">
                          {pct(v)}
                        </span>
                      </li>
                    ))}
                </ul>
              </div>
            </div>
          </Card>
        </div>
      ) : (
        <Card>
          <CardHeader title="Risk drivers" />
          <EmptyState
            title="No driver breakdown available"
            detail={score.explanation_unavailable_reason ?? undefined}
          />
        </Card>
      )}

      {/* ── PK/PD and target biology ───────────────────────────────────── */}
      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader
            title="PK/PD translational bridge"
            subtitle={pkpd?.note}
          />
          <div className="p-5">
            {pkpd?.current ? (
              <KeyValue
                items={[
                  ["Exposure margin",
                    pkpd.current.exposure_margin_ratio
                      ? <span className="tnum font-mono">{num(pkpd.current.exposure_margin_ratio, 2)}×</span>
                      : "Not reported"],
                  ["Therapeutic index",
                    pkpd.current.therapeutic_index_ratio
                      ? <span className="tnum font-mono">{num(pkpd.current.therapeutic_index_ratio, 2)}×</span>
                      : "Not reported"],
                  ["Bridge quality", humanise(pkpd.current.exposure_bridge_quality)],
                  ["Target engagement", humanise(pkpd.current.target_engagement_evidence)],
                  ["Exposure–response",
                    pkpd.current.exposure_response_established === null
                      ? "Not reported"
                      : pkpd.current.exposure_response_established ? "Established" : "Not established"],
                  ["Species in PK scaling", pkpd.current.pk_scaling_species_count ?? "Not reported"],
                  ["Assembled", shortDate(pkpd.current.available_at)],
                ]}
              />
            ) : (
              <EmptyState
                title="No PK/PD bridge on record at the cutoff"
                detail="The exposure features are imputed and the prediction interval widens to reflect that."
              />
            )}
          </div>
        </Card>

        <Card>
          <CardHeader
            title="Target validation"
            subtitle={targetAssessment?.note}
          />
          <div className="p-5">
            {targetAssessment?.current ? (
              <KeyValue
                items={[
                  ["Human genetic support", humanise(targetAssessment.current.genetic_evidence_tier)],
                  ["Association score",
                    targetAssessment.current.association_score !== null
                      ? <span className="tnum font-mono">{num(targetAssessment.current.association_score, 2)}</span>
                      : "Not reported"],
                  ["Causal support", humanise(targetAssessment.current.causal_support)],
                  ["Pathway redundancy", humanise(targetAssessment.current.pathway_redundancy)],
                  ["Human tissue expression", humanise(targetAssessment.current.human_tissue_expression_evidence)],
                  ["Prior clinical programs", targetAssessment.current.prior_clinical_programs_on_target ?? "—"],
                  ["Approved drugs on target", targetAssessment.current.prior_approved_drugs_on_target ?? "—"],
                  ["Assessed", shortDate(targetAssessment.current.available_at)],
                ]}
              />
            ) : (
              <EmptyState title="No target assessment available at the cutoff" />
            )}
          </div>
        </Card>
      </div>

      {/* ── Evidence ───────────────────────────────────────────────────── */}
      <EvidenceMatrixGrid matrix={matrix} />

      <Card>
        <CardHeader
          title="Supporting evidence"
          subtitle={`Every item available on or before ${shortDate(cutoff)}. Items dated after the cutoff exist in the record but never contribute to this estimate.`}
        />
        <EvidenceTable items={evidence} cutoff={cutoff} />
      </Card>

      {/* ── Safety ─────────────────────────────────────────────────────── */}
      {safety?.findings?.length ? (
        <Card>
          <CardHeader
            title="Preclinical safety findings"
            subtitle="Findings that constrain the dose, and therefore the exposure, achievable in humans."
            action={
              <span className="tnum font-mono text-[11px] text-ink-400">
                {safety.n_findings} finding{safety.n_findings === 1 ? "" : "s"}
              </span>
            }
          />
          <div className="scroll-thin overflow-x-auto">
            <table className="w-full border-collapse">
              <thead>
                <tr>
                  <Th>Organ system</Th><Th>Species</Th><Th>Severity</Th>
                  <Th>Reversible</Th><Th>Monitorable</Th><Th>On-target</Th><Th>Available</Th>
                </tr>
              </thead>
              <tbody>
                {safety.findings.map((f: any) => (
                  <tr key={f.id} className="hover:bg-ink-50">
                    <Td>{humanise(f.organ_system)}</Td>
                    <Td>{humanise(f.species)}</Td>
                    <Td>
                      <Badge tone={["severe", "life_threatening"].includes(f.severity) ? "refute" : "muted"}>
                        {humanise(f.severity)}
                      </Badge>
                    </Td>
                    <Td>{f.reversible === null ? "Not reported" : f.reversible ? "Yes" : "No"}</Td>
                    <Td>{f.monitorable_in_human === null ? "Not reported" : f.monitorable_in_human ? "Yes" : "No"}</Td>
                    <Td>{f.on_target === null ? "—" : f.on_target ? "Yes" : "No"}</Td>
                    <Td mono className="text-[11.5px]">{shortDate(f.available_at)}</Td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      ) : null}

      {/* ── Comparators ────────────────────────────────────────────────── */}
      <Card>
        <CardHeader
          title="Historical comparators"
          subtitle={`Programs that made a structurally similar bet and whose outcome was already known at ${shortDate(cutoff)}.`}
          action={
            comparators.summary.observed_failure_share !== null &&
            comparators.summary.observed_failure_share !== undefined ? (
              <span className="tnum font-mono text-[11px] text-ink-400">
                {comparators.summary.n_efficacy_failures}/
                {(comparators.summary.n_efficacy_failures ?? 0) +
                  (comparators.summary.n_progressed_past_phase_2 ?? 0)}{" "}
                failed on efficacy
              </span>
            ) : null
          }
        />
        {comparators.comparators.length ? (
          <>
            <div className="scroll-thin overflow-x-auto">
              <table className="w-full border-collapse">
                <thead>
                  <tr>
                    <Th>Program</Th><Th>Target</Th><Th>Indication</Th>
                    <Th align="right">Similarity</Th><Th>Why comparable</Th>
                    <Th>Outcome</Th><Th>Resolved</Th>
                  </tr>
                </thead>
                <tbody>
                  {comparators.comparators.map((c) => (
                    <tr key={c.track_id} className="hover:bg-ink-50">
                      <Td>
                        <Link
                          href={`/programs/${c.track_id}`}
                          className="font-medium text-ink-800 hover:text-accent-700 hover:underline"
                        >
                          {c.program_name}
                        </Link>
                        {c.sponsor ? (
                          <span className="block text-[11px] text-ink-400">{c.sponsor}</span>
                        ) : null}
                      </Td>
                      <Td mono>{c.target_symbol ?? "—"}</Td>
                      <Td className="max-w-[190px]">
                        <span className="block truncate" title={c.indication}>{c.indication}</span>
                      </Td>
                      <Td align="right" mono>
                        <div className="flex items-center justify-end gap-2">
                          <span className="h-1.5 w-10 overflow-hidden rounded-[1px] bg-ink-100">
                            <span
                              className="block h-full bg-accent-500"
                              style={{ width: `${c.similarity * 100}%` }}
                            />
                          </span>
                          {num(c.similarity, 2)}
                        </div>
                      </Td>
                      <Td className="max-w-[230px] text-[11.5px] text-ink-500">{c.rationale}</Td>
                      <Td>
                        <Badge
                          tone={
                            c.outcome_label.includes("failure") ? "refute"
                              : c.outcome_label === "regulatory_approval" ||
                                c.outcome_label === "progressed_past_phase_2" ? "support"
                              : "muted"
                          }
                        >
                          {c.outcome_display}
                        </Badge>
                      </Td>
                      <Td mono className="text-[11.5px]">{shortDate(c.outcome_date)}</Td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="border-t border-[var(--border-subtle)] p-5">
              <Callout tone="warn" title="Similarity is not exchangeability">
                {comparators.summary.caveat}
              </Callout>
            </div>
          </>
        ) : (
          <EmptyState
            title="No comparators met the threshold"
            detail={comparators.summary.caveat}
          />
        )}
      </Card>

      {/* ── Scenario analysis ──────────────────────────────────────────── */}
      <ScenarioPanel trackId={id} levers={levers.levers} current={score} />

      {/* ── Provenance footer ──────────────────────────────────────────── */}
      <Card>
        <CardHeader title="Reproducing this estimate" />
        <div className="p-5">
          <Prose className="mb-4">
            Every number on this page is reconstructable. The feature snapshot
            records the exact vector the model saw, which evidence rows produced
            each value, and the schema version under which it was built.
          </Prose>
          <KeyValue
            items={[
              ["Model version", <MonoId key="m" value={score.model_version} />],
              ["Model family", score.model_family],
              ["Feature snapshot", <MonoId key="s" value={score.feature_snapshot_id ?? "—"} />],
              ["Score record", <MonoId key="r" value={score.score_record_id ?? "not persisted"} />],
              ["As-of cutoff", shortDate(cutoff)],
              ["Computed", shortDate(score.computed_at)],
              ["Uncertainty method", <span key="u" className="font-mono text-[11.5px]">{score.uncertainty.method}</span>],
              ["Draws", <span key="d" className="tnum font-mono text-[11.5px]">
                {score.uncertainty.n_imputations} imputations × {score.uncertainty.n_param_samples} parameter samples
              </span>],
            ]}
          />
          {program.aliases.length ? (
            <div className="mt-5">
              <SectionLabel>Known designations for this asset</SectionLabel>
              <div className="mt-2 flex flex-wrap gap-1.5">
                {program.aliases.map((a) => (
                  <Pill key={a.alias}>
                    <span className="font-mono">{a.alias}</span>
                    <span className="ml-1.5 text-ink-400">{humanise(a.alias_type)}</span>
                  </Pill>
                ))}
              </div>
              <p className="mt-2 max-w-[70ch] text-[11.5px] text-ink-500">
                Resolved to one program so the same asset cannot appear twice in a
                cohort, and cannot straddle a training/test split under a different
                name.
              </p>
            </div>
          ) : null}
        </div>
      </Card>
    </div>
  );
}
