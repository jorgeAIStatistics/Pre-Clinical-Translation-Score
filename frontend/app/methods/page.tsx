import { CalibrationPlot, DecisionCurve } from "@/components/charts";
import {
  Badge, Callout, Card, CardHeader, Prose, SectionLabel, Stat, Td, Th,
} from "@/components/primitives";
import { api } from "@/lib/api";
import { ci, humanise, num, pct, shortDate } from "@/lib/format";

export const dynamic = "force-dynamic";

/**
 * Methods & validation.
 *
 * The page that decides whether a statistically literate reader trusts the
 * product. It therefore leads with the unflattering material — whether the
 * model beats the base rate at all, what the split discards, which domains
 * contribute nothing, which coefficients contradict their stated priors — and
 * treats discrimination as one line item rather than the headline.
 */
export default async function MethodsPage() {
  const [report, coefficients, versions, cohorts] = await Promise.all([
    api.validation(),
    api.coefficients().catch(() => []),
    api.modelVersions().catch(() => []),
    api.cohorts().catch(() => []),
  ]);

  const pm = report.primary_metrics;
  const cal = pm.calibration;
  const design = report.evaluation_design;
  const oracle = report.oracle_benchmark;
  const discordant = report.coefficient_sign_audit?.discordant ?? [];
  const ablations = report.domain_ablation?.by_domain ?? [];

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-[20px] font-semibold tracking-[-0.01em] text-ink-900">
          Methods &amp; validation
        </h1>
        <Prose className="mt-2">
          How the cohort was constructed, how the model was selected and
          calibrated, how it was evaluated, and what it cannot do. Everything
          here is generated from the model artifact itself, not written by hand.
        </Prose>
        <div className="mt-3 flex flex-wrap items-center gap-2">
          <Badge tone="accent">model {report.model_version}</Badge>
          {report.generated_at ? (
            <Badge tone="muted">report generated {shortDate(report.generated_at)}</Badge>
          ) : null}
        </div>
      </header>

      {/* ── The threshold question ─────────────────────────────────────── */}
      {report.usefulness?.headline ? (
        <Callout
          tone={report.usefulness.verdict === "beats_base_rate" ? "accent" : "warn"}
          title="Does this model beat simply quoting the base rate?"
        >
          <p className="text-[13px]">{report.usefulness.headline}</p>
          {report.usefulness.why_this_matters ? (
            <p className="mt-2 text-[12px] text-ink-500">{report.usefulness.why_this_matters}</p>
          ) : null}
        </Callout>
      ) : null}

      {/* ── Headline metrics ───────────────────────────────────────────── */}
      <Card>
        <CardHeader
          title="Held-out performance"
          subtitle={design.statement}
          action={
            <span className="tnum font-mono text-[11px] text-ink-400">
              n = {pm.n} · {pm.n_positive} events
            </span>
          }
        />
        <div className="grid gap-6 p-5 lg:grid-cols-[minmax(0,1fr)_320px]">
          <div>
            <div className="grid grid-cols-2 gap-x-8 gap-y-5 sm:grid-cols-3">
              <Stat label="AUROC" value={ci(pm.discrimination.auroc)} hint="Discrimination (95% CI)" />
              <Stat label="AUPRC" value={ci(pm.discrimination.auprc)} hint={`Prevalence ${pct(pm.prevalence)}`} />
              <Stat label="Brier score" value={ci(pm.accuracy.brier)} hint="Lower is better" />
              <Stat label="Brier skill" value={ci(pm.accuracy.scaled_brier)} hint="vs. base rate; >0 is useful" />
              <Stat label="Log loss" value={ci(pm.accuracy.log_loss)} hint="Proper scoring rule" />
              <Stat label="ECE" value={num(cal.ece, 3)} hint="Expected calibration error" />
              <Stat label="Calibration slope" value={ci(cal.calibration_slope, 2)} hint="1.0 is perfect" />
              <Stat label="Calibration intercept" value={ci(cal.calibration_intercept, 2)} hint="0.0 is perfect" />
            </div>

            {oracle?.available ? (
              <div className="mt-6">
                <Callout tone="neutral" title="The attainable ceiling on this dataset">
                  <p>
                    Outcomes in the demonstration dataset are Bernoulli draws from a
                    known probability, so the maximum discrimination any model could
                    reach is computable:{" "}
                    <span className="tnum font-mono text-ink-800">
                      AUROC {num(oracle.oracle_auroc, 3)}
                    </span>
                    , with log loss{" "}
                    <span className="tnum font-mono text-ink-800">
                      {num(oracle.oracle_log_loss, 3)}
                    </span>{" "}
                    against a base-rate model&apos;s{" "}
                    <span className="tnum font-mono text-ink-800">
                      {num(oracle.base_rate_log_loss, 3)}
                    </span>
                    .
                  </p>
                  <p className="mt-2 text-[12px] text-ink-500">{oracle.interpretation}</p>
                </Callout>
              </div>
            ) : null}
          </div>

          <div>
            <SectionLabel>Calibration</SectionLabel>
            <div className="mt-2">
              <CalibrationPlot
                bins={cal.bins}
                slope={cal.calibration_slope.value}
                intercept={cal.calibration_intercept.value}
              />
            </div>
            <p className="mt-2 text-[11.5px] leading-[1.55] text-ink-500">
              Points are quantile bins, sized by count. Vertical bars are Wilson
              intervals on the observed rate. The dashed diagonal is perfect
              calibration.
            </p>
          </div>
        </div>
      </Card>

      {/* ── Decision curve ─────────────────────────────────────────────── */}
      <Card>
        <CardHeader
          title="Clinical utility"
          subtitle={pm.decision_curve.interpretation}
        />
        <div className="p-5">
          {/* Charts scale to their container; an unbounded width blows the
              SVG's internal font sizes up with it. Capping the width keeps the
              axis labels at the size they were designed for. */}
          <div className="max-w-[640px]">
            <DecisionCurve data={pm.decision_curve} />
          </div>
        </div>
      </Card>

      {/* ── Evaluation design ──────────────────────────────────────────── */}
      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader title="Evaluation design" subtitle={design.split} />
          <div className="space-y-4 p-5">
            <div className="flex flex-wrap gap-6">
              {Object.entries(design.sizes).map(([k, v]) => (
                <Stat key={k} label={humanise(k)} value={String(v)} />
              ))}
            </div>

            <div>
              <SectionLabel>Leakage audit</SectionLabel>
              <div className="mt-2 flex flex-wrap gap-1.5">
                {Object.entries(design.leakage_audit)
                  .filter(([k]) => !["clean", "scope", "enforced_keys", "informational_overlap"].includes(k))
                  .map(([k, v]) => (
                    <Badge key={k} tone={Number(v) === 0 ? "support" : "refute"}>
                      {humanise(k)}: {String(v)} shared
                    </Badge>
                  ))}
              </div>
              <p className="mt-2 text-[11.5px] leading-[1.55] text-ink-500">
                Enforced grouping keys:{" "}
                <span className="font-mono">
                  {(design.leakage_audit as any).enforced_keys?.join(", ") ?? "—"}
                </span>
                . Overlap on keys outside the enforced scope is reported for
                context: sharing a target with a training program is part of the
                information a real analyst had, not leakage.
              </p>
            </div>

            <div>
              <SectionLabel>Grouping</SectionLabel>
              <p className="mt-1.5 text-[12px] leading-[1.6] text-ink-600">
                {(design.grouping as any).n_components} independent connected
                components across {(design.grouping as any).n_rows} rows; largest
                component {(design.grouping as any).largest_component}.
              </p>
              <p className="mt-1.5 text-[11.5px] leading-[1.55] text-ink-500">
                {(design.grouping as any).effective_sample_size_note}
              </p>
            </div>
          </div>
        </Card>

        <Card>
          <CardHeader
            title="Cohort construction"
            subtitle={report.cohort?.endpoint?.estimand}
          />
          <div className="space-y-4 p-5">
            <div className="flex flex-wrap gap-6">
              <Stat label="Included" value={report.cohort?.n_entries ?? "—"} />
              <Stat label="Positive" value={report.cohort?.n_positive ?? "—"} />
              <Stat label="Negative" value={report.cohort?.n_negative ?? "—"} />
              <Stat label="Excluded" value={report.cohort?.n_excluded ?? "—"} />
            </div>
            <div>
              <SectionLabel>Exclusion waterfall</SectionLabel>
              <ul className="mt-2 space-y-1">
                {Object.entries(report.cohort?.inclusion_summary?.exclusions ?? {}).map(
                  ([reason, n]) => (
                    <li key={reason} className="flex items-baseline justify-between gap-4 text-[12px]">
                      <span className="text-ink-600">{humanise(reason.split(":")[0])}
                        {reason.includes(":") ? (
                          <span className="ml-1 text-ink-400">{reason.split(":").slice(1).join(":")}</span>
                        ) : null}
                      </span>
                      <span className="tnum font-mono text-ink-500">{String(n)}</span>
                    </li>
                  ),
                )}
              </ul>
              <p className="mt-2 max-w-[62ch] text-[11.5px] leading-[1.55] text-ink-500">
                Commercial and strategic discontinuations are excluded rather than
                counted as failures: a model trained on them would predict
                corporate behaviour while claiming to predict biology.
              </p>
            </div>
          </div>
        </Card>
      </div>

      {/* ── Model selection ────────────────────────────────────────────── */}
      <Card>
        <CardHeader
          title="Model selection"
          subtitle={report.model_comparison?.rationale}
        />
        <div className="scroll-thin overflow-x-auto">
          <table className="w-full border-collapse">
            <thead>
              <tr>
                <Th>Family</Th><Th align="right">Log loss</Th><Th align="right">AUROC</Th>
                <Th align="right">Cal. slope</Th><Th align="right">Required margin</Th>
                <Th>Verdict</Th>
              </tr>
            </thead>
            <tbody>
              {(report.model_comparison?.comparisons ?? []).map((c: any) => (
                <tr key={c.family} className="align-top hover:bg-ink-50">
                  <Td className="whitespace-nowrap font-medium">
                    {humanise(c.family)}
                    {c.is_baseline ? (
                      <span className="ml-2"><Badge tone="accent">baseline</Badge></span>
                    ) : null}
                  </Td>
                  <Td align="right" mono>{num(c.log_loss, 4)}</Td>
                  <Td align="right" mono>{num(c.auroc, 3)}</Td>
                  <Td align="right" mono>{num(c.calibration_slope, 2)}</Td>
                  <Td align="right" mono>
                    {c.required_margin !== undefined ? pct(c.required_margin, 0) : "—"}
                  </Td>
                  <Td className="max-w-[46ch] text-[11.5px] leading-[1.5] text-ink-600">
                    {c.verdict}
                    {c.margin_rationale ? (
                      <span className="mt-1 block text-[10.5px] text-ink-400">
                        {c.margin_rationale.join(" ")}
                      </span>
                    ) : null}
                  </Td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="border-t border-[var(--border-subtle)] p-5">
          <p className="max-w-[86ch] text-[11.5px] leading-[1.6] text-ink-500">
            {report.model_comparison?.gate?.note}{" "}
            A challenger that cannot produce an exact additive explanation, or an
            analytic posterior, must clear a larger margin — losing those
            capabilities is a real cost to the product, and the gate prices it
            explicitly rather than ignoring it.
          </p>
        </div>
      </Card>

      {/* ── Calibration method ─────────────────────────────────────────── */}
      <Card>
        <CardHeader
          title="Calibration method"
          subtitle={report.calibration_selection?.reason}
        />
        <div className="p-5">
          <div className="flex flex-wrap gap-6">
            <Stat label="Selected" value={humanise(report.calibration_selection?.method)} mono={false} />
            <Stat label="Design" value={humanise(report.calibration_selection?.design)} mono={false} />
            <Stat
              label="Calibration observations"
              value={
                report.calibration_selection?.n_out_of_fold_predictions ??
                report.calibration_selection?.n_calibration ?? "—"
              }
            />
          </div>
          {report.calibration_selection?.candidate_scores ? (
            <div className="mt-4">
              <SectionLabel>Cross-validated log loss by method</SectionLabel>
              <ul className="mt-2 space-y-1">
                {Object.entries(report.calibration_selection.candidate_scores).map(([m, v]) => (
                  <li key={m} className="flex items-baseline justify-between gap-4 text-[12px]">
                    <span className="text-ink-600">{humanise(m)}</span>
                    <span className="tnum font-mono text-ink-500">{num(v as number, 4)}</span>
                  </li>
                ))}
              </ul>
            </div>
          ) : null}
          {Object.keys(report.calibration_selection?.excluded_candidates ?? {}).length ? (
            <div className="mt-4 space-y-1.5">
              {Object.entries(report.calibration_selection.excluded_candidates).map(([m, why]) => (
                <p key={m} className="text-[11.5px] leading-[1.55] text-ink-500">
                  <span className="font-medium text-ink-600">{humanise(m)}:</span> {String(why)}
                </p>
              ))}
            </div>
          ) : null}
          {report.calibration_selection?.known_bias ? (
            <p className="mt-4 max-w-[86ch] text-[11.5px] leading-[1.6] text-ink-500">
              {report.calibration_selection.known_bias}
            </p>
          ) : null}
        </div>
      </Card>

      {/* ── Ablation and sensitivity ───────────────────────────────────── */}
      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader
            title="Domain ablation"
            subtitle={report.domain_ablation?.interpretation}
          />
          <div className="scroll-thin max-h-[380px] overflow-auto">
            <table className="w-full border-collapse">
              <thead className="sticky top-0 bg-[var(--surface-raised)]">
                <tr>
                  <Th>Domain removed</Th><Th align="right">Δ log loss</Th><Th align="right">Δ AUROC</Th>
                </tr>
              </thead>
              <tbody>
                {ablations.map((a: any) => (
                  <tr key={a.domain} className="hover:bg-ink-50">
                    <Td>{a.domain_label}</Td>
                    <Td align="right" mono
                        className={a.delta_log_loss_vs_full > 0.005 ? "font-semibold" : ""}>
                      {a.delta_log_loss_vs_full === null ? "—"
                        : `${a.delta_log_loss_vs_full >= 0 ? "+" : ""}${num(a.delta_log_loss_vs_full, 4)}`}
                    </Td>
                    <Td align="right" mono>
                      {a.delta_auroc_vs_full === null ? "—"
                        : `${a.delta_auroc_vs_full >= 0 ? "+" : ""}${num(a.delta_auroc_vs_full, 4)}`}
                    </Td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>

        <Card>
          <CardHeader
            title="Robustness checks"
            subtitle="Two checks on whether the reported numbers mean what they appear to."
          />
          <div className="space-y-5 p-5">
            <div>
              <SectionLabel>Target-grouped sensitivity</SectionLabel>
              {report.target_grouped_sensitivity?.available ? (
                <>
                  <p className="mt-1.5 text-[12px] leading-[1.6] text-ink-600">
                    With no program sharing a target allowed across the split,
                    AUROC is{" "}
                    <span className="tnum font-mono text-ink-800">
                      {num(report.target_grouped_sensitivity.auroc, 3)}
                    </span>
                    {" "}against{" "}
                    <span className="tnum font-mono text-ink-800">
                      {num(pm.discrimination.auroc.value, 3)}
                    </span>{" "}
                    on the headline temporal split.
                  </p>
                  <p className="mt-1.5 text-[11.5px] leading-[1.55] text-ink-500">
                    {report.target_grouped_sensitivity.interpretation}
                  </p>
                </>
              ) : (
                <p className="mt-1.5 text-[12px] text-ink-500">
                  {report.target_grouped_sensitivity?.reason ?? "Not available."}
                </p>
              )}
            </div>

            <div>
              <SectionLabel>Interval agreement</SectionLabel>
              {report.interval_agreement?.available ? (
                <>
                  <p className="mt-1.5 text-[12px] leading-[1.6] text-ink-600">
                    Analytic interval widths are{" "}
                    <span className="tnum font-mono text-ink-800">
                      {num(report.interval_agreement.median_width_ratio_analytic_over_bootstrap, 2)}×
                    </span>{" "}
                    the bootstrap-refit widths over{" "}
                    {report.interval_agreement.n_refits} refits — verdict{" "}
                    <Badge
                      tone={report.interval_agreement.verdict === "acceptable" ? "support" : "warn"}
                    >
                      {humanise(report.interval_agreement.verdict)}
                    </Badge>
                  </p>
                  <p className="mt-1.5 text-[11.5px] leading-[1.55] text-ink-500">
                    {report.interval_agreement.interpretation}
                  </p>
                </>
              ) : (
                <p className="mt-1.5 text-[12px] text-ink-500">
                  {report.interval_agreement?.reason ?? "Not available."}
                </p>
              )}
            </div>
          </div>
        </Card>
      </div>

      {/* ── Subgroups ──────────────────────────────────────────────────── */}
      <Card>
        <CardHeader
          title="Performance by subgroup"
          subtitle="An aggregate metric can hide a model that works for one modality and is noise for another. Subgroups below the reporting threshold are shown as not reportable rather than reported with a caveat."
        />
        <div className="divide-y divide-[var(--border-subtle)]">
          {Object.entries(report.subgroup_performance).map(([variable, data]) => (
            <div key={variable} className="p-5">
              <SectionLabel>{humanise(variable)}</SectionLabel>
              <div className="scroll-thin mt-3 overflow-x-auto">
                <table className="w-full border-collapse">
                  <thead>
                    <tr>
                      <Th>Level</Th><Th align="right">n</Th><Th align="right">Events</Th>
                      <Th align="right">AUROC</Th><Th align="right">Brier</Th>
                      <Th align="right">Cal. slope</Th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.levels.map((lv: any) => (
                      <tr key={lv.level} className="hover:bg-ink-50">
                        <Td>{humanise(lv.level)}</Td>
                        <Td align="right" mono>{lv.n}</Td>
                        <Td align="right" mono>{lv.n_positive}</Td>
                        {lv.reported ? (
                          <>
                            <Td align="right" mono>{ci(lv.auroc)}</Td>
                            <Td align="right" mono>{num(lv.brier?.value, 3)}</Td>
                            <Td align="right" mono>{num(lv.calibration_slope?.value, 2)}</Td>
                          </>
                        ) : (
                          <Td align="right" className="text-[11.5px] text-ink-400" colSpan={3}>
                            Not reportable — {lv.reason}
                          </Td>
                        )}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ))}
        </div>
      </Card>

      {/* ── Coefficients ───────────────────────────────────────────────── */}
      <Card>
        <CardHeader
          title="Fitted coefficients"
          subtitle="On the standardised design-matrix scale: the change in log-odds per one training standard deviation, holding the others fixed. Every driver shown anywhere in the product is one of these multiplied by a feature value."
          action={
            discordant.length ? (
              <Badge tone="warn">
                {report.coefficient_sign_audit.n_discordant_credible} credible sign discordance
                {report.coefficient_sign_audit.n_discordant_credible === 1 ? "" : "s"}
              </Badge>
            ) : (
              <Badge tone="support">No credible sign discordances</Badge>
            )
          }
        />
        <div className="scroll-thin max-h-[520px] overflow-auto">
          <table className="w-full border-collapse">
            <thead className="sticky top-0 bg-[var(--surface-raised)]">
              <tr>
                <Th>Feature</Th><Th>Domain</Th>
                <Th align="right">Coefficient</Th><Th align="right">95% credible interval</Th>
                <Th align="right">OR per SD</Th><Th>Prior direction</Th>
              </tr>
            </thead>
            <tbody>
              {coefficients.slice(0, 60).map((c) => (
                <tr key={c.column} className="hover:bg-ink-50">
                  <Td>
                    <span className="text-ink-800">{c.label ?? c.column}</span>
                    <span className="mt-0.5 block font-mono text-[10px] text-ink-400">{c.column}</span>
                  </Td>
                  <Td className="text-[11.5px] text-ink-500">{humanise(c.domain)}</Td>
                  <Td align="right" mono>{num(c.coefficient, 3)}</Td>
                  <Td align="right" mono className="text-[11.5px] text-ink-500">
                    {c.ci_low === null ? "—" : `${num(c.ci_low, 2)} – ${num(c.ci_high, 2)}`}
                  </Td>
                  <Td align="right" mono>{num(c.odds_ratio_per_sd, 2)}</Td>
                  <Td>
                    {c.expected_direction && c.expected_direction !== "uncertain" ? (
                      c.sign_matches_expectation === false ? (
                        <Badge tone="warn">opposes prior</Badge>
                      ) : (
                        <Badge tone="muted">{c.expected_direction === "increases_risk" ? "↑ risk" : "↓ risk"}</Badge>
                      )
                    ) : (
                      <span className="text-[11px] text-ink-400">estimated</span>
                    )}
                  </Td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {report.coefficient_sign_audit?.interpretation ? (
          <div className="border-t border-[var(--border-subtle)] p-5">
            <p className="max-w-[86ch] text-[11.5px] leading-[1.6] text-ink-500">
              {report.coefficient_sign_audit.interpretation}
            </p>
          </div>
        ) : null}
      </Card>

      {/* ── Limitations ────────────────────────────────────────────────── */}
      <Card>
        <CardHeader
          title="Limitations"
          subtitle="Written into the model artifact at training time, not appended afterwards."
        />
        <div className="p-5">
          <ol className="space-y-3">
            {report.limitations.map((l, i) => (
              <li key={i} className="flex gap-3">
                <span className="tnum mt-[1px] font-mono text-[11px] text-ink-400">
                  {String(i + 1).padStart(2, "0")}
                </span>
                <p className="max-w-[86ch] text-[12.5px] leading-[1.65] text-ink-600">{l}</p>
              </li>
            ))}
          </ol>
        </div>
      </Card>

      {/* ── Registry ───────────────────────────────────────────────────── */}
      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader title="Model registry" />
          <div className="scroll-thin overflow-x-auto">
            <table className="w-full border-collapse">
              <thead>
                <tr>
                  <Th>Version</Th><Th>Family</Th><Th align="right">Train / test</Th>
                  <Th>Trained</Th><Th>Active</Th>
                </tr>
              </thead>
              <tbody>
                {versions.map((v: any) => (
                  <tr key={v.version} className="hover:bg-ink-50">
                    <Td mono>{v.version}</Td>
                    <Td className="text-[11.5px]">{humanise(v.family)}</Td>
                    <Td align="right" mono>{v.n_train} / {v.n_test}</Td>
                    <Td mono className="text-[11.5px]">{shortDate(v.trained_at)}</Td>
                    <Td>{v.is_active ? <Badge tone="accent">active</Badge> : null}</Td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>

        <Card>
          <CardHeader title="Cohort versions" />
          <div className="scroll-thin overflow-x-auto">
            <table className="w-full border-collapse">
              <thead>
                <tr>
                  <Th>Cohort</Th><Th align="right">n</Th><Th align="right">Positive</Th>
                  <Th>As-of rule</Th><Th>Dataset hash</Th>
                </tr>
              </thead>
              <tbody>
                {cohorts.map((c: any) => (
                  <tr key={c.id} className="hover:bg-ink-50">
                    <Td>{c.name}/{c.version}</Td>
                    <Td align="right" mono>{c.n_entries}</Td>
                    <Td align="right" mono>{c.n_positive}</Td>
                    <Td className="text-[11.5px]">{humanise(c.as_of_rule)}</Td>
                    <Td mono className="text-[10.5px] text-ink-400">
                      {c.dataset_hash?.slice(0, 12)}…
                    </Td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </div>
    </div>
  );
}
