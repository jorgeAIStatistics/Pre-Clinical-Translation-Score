import Link from "next/link";

import { PortfolioTable } from "@/components/portfolio/PortfolioTable";
import { Callout, Card, CardHeader, Prose, Stat } from "@/components/primitives";
import { api } from "@/lib/api";
import { pct } from "@/lib/format";

export const dynamic = "force-dynamic";

export default async function PortfolioPage() {
  const portfolio = await api.portfolio({ limit: 60, score: true });

  const scored = portfolio.rows.filter((r) => r.probability !== null);
  const meanRisk = scored.length
    ? scored.reduce((a, r) => a + (r.probability ?? 0), 0) / scored.length
    : null;
  const meanWidth = scored.length
    ? scored.reduce((a, r) => a + ((r.interval_high ?? 0) - (r.interval_low ?? 0)), 0) / scored.length
    : null;
  const lowConfidence = scored.filter(
    (r) => r.evidence_confidence === "low" || r.evidence_confidence === "very_low",
  ).length;

  return (
    <div className="space-y-6">
      <header className="flex flex-wrap items-end justify-between gap-6">
        <div>
          <h1 className="text-[20px] font-semibold tracking-[-0.01em] text-ink-900">
            {portfolio.project_name ?? "Portfolio"}
          </h1>
          <Prose className="mt-2">
            Translational-risk estimates for every program in the portfolio, each
            computed as of its own development decision date using only evidence
            available at that point.
          </Prose>
        </div>
        <dl className="flex gap-8">
          <Stat label="Programs" value={portfolio.total} />
          <Stat label="Mean estimate" value={pct(meanRisk)} hint="Across scored programs" />
          <Stat
            label="Mean interval width"
            value={pct(meanWidth)}
            hint="Absolute, 95% level"
          />
          <Stat
            label="Low confidence"
            value={`${lowConfidence}`}
            hint="Thin or incomplete evidence"
          />
        </dl>
      </header>

      <Callout tone="neutral" title="What this number is">
        <p className="mb-2">{portfolio.estimand}</p>
        <p className="text-[12px] text-ink-500">
          Estimates are cause-specific: programs withdrawn for safety,
          pharmacokinetic or commercial reasons are excluded from the training
          cohort rather than counted as efficacy failures.{" "}
          <Link href="/methods" className="text-accent-700 underline underline-offset-2">
            Methods &amp; validation
          </Link>
        </p>
      </Callout>

      {portfolio.model_usefulness?.headline ? (
        <Callout
          tone={
            portfolio.model_usefulness.verdict === "beats_base_rate" ? "accent" : "warn"
          }
          title="Held-out performance of the active model"
        >
          {portfolio.model_usefulness.headline}
        </Callout>
      ) : null}

      <Card>
        <CardHeader
          title="Programs"
          subtitle="Sorted by name by default. Risk, its interval and the evidence behind it are shown together — a ranking by probability alone would hide how much each estimate rests on."
          action={
            portfolio.model_version ? (
              <span className="font-mono text-[11px] text-ink-400">
                model {portfolio.model_version}
              </span>
            ) : null
          }
        />
        <PortfolioTable rows={portfolio.rows} />
      </Card>
    </div>
  );
}
