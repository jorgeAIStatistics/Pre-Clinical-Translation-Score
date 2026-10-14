import Link from "next/link";

import { PortfolioTable } from "@/components/portfolio/PortfolioTable";
import { Callout, Card, CardHeader, Prose, Stat } from "@/components/primitives";
import { api } from "@/lib/api";
import { pct } from "@/lib/format";

export const dynamic = "force-dynamic";

const PAGE_SIZE = 60;

export default async function PortfolioPage({
  searchParams,
}: {
  searchParams: Promise<{ page?: string }>;
}) {
  const { page } = await searchParams;
  const pageNumber = Math.max(1, Number.parseInt(page ?? "1", 10) || 1);
  const offset = (pageNumber - 1) * PAGE_SIZE;

  const portfolio = await api.portfolio({
    limit: PAGE_SIZE,
    offset,
    score: true,
  });

  // Summaries come from the server, which computed them over this page and
  // says so in `page_summary.scope`. Recomputing them here from `rows` would
  // produce the same numbers under a label the page chose for itself — which
  // is how the previous version came to describe a 60-row mean as covering
  // "every program in the portfolio".
  const summary = portfolio.page_summary;
  const firstRow = portfolio.total_matching === 0 ? 0 : offset + 1;
  const lastRow = offset + portfolio.n_returned;
  const refused = summary.n_refused_on_page;

  return (
    <div className="space-y-6">
      <header className="flex flex-wrap items-end justify-between gap-6">
        <div>
          <h1 className="text-[20px] font-semibold tracking-[-0.01em] text-ink-900">
            {portfolio.project_name ?? "Portfolio"}
          </h1>
          <Prose className="mt-2">
            Each programme is assessed as of its own development decision date,
            using only evidence available at that point. Showing{" "}
            <span className="tnum font-mono">{firstRow}</span>–
            <span className="tnum font-mono">{lastRow}</span> of{" "}
            <span className="tnum font-mono">{portfolio.total_matching}</span>{" "}
            matching programmes; the summaries beside this describe this page,
            not the whole portfolio.
          </Prose>
        </div>
        <dl className="flex gap-8">
          <Stat
            label="Programmes"
            value={portfolio.total_matching}
            hint={`${portfolio.n_returned} on this page`}
          />
          <Stat
            label="Mean estimate"
            value={pct(summary.mean_probability)}
            hint={`${summary.n_scored_on_page} scored on this page`}
          />
          <Stat
            label="Mean interval width"
            value={pct(summary.mean_interval_width)}
            hint="Absolute, 95% level, this page"
          />
          <Stat
            label={refused ? "No estimate" : "Low confidence"}
            value={`${refused || summary.n_low_confidence_on_page}`}
            hint={
              refused
                ? "Model not permitted to score"
                : "Thin or incomplete evidence, this page"
            }
          />
        </dl>
      </header>

      {refused ? (
        <Callout tone="warn" title="No estimates on this page">
          <p>
            The active model is not permitted to score{" "}
            <span className="tnum font-mono">{refused}</span> of{" "}
            <span className="tnum font-mono">{portfolio.n_returned}</span>{" "}
            programmes on this page. Evidence counts, completeness and
            provenance are unaffected; open a programme to see which evidence
            domains are covered and which are not.
          </p>
        </Callout>
      ) : null}

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

        {/* Pagination. Server-rendered links rather than client state, so a
            page is addressable, shareable and survives a reload — and so the
            row range in the header always matches the rows below it. */}
        {portfolio.total_matching > portfolio.n_returned ? (
          <nav
            aria-label="Portfolio pages"
            className="flex items-center justify-between gap-4 border-t border-ink-200 px-5 py-3"
          >
            <span className="tnum text-[12px] text-ink-500">
              Rows {firstRow}–{lastRow} of {portfolio.total_matching}
            </span>
            <span className="flex items-center gap-2">
              {pageNumber > 1 ? (
                <Link
                  href={`/?page=${pageNumber - 1}`}
                  className="rounded border border-ink-200 px-3 py-1.5 text-[12.5px] text-ink-700 hover:bg-ink-50 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent-600"
                >
                  ← Previous
                </Link>
              ) : (
                <span
                  aria-disabled="true"
                  className="rounded border border-ink-100 px-3 py-1.5 text-[12.5px] text-ink-300"
                >
                  ← Previous
                </span>
              )}
              <span className="tnum px-1 text-[12px] text-ink-500">
                page {pageNumber} of{" "}
                {Math.max(1, Math.ceil(portfolio.total_matching / PAGE_SIZE))}
              </span>
              {portfolio.has_more ? (
                <Link
                  href={`/?page=${pageNumber + 1}`}
                  className="rounded border border-ink-200 px-3 py-1.5 text-[12.5px] text-ink-700 hover:bg-ink-50 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent-600"
                >
                  Next →
                </Link>
              ) : (
                <span
                  aria-disabled="true"
                  className="rounded border border-ink-100 px-3 py-1.5 text-[12.5px] text-ink-300"
                >
                  Next →
                </span>
              )}
            </span>
          </nav>
        ) : null}
      </Card>
    </div>
  );
}
