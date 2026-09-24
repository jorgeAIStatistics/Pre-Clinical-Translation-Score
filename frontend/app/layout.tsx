import type { Metadata } from "next";
import Link from "next/link";

import { api } from "@/lib/api";
import { DemoBanner } from "@/components/primitives";

import "./globals.css";

export const metadata: Metadata = {
  title: "Translational Risk Scorer",
  description:
    "A quantitative translational intelligence system: estimates and explains the risk that preclinical evidence fails to translate into human clinical efficacy.",
};

const NAV = [
  { href: "/", label: "Portfolio" },
  { href: "/evidence", label: "Evidence" },
  { href: "/methods", label: "Methods & validation" },
  { href: "/provenance", label: "Provenance" },
] as const;

export default async function RootLayout({ children }: { children: React.ReactNode }) {
  const health = await api.health();

  return (
    <html lang="en">
      <body className="min-h-screen antialiased">
        {health?.is_demonstration_deployment ? <DemoBanner /> : null}

        <header className="sticky top-0 z-20 border-b border-[var(--border-subtle)] bg-[var(--surface-raised)]/95 backdrop-blur">
          <div className="mx-auto flex h-14 max-w-[1440px] items-center gap-8 px-6">
            <Link href="/" className="group flex items-baseline gap-2.5">
              <span className="text-[13.5px] font-semibold tracking-[-0.01em] text-ink-900">
                Translational Risk Scorer
              </span>
              <span className="hidden text-[10.5px] uppercase tracking-[0.1em] text-ink-400 sm:inline">
                Preclinical → Phase II
              </span>
            </Link>

            <nav className="flex items-center gap-1" aria-label="Main">
              {NAV.map((item) => (
                <Link
                  key={item.href}
                  href={item.href}
                  className="rounded-[3px] px-2.5 py-1.5 text-[12.5px] text-ink-600 transition-colors hover:bg-ink-100 hover:text-ink-900"
                >
                  {item.label}
                </Link>
              ))}
            </nav>

            <div className="ml-auto flex items-center gap-4 text-[11px] text-ink-400">
              {health?.active_model ? (
                <span className="font-mono" title="Active model artifact version">
                  {health.active_model}
                </span>
              ) : (
                <span className="text-[#8a6320]">No active model</span>
              )}
              <span className="font-mono" title="Feature schema version">
                {health?.feature_schema_version ?? "—"}
              </span>
              <span
                className="inline-flex items-center gap-1.5"
                title={`API ${health?.status ?? "unreachable"} · database ${health?.database ?? "unknown"}`}
              >
                <span
                  aria-hidden
                  className="inline-block h-1.5 w-1.5 rounded-full"
                  style={{
                    background: health?.status === "ok"
                      ? "var(--color-support)"
                      : "var(--color-refute)",
                  }}
                />
                {health?.status === "ok" ? "API" : "API down"}
              </span>
            </div>
          </div>
        </header>

        <main className="mx-auto max-w-[1440px] px-6 py-7">{children}</main>

        <footer className="mt-16 border-t border-[var(--border-subtle)]">
          <div className="mx-auto max-w-[1440px] px-6 py-6">
            <p className="max-w-[92ch] text-[11.5px] leading-[1.6] text-ink-500">
              This platform estimates the risk that preclinical evidence fails to
              translate into human clinical efficacy. It is decision support, not a
              substitute for scientific judgement, and it carries no regulatory
              standing. Every estimate is conditional on the evidence recorded at a
              stated cutoff date and on the documented modelling choices set out
              under <Link href="/methods" className="text-accent-700 underline underline-offset-2">Methods &amp; validation</Link>.
            </p>
          </div>
        </footer>
      </body>
    </html>
  );
}
