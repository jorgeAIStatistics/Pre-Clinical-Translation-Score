import {
  Badge, Callout, Card, CardHeader, Prose, SectionLabel, Stat, Td, Th,
} from "@/components/primitives";
import { api } from "@/lib/api";
import { humanise, num } from "@/lib/format";

export const dynamic = "force-dynamic";

const STATUS_TONE = {
  ok: "support",
  degraded: "warn",
  unreachable: "refute",
  not_configured: "muted",
  disabled: "muted",
} as const;

/**
 * Data provenance.
 *
 * Shows where evidence comes from, whether each source is currently reachable,
 * and what the platform does when one is not. Reachability is presented as a
 * normal operating state rather than an error, because a public biomedical API
 * being unavailable is routine and the honest response is to say so — not to
 * return an empty result that reads as "no evidence found".
 */
export default async function ProvenancePage() {
  const [health, catalogue, runs] = await Promise.all([
    api.connectors().catch(() => []),
    api.connectorCatalogue().catch(() => ({ connectors: [], note: "" })),
    api.ingestionRuns().catch(() => []),
  ]);

  const meta = new Map<string, any>(
    (catalogue.connectors ?? []).map((c: any) => [c.name, c]),
  );
  const reachable = health.filter((h) => h.status === "ok").length;

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-[20px] font-semibold tracking-[-0.01em] text-ink-900">
          Data provenance
        </h1>
        <Prose className="mt-2">
          Every value in this platform can be traced back to the payload that
          produced it. Connectors retain the verbatim response they receive
          before anything interprets it, so a derived number can always be
          re-checked against its source rather than trusted.
        </Prose>
      </header>

      <div className="flex gap-8">
        <Stat label="Connectors" value={health.length} />
        <Stat label="Reachable now" value={`${reachable} / ${health.length}`} />
      </div>

      <Card>
        <CardHeader
          title="Ingestion sources"
          subtitle={catalogue.note}
        />
        <div className="scroll-thin overflow-x-auto">
          <table className="w-full border-collapse">
            <thead>
              <tr>
                <Th>Source</Th><Th>Status</Th><Th align="right">Latency</Th>
                <Th>Detail</Th><Th>Terms of use</Th>
              </tr>
            </thead>
            <tbody>
              {health.map((h) => {
                const m = meta.get(h.connector);
                return (
                  <tr key={h.connector} className="align-top hover:bg-ink-50">
                    <Td>
                      <span className="font-medium text-ink-800">
                        {m?.display_name ?? h.connector}
                      </span>
                      {m?.homepage ? (
                        <a
                          href={m.homepage}
                          target="_blank"
                          rel="noreferrer noopener"
                          className="mt-0.5 block text-[11px] text-accent-700 hover:underline"
                        >
                          {m.homepage.replace(/^https?:\/\//, "")}
                        </a>
                      ) : null}
                    </Td>
                    <Td>
                      <Badge tone={STATUS_TONE[h.status] ?? "muted"}>
                        {humanise(h.status)}
                      </Badge>
                    </Td>
                    <Td align="right" mono>
                      {h.latency_ms === null ? "—" : `${num(h.latency_ms, 0)} ms`}
                    </Td>
                    <Td className="max-w-[52ch] text-[11.5px] leading-[1.55] text-ink-500">
                      {h.detail || "—"}
                    </Td>
                    <Td className="max-w-[34ch] text-[11px] leading-[1.5] text-ink-500">
                      {m?.license_note || "—"}
                    </Td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Card>

      <Card>
        <CardHeader title="The provenance chain" />
        <div className="p-5">
          <ol className="space-y-4">
            {[
              ["Raw record",
                "The verbatim bytes a connector received, content-addressed by SHA-256. Append-only: never rewritten, never deleted, so any derived value can be re-derived and any extraction bug diagnosed against exactly the payload that caused it."],
              ["Source",
                "The citable object — an article, a registry entry, a label — with its identifiers, publication date, retrieval timestamp and retraction status."],
              ["Evidence item",
                "One structured, translationally-relevant claim: one system, one endpoint, one effect. Carries its own available_at, the extraction method, the extractor version and the verbatim snippet it was read from."],
              ["Feature snapshot",
                "The model-ready vector for one program at one cutoff date, with a per-feature map back to the evidence rows that produced each value, a completeness assessment and a hash of the vector itself."],
              ["Score record",
                "The prediction, its interval, the decomposition of that interval and the driver attribution — bound to a model version and a feature snapshot, so it can be reconstructed after the model has been retrained."],
            ].map(([title, body], i) => (
              <li key={title} className="flex gap-4">
                <span className="tnum mt-[2px] font-mono text-[11px] text-ink-400">
                  {String(i + 1).padStart(2, "0")}
                </span>
                <div>
                  <div className="text-[12.5px] font-medium text-ink-800">{title}</div>
                  <p className="mt-0.5 max-w-[84ch] text-[12px] leading-[1.6] text-ink-600">{body}</p>
                </div>
              </li>
            ))}
          </ol>
        </div>
      </Card>

      <Card>
        <CardHeader title="How the platform handles source failure" />
        <div className="space-y-3 p-5">
          <Callout tone="neutral" title="An unreachable source is a state, not an error">
            A connector that cannot reach its API reports{" "}
            <span className="font-mono text-[11.5px]">unreachable</span> and, where a
            cached payload exists, serves it with every derived record flagged{" "}
            <span className="font-mono text-[11.5px]">from_cache</span>. It never
            fabricates, and it never returns an empty result that a caller could
            mistake for &ldquo;no evidence found&rdquo;.
          </Callout>
          <Callout tone="warn" title="Scores that cannot be back-dated">
            Some sources publish aggregate scores computed from whatever evidence
            exists at the moment of retrieval. Those records are dated to their
            retrieval timestamp and carry an explicit temporal caveat, because
            using them for an earlier cutoff would import information that did not
            exist at that cutoff.
          </Callout>
        </div>
      </Card>

      {runs?.length ? (
        <Card>
          <CardHeader title="Recent ingestion runs" />
          <div className="scroll-thin overflow-x-auto">
            <table className="w-full border-collapse">
              <thead>
                <tr>
                  <Th>Connector</Th><Th>Operation</Th><Th>Status</Th>
                  <Th align="right">Raw</Th><Th align="right">Normalised</Th><Th align="right">Errors</Th>
                </tr>
              </thead>
              <tbody>
                {runs.map((r: any) => (
                  <tr key={r.id} className="hover:bg-ink-50">
                    <Td>{r.connector}</Td>
                    <Td>{humanise(r.operation)}</Td>
                    <Td><Badge tone={r.status === "succeeded" ? "support" : "warn"}>{r.status}</Badge></Td>
                    <Td align="right" mono>{r.n_records_raw}</Td>
                    <Td align="right" mono>{r.n_records_normalised}</Td>
                    <Td align="right" mono>{r.n_errors}</Td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      ) : (
        <Card>
          <CardHeader title="Recent ingestion runs" />
          <div className="p-5">
            <SectionLabel>No runs recorded</SectionLabel>
            <Prose className="mt-2">
              This deployment is seeded with demonstration data, which is generated
              rather than ingested. Once a connector pipeline runs, each execution
              appears here with its request, record and error counts.
            </Prose>
          </div>
        </Card>
      )}
    </div>
  );
}
