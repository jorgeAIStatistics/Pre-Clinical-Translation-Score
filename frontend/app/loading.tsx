export default function Loading() {
  return (
    <div className="space-y-4 py-8" aria-busy="true" aria-live="polite">
      <div className="h-5 w-52 animate-pulse rounded bg-ink-200" />
      <div className="h-3 w-[36ch] animate-pulse rounded bg-ink-100" />
      <div className="mt-6 space-y-2">
        {Array.from({ length: 8 }).map((_, i) => (
          <div key={i} className="h-9 animate-pulse rounded bg-ink-100" style={{ opacity: 1 - i * 0.09 }} />
        ))}
      </div>
      <span className="sr-only">Loading</span>
    </div>
  );
}
