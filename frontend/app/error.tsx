"use client";

import Link from "next/link";
import { useEffect } from "react";

import { Callout } from "@/components/primitives";

export default function Error({ error, reset }: { error: Error; reset: () => void }) {
  useEffect(() => {
    console.error(error);
  }, [error]);

  return (
    <div className="mx-auto max-w-[62ch] py-16">
      <h1 className="text-[18px] font-semibold text-ink-900">Something went wrong</h1>
      <div className="mt-4">
        <Callout tone="warn" title="What happened">
          <p className="mb-3">{error.message || "An unexpected error occurred."}</p>
          <p className="text-[12px] text-ink-500">
            The most common cause is that the API is not running. Start it with{" "}
            <code className="rounded bg-ink-100 px-1 py-0.5 font-mono text-[11px]">
              make api
            </code>{" "}
            from the repository root, then retry.
          </p>
        </Callout>
      </div>
      <div className="mt-5 flex gap-3">
        <button
          onClick={reset}
          className="rounded-[3px] border border-ink-300 bg-white px-3 py-1.5 text-[12.5px] text-ink-700 hover:bg-ink-50"
        >
          Retry
        </button>
        <Link
          href="/"
          className="rounded-[3px] px-3 py-1.5 text-[12.5px] text-accent-700 hover:underline"
        >
          Back to portfolio
        </Link>
      </div>
    </div>
  );
}
