/**
 * Build-time configuration shared by server and client components.
 *
 * Kept separate from `lib/api.ts` because that module is server-only — it
 * holds the API base URL used for server-side fetches and must not be pulled
 * into a client bundle. A client component that needs one flag should not drag
 * the whole API client with it.
 */

/**
 * Whether this deployment asks the API for demonstration scores.
 *
 * The backend refuses to score with a model trained on synthetic data unless
 * the caller declares a demonstration context — there is no default-on and no
 * server-side override, so that a number from a simulation cannot appear
 * without someone having asked for one. This build is a demonstration
 * deployment and says so on every page, so it opts in; the opt-in lives here,
 * in one visible place.
 *
 * Set `NEXT_PUBLIC_DEMO_CONTEXT=false` to see what a deployment pointed at
 * real evidence would show: refusals with their reasons, and the evidence
 * review, completeness and provenance surfaces unaffected.
 */
export const DEMO_CONTEXT =
  (process.env.NEXT_PUBLIC_DEMO_CONTEXT ?? "true").toLowerCase() !== "false";
