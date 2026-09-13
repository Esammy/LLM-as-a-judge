import Link from "next/link";
import { notFound } from "next/navigation";
import { ApiError, formatPercent, formatWhen, getHistory, getRun } from "@/lib/api";
import { ApiDown, Flags, Fingerprint, Panel, Stat, Trend, Verdict } from "@/components/ui";

export const dynamic = "force-dynamic";

export default async function RunPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;

  let run;
  let history;
  try {
    run = await getRun(id);
    history = (await getHistory(id)).runs;
  } catch (error) {
    if (error instanceof ApiError && error.status === 404) notFound();
    return <ApiDown detail={error instanceof Error ? error.message : String(error)} />;
  }

  return (
    <div className="space-y-8">
      <header>
        <Link href="/" className="text-sm text-[var(--color-muted)] hover:underline">
          &larr; all runs
        </Link>
        <h1 className="mt-2 font-mono text-xl font-semibold tracking-tight">{run.run_id}</h1>
        <p className="mt-1 text-sm text-[var(--color-muted)]">
          {run.dataset} graded under {run.rubric.id}@{run.rubric.version}{" "}
          <Fingerprint value={run.rubric.fingerprint} /> &middot; {run.provider}/{run.model}{" "}
          &middot; {formatWhen(run.created_at)}
        </p>
      </header>

      {run.status === "failed" && run.error && (
        <Panel className="border-l-4 px-4 py-3" >
          <p className="text-sm font-medium" style={{ color: "var(--color-fail)" }}>
            This run failed
          </p>
          <pre className="mt-2 overflow-x-auto font-mono text-xs text-[var(--color-muted)]">
            {run.error}
          </pre>
        </Panel>
      )}

      <section className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
        <Stat value={formatPercent(run.pass_rate)} label="pass rate" />
        <Stat value={run.mean_score.toFixed(2)} label="mean score" />
        <Stat value={String(run.total)} label="cases" />
        <Stat value={String(run.passed)} label="passed" />
        <Stat value={String(run.failed)} label="failed" />
        <Stat value={String(run.errored)} label="errored" />
      </section>

      {run.errored > 0 && (
        <p className="text-sm text-[var(--color-muted)]">
          {run.errored} case(s) errored. These are excluded from the mean score - an
          unparseable response is a missing measurement, not a bad one - but they still
          count against the pass rate, because a run that crashed did not pass.
        </p>
      )}

      <section>
        <h2 className="mb-2 text-sm font-semibold uppercase tracking-wide text-[var(--color-muted)]">
          Pass rate over comparable runs
        </h2>
        <Panel>
          <Trend runs={history} />
        </Panel>
        <p className="mt-2 text-xs text-[var(--color-muted)]">
          Only runs of <span className="font-mono">{run.dataset}</span> graded under
          fingerprint <Fingerprint value={run.rubric.fingerprint} /> appear here. Scores
          from a different rubric are not comparable, so plotting them on one line would
          produce something that looks like evidence and is not.
        </p>
      </section>

      <section>
        <h2 className="mb-2 text-sm font-semibold uppercase tracking-wide text-[var(--color-muted)]">
          Cases
        </h2>
        <Panel className="overflow-x-auto">
          <table className="w-full min-w-[760px] text-left">
            <thead>
              <tr className="border-b border-[var(--color-line)] text-xs uppercase tracking-wide text-[var(--color-muted)]">
                <th className="px-4 py-2 font-medium">Case</th>
                <th className="px-4 py-2 text-right font-medium">Score</th>
                <th className="px-4 py-2 font-medium">Verdict</th>
                <th className="px-4 py-2 font-medium">Flags</th>
                <th className="px-4 py-2 font-medium">Reasoning</th>
              </tr>
            </thead>
            <tbody>
              {run.judgements.map((judgement) => (
                <tr
                  key={judgement.case_id}
                  className="border-b border-[var(--color-line)] align-top last:border-0"
                >
                  <td className="px-4 py-2 font-mono text-xs">{judgement.case_id}</td>
                  <td className="tabular px-4 py-2 text-right text-sm">
                    {judgement.score.toFixed(2)}
                  </td>
                  <td className="px-4 py-2 text-sm">
                    <Verdict verdict={judgement.verdict} />
                  </td>
                  <td className="px-4 py-2">
                    <Flags flags={judgement.flags} />
                  </td>
                  <td className="max-w-md px-4 py-2 text-sm text-[var(--color-muted)]">
                    {judgement.error ?? judgement.reasoning}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </Panel>
      </section>
    </div>
  );
}
