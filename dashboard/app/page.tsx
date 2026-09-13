import { listRuns } from "@/lib/api";
import { ApiDown, Panel, RunRow } from "@/components/ui";

export const dynamic = "force-dynamic";

export default async function RunsPage() {
  let runs;
  try {
    runs = (await listRuns({ limit: 50 })).runs;
  } catch (error) {
    return <ApiDown detail={error instanceof Error ? error.message : String(error)} />;
  }

  if (runs.length === 0) {
    return (
      <Panel className="px-5 py-6">
        <h2 className="text-base font-semibold">No runs yet</h2>
        <p className="mt-2 text-sm text-[var(--color-muted)]">Submit one:</p>
        <pre className="mt-3 overflow-x-auto rounded bg-[var(--color-paper)] p-3 font-mono text-xs">
{`curl -X POST localhost:8000/runs \\
  -H 'content-type: application/json' \\
  -d '{"dataset":"example.jsonl","rubric":"answer-quality.v2.yaml"}'`}
        </pre>
      </Panel>
    );
  }

  return (
    <section>
      <div className="mb-4 flex items-baseline justify-between">
        <h1 className="text-xl font-semibold tracking-tight">Runs</h1>
        <p className="text-sm text-[var(--color-muted)]">{runs.length} most recent</p>
      </div>

      <Panel className="overflow-x-auto">
        <table className="w-full min-w-[760px] text-left">
          <thead>
            <tr className="border-b border-[var(--color-line)] text-xs uppercase tracking-wide text-[var(--color-muted)]">
              <th className="px-4 py-2 font-medium">Run</th>
              <th className="px-4 py-2 font-medium">Dataset</th>
              {/* The fingerprint is shown in the list, not buried in a detail
                  view: two rows under the same rubric id and version but with
                  different fingerprints are not comparable, and that is exactly
                  the thing a reader needs to notice before they draw a
                  conclusion from the column to the right. */}
              <th className="px-4 py-2 font-medium">Rubric</th>
              <th className="px-4 py-2 font-medium">Status</th>
              <th className="px-4 py-2 text-right font-medium">Pass</th>
              <th className="px-4 py-2 text-right font-medium">Mean</th>
              <th className="px-4 py-2 text-right font-medium">When</th>
            </tr>
          </thead>
          <tbody>
            {runs.map((run) => (
              <RunRow key={run.run_id} run={run} />
            ))}
          </tbody>
        </table>
      </Panel>
    </section>
  );
}
