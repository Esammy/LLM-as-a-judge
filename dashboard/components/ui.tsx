import Link from "next/link";
import type { Judgement, RunSummary } from "@/lib/api";
import { formatPercent, formatWhen } from "@/lib/api";

export function Panel({
  children,
  className = "",
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div
      className={`rounded-xl border border-[var(--color-line)] bg-[var(--color-panel)] ${className}`}
    >
      {children}
    </div>
  );
}

export function Stat({ value, label }: { value: string; label: string }) {
  return (
    <Panel className="px-4 py-3">
      <div className="tabular text-2xl font-semibold tracking-tight">{value}</div>
      <div className="text-xs uppercase tracking-wide text-[var(--color-muted)]">{label}</div>
    </Panel>
  );
}

export function Verdict({ verdict }: { verdict: Judgement["verdict"] }) {
  const colour = {
    pass: "var(--color-pass)",
    fail: "var(--color-fail)",
    error: "var(--color-warn)",
  }[verdict];

  return (
    <span className="font-medium" style={{ color: colour }}>
      {verdict}
    </span>
  );
}

export function Status({ status }: { status: RunSummary["status"] }) {
  const colour = {
    completed: "var(--color-pass)",
    failed: "var(--color-fail)",
    running: "var(--color-mint)",
    queued: "var(--color-muted)",
  }[status];

  return (
    <span className="text-sm font-medium" style={{ color: colour }}>
      {status}
    </span>
  );
}

/**
 * A flag is a reason to distrust a score, so it is rendered next to the score
 * rather than tucked away. `no_evidence` in particular means "unverifiable",
 * which is a different finding from "wrong" - the whole point of surfacing it.
 */
export function Flags({ flags }: { flags: string[] }) {
  if (flags.length === 0) return null;
  return (
    <span className="inline-flex flex-wrap gap-1">
      {flags.map((flag) => (
        <span
          key={flag}
          className="rounded bg-[var(--color-paper)] px-1.5 py-0.5 font-mono text-[11px] text-[var(--color-muted)]"
        >
          {flag}
        </span>
      ))}
    </span>
  );
}

export function Fingerprint({ value }: { value: string }) {
  return (
    <code
      title={value}
      className="font-mono text-xs text-[var(--color-muted)]"
    >
      {value.slice(0, 12)}
    </code>
  );
}

/**
 * A pass-rate trend.
 *
 * Every point comes from `/runs/{id}/history`, which the API scopes to one
 * dataset and one rubric fingerprint. That scoping is the reason this chart is
 * safe to draw at all: a quality line that silently spans a rubric change looks
 * like evidence and is not.
 *
 * Hand-rolled SVG rather than a charting library - it is one polyline, and a
 * dependency for that would be a poor trade.
 */
export function Trend({ runs }: { runs: RunSummary[] }) {
  const points = [...runs].reverse();
  if (points.length < 2) {
    return (
      <p className="px-4 py-6 text-sm text-[var(--color-muted)]">
        Not enough comparable runs yet. A trend needs at least two runs of the same
        dataset under the same rubric fingerprint.
      </p>
    );
  }

  const width = 640;
  const height = 140;
  const pad = 16;

  const coords = points.map((run, index) => {
    const x = pad + (index * (width - 2 * pad)) / (points.length - 1);
    const y = height - pad - run.pass_rate * (height - 2 * pad);
    return { x, y, run };
  });

  const path = coords.map((c) => `${c.x.toFixed(1)},${c.y.toFixed(1)}`).join(" ");

  return (
    <div className="overflow-x-auto px-4 py-4">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="h-[140px] w-full min-w-[480px]"
        role="img"
        aria-label={`Pass rate across ${points.length} comparable runs`}
      >
        {[0, 0.5, 1].map((level) => {
          const y = height - pad - level * (height - 2 * pad);
          return (
            <g key={level}>
              <line
                x1={pad}
                x2={width - pad}
                y1={y}
                y2={y}
                stroke="var(--color-line)"
                strokeWidth={1}
              />
              <text x={0} y={y + 3} fontSize={9} fill="var(--color-muted)">
                {level * 100}
              </text>
            </g>
          );
        })}
        <polyline
          points={path}
          fill="none"
          stroke="var(--color-mint)"
          strokeWidth={2}
          strokeLinejoin="round"
        />
        {coords.map((c) => (
          <circle key={c.run.run_id} cx={c.x} cy={c.y} r={3} fill="var(--color-mint)">
            <title>
              {`${formatWhen(c.run.created_at)} - ${formatPercent(c.run.pass_rate)}`}
            </title>
          </circle>
        ))}
      </svg>
    </div>
  );
}

export function RunRow({ run }: { run: RunSummary }) {
  return (
    <tr className="border-b border-[var(--color-line)] last:border-0">
      <td className="px-4 py-2">
        <Link href={`/runs/${run.run_id}`} className="font-mono text-xs hover:underline">
          {run.run_id}
        </Link>
      </td>
      <td className="px-4 py-2 text-sm">{run.dataset}</td>
      <td className="px-4 py-2 text-sm">
        {run.rubric.id}@{run.rubric.version}{" "}
        <Fingerprint value={run.rubric.fingerprint} />
      </td>
      <td className="px-4 py-2">
        <Status status={run.status} />
      </td>
      <td className="tabular px-4 py-2 text-right text-sm">
        {run.status === "completed" ? formatPercent(run.pass_rate) : "-"}
      </td>
      <td className="tabular px-4 py-2 text-right text-sm">
        {run.status === "completed" ? run.mean_score.toFixed(2) : "-"}
      </td>
      <td className="px-4 py-2 text-right text-xs text-[var(--color-muted)]">
        {formatWhen(run.created_at)}
      </td>
    </tr>
  );
}

export function ApiDown({ detail }: { detail: string }) {
  return (
    <Panel className="px-5 py-6">
      <h2 className="text-base font-semibold">The API is not reachable</h2>
      <p className="mt-2 max-w-2xl text-sm text-[var(--color-muted)]">
        The dashboard reads from the judgekit API. Start it with{" "}
        <code className="font-mono">make api</code>, or bring the whole stack up with{" "}
        <code className="font-mono">docker compose -f deploy/docker-compose.yml up</code>.
      </p>
      <pre className="mt-3 overflow-x-auto rounded bg-[var(--color-paper)] p-3 font-mono text-xs text-[var(--color-muted)]">
        {detail}
      </pre>
    </Panel>
  );
}
