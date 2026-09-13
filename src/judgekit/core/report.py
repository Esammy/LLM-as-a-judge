"""Self-contained HTML reports.

The report is a single file with no external requests - no CDN, no webfont, no
analytics. That matters more than it sounds: eval results routinely contain
customer data, and a report that phones out is one that cannot be opened on a
laptop in a regulated environment or attached to a CI artifact without review.

It is also the artifact people actually look at, so it leads with the things
that are easy to get wrong - which rubric produced these numbers, how many
cases errored rather than failed, and which results are flagged as
untrustworthy - rather than burying them under a headline percentage.
"""

from __future__ import annotations

import html
import json
from collections.abc import Mapping

from judgekit.core.models import Verdict
from judgekit.core.runner import RunResult

_CSS = """
:root {
  --bg: #fbfbfa; --fg: #1a1a18; --muted: #6b6b66; --line: #e4e4df;
  --panel: #ffffff; --pass: #1f7a4d; --fail: #b3261e; --error: #8a5a00;
  --flag-bg: #f3f0e8; --accent: #2f6f5e;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #14140f; --fg: #ececdf; --muted: #9a9a8e; --line: #2e2e26;
    --panel: #1c1c16; --pass: #6fd39b; --fail: #ff8b80; --error: #e0b25e;
    --flag-bg: #2a2a20; --accent: #7fd0b6;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 32px 16px; background: var(--bg); color: var(--fg);
  font: 14px/1.55 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
}
.wrap { max-width: 1100px; margin: 0 auto; }
h1 { font-size: 22px; margin: 0 0 4px; letter-spacing: -0.01em; }
h2 { font-size: 15px; margin: 36px 0 12px; letter-spacing: 0.04em;
     text-transform: uppercase; color: var(--muted); font-weight: 600; }
.sub { color: var(--muted); margin: 0 0 28px; }
.cards { display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); }
.card { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 14px 16px; }
.card .v { font-size: 26px; font-weight: 650; letter-spacing: -0.02em; }
.card .l { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; }
.prov { background: var(--panel); border: 1px solid var(--line); border-radius: 10px;
        padding: 14px 16px; margin-top: 12px; }
.prov dl { display: grid; grid-template-columns: max-content 1fr; gap: 6px 18px; margin: 0; }
.prov dt { color: var(--muted); }
.prov dd { margin: 0; font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; font-size: 13px; }
.scroll { overflow-x: auto; border: 1px solid var(--line); border-radius: 10px; background: var(--panel); }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { text-align: left; padding: 9px 12px; border-bottom: 1px solid var(--line); vertical-align: top; }
th { color: var(--muted); font-weight: 600; font-size: 12px;
     text-transform: uppercase; letter-spacing: 0.04em; white-space: nowrap; }
tr:last-child td { border-bottom: 0; }
td.num { text-align: right; font-variant-numeric: tabular-nums;
         font-family: ui-monospace, Menlo, Consolas, monospace; white-space: nowrap; }
.id { font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 12px; }
.v-pass { color: var(--pass); font-weight: 600; }
.v-fail { color: var(--fail); font-weight: 600; }
.v-error { color: var(--error); font-weight: 600; }
.flag { display: inline-block; background: var(--flag-bg); border-radius: 4px;
        padding: 1px 6px; margin: 0 4px 2px 0; font-size: 11px;
        font-family: ui-monospace, Menlo, Consolas, monospace; }
.reason { color: var(--muted); max-width: 460px; }
.note { border-left: 3px solid var(--accent); padding: 10px 14px; margin: 14px 0;
        background: var(--panel); border-radius: 0 8px 8px 0; color: var(--muted); }
footer { margin-top: 40px; color: var(--muted); font-size: 12px;
         border-top: 1px solid var(--line); padding-top: 14px; }
"""


def _esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def _verdict_cell(verdict: Verdict) -> str:
    return f'<span class="v-{verdict.value}">{verdict.value}</span>'


def render_html(result: RunResult, *, domains: Mapping[str, str | None] | None = None) -> str:
    """Render a run as a standalone HTML document."""
    domains = domains or {}
    by_domain = result.by_domain(dict(domains))

    cards = [
        ("pass rate", f"{result.pass_rate:.0%}"),
        ("mean score", f"{result.mean_score:.2f}"),
        ("cases", str(result.total)),
        ("passed", str(result.passed)),
        ("failed", str(result.failed)),
        ("errored", str(result.errored)),
    ]
    cards_html = "".join(
        f'<div class="card"><div class="v">{_esc(v)}</div><div class="l">{_esc(label)}</div></div>'
        for label, v in cards
    )

    # Provenance sits near the top on purpose. A pass rate without the rubric
    # that produced it is not a measurement, it is a number.
    provenance = [
        ("dataset", f"{result.dataset_name} @ {result.dataset_version}"),
        ("rubric", f"{result.rubric.id} @ {result.rubric.version}"),
        ("fingerprint", result.rubric.fingerprint),
        ("provider", f"{result.provider} ({result.model})"),
        ("run id", result.run_id),
        ("started", result.started_at.isoformat(timespec="seconds")),
        ("duration", f"{result.duration_seconds:.2f}s"),
        ("tokens", f"{result.usage.total_tokens:,}"),
        ("cost", f"${result.usage.cost_usd:.4f}"),
    ]
    prov_html = "".join(f"<dt>{_esc(k)}</dt><dd>{_esc(v)}</dd>" for k, v in provenance)

    error_note = ""
    if result.errored:
        error_note = (
            f'<div class="note"><strong>{result.errored} case(s) errored.</strong> '
            "These are excluded from the mean score - an unparseable response is a "
            "missing measurement, not a bad one - but they still count against the "
            "pass rate, because a run that crashed did not pass.</div>"
        )

    domain_rows = "".join(
        f"<tr><td>{_esc(name)}</td>"
        f'<td class="num">{s.total}</td>'
        f'<td class="num">{s.passed}</td>'
        f'<td class="num">{s.failed}</td>'
        f'<td class="num">{s.errored}</td>'
        f'<td class="num">{s.pass_rate:.0%}</td>'
        f'<td class="num">{s.mean_score:.2f}</td></tr>'
        for name, s in by_domain.items()
    )
    domain_section = (
        f"""<h2>By domain</h2><div class="scroll"><table>
<thead><tr><th>Domain</th><th>Cases</th><th>Passed</th><th>Failed</th>
<th>Errored</th><th>Pass rate</th><th>Mean</th></tr></thead>
<tbody>{domain_rows}</tbody></table></div>"""
        if by_domain
        else ""
    )

    case_rows = []
    for j in result.judgements:
        flags = "".join(f'<span class="flag">{_esc(f.value)}</span>' for f in j.flags) or ""
        detail = j.error if j.error else j.reasoning
        case_rows.append(
            f'<tr><td class="id">{_esc(j.case_id)}</td>'
            f'<td class="num">{j.score:.2f}</td>'
            f"<td>{_verdict_cell(j.verdict)}</td>"
            f"<td>{flags}</td>"
            f'<td class="reason">{_esc(detail)}</td>'
            f'<td class="num">{j.elapsed_ms:.0f}ms</td></tr>'
        )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>judgekit - {_esc(result.dataset_name)} @ {_esc(result.rubric.version)}</title>
<style>{_CSS}</style></head>
<body><div class="wrap">
<h1>{_esc(result.dataset_name)}</h1>
<p class="sub">Graded under {_esc(result.rubric.id)} @ {_esc(result.rubric.version)}</p>
<div class="cards">{cards_html}</div>
{error_note}
<div class="prov"><dl>{prov_html}</dl></div>
{domain_section}
<h2>Cases</h2>
<div class="scroll"><table>
<thead><tr><th>Case</th><th>Score</th><th>Verdict</th><th>Flags</th>
<th>Reasoning</th><th>Time</th></tr></thead>
<tbody>{"".join(case_rows)}</tbody></table></div>
<footer>Generated by judgekit. Scores are comparable only against runs under
rubric fingerprint <code>{_esc(result.rubric.fingerprint[:16])}</code>.</footer>
</div></body></html>
"""


def render_json(result: RunResult) -> str:
    """Serialise a run for storage or a later ``judgekit compare``."""
    return result.model_dump_json(indent=2)


def load_json(text: str) -> RunResult:
    """Read back a run written by :func:`render_json`."""
    return RunResult.model_validate(json.loads(text))
