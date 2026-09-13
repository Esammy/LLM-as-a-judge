"""The ``judgekit`` command line.

Exit codes are part of the contract, because most of these commands are meant
to run in CI:

``0``
    Everything passed.
``1``
    A gate failed - a rubric was edited in place, quality dropped below
    tolerance, a dataset is invalid. This is a real finding, not a crash.
``2``
    Typer's own usage error (bad flags, missing arguments).

Nothing here reaches the network unless a real provider is selected, and the
default provider is the deterministic stub.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from judgekit.core.bias import (
    measure_position_bias,
    measure_self_preference,
    measure_verbosity_bias,
)
from judgekit.core.calibration import calibrate_run
from judgekit.core.errors import JudgekitError
from judgekit.core.judge import Judge
from judgekit.core.models import Dataset, Verdict
from judgekit.core.report import load_json, render_html, render_json
from judgekit.core.rubric import (
    Rubric,
    load_rubrics,
    read_lock,
    verify_lock,
    write_lock,
)
from judgekit.core.runner import Runner, RunResult, compare
from judgekit.providers.stub import StubProvider

app = typer.Typer(
    name="judgekit",
    help="Measure the judge, not just the model.",
    no_args_is_help=True,
    add_completion=False,
)
rubric_app = typer.Typer(
    name="rubric", help="Inspect, lock and verify rubrics.", no_args_is_help=True
)
app.add_typer(rubric_app)

console = Console()
err_console = Console(stderr=True)

GATE_FAILED = 1

DatasetArg = Annotated[Path, typer.Argument(help="Dataset file (.jsonl, .json or .yaml).")]
RubricOpt = Annotated[Path, typer.Option("--rubric", "-r", help="Rubric file to grade under.")]
RubricDirOpt = Annotated[Path, typer.Option("--dir", "-d", help="Directory holding rubrics.")]


def _fail(message: str) -> None:
    """Report a gate failure and exit 1.

    The message is escaped because Rich parses square brackets as markup.
    An unescaped "[edited-without-bump]" is read as a style tag and
    silently dropped, which quietly removes the most useful word in a CI
    log at exactly the moment someone is reading it.
    """
    err_console.print(f"[bold red]x[/bold red] {escape(message)}")
    raise typer.Exit(GATE_FAILED)


def _verdict_style(verdict: Verdict) -> str:
    return {"pass": "green", "fail": "red", "error": "yellow"}[verdict.value]


def _print_summary(result: RunResult, dataset: Dataset) -> None:
    table = Table(show_header=True, header_style="dim", box=None, padding=(0, 2, 0, 0))
    table.add_column("case")
    table.add_column("score", justify="right")
    table.add_column("verdict")
    table.add_column("flags", style="dim")

    for j in result.judgements:
        table.add_row(
            escape(j.case_id),
            f"{j.score:.2f}",
            f"[{_verdict_style(j.verdict)}]{j.verdict.value}[/]",
            ", ".join(f.value for f in j.flags),
        )

    console.print(table)
    console.print()
    console.print(
        f"[bold]{result.pass_rate:.0%} passed[/bold] "
        f"({result.passed}/{result.total}), mean {result.mean_score:.2f}"
        + (f", [yellow]{result.errored} errored[/yellow]" if result.errored else "")
    )
    # Always printed, never optional: a score without its rubric is not a
    # measurement, and the fingerprint is what makes two runs comparable.
    console.print(
        f"[dim]{result.rubric.id}@{result.rubric.version} "
        f"({result.rubric.fingerprint[:12]}) - {result.provider}/{result.model} "
        f"- {result.duration_seconds:.2f}s - {len(dataset)} cases[/dim]"
    )


@app.command()
def run(
    dataset_path: DatasetArg,
    rubric_path: RubricOpt,
    out: Annotated[
        Path | None, typer.Option("--out", "-o", help="Write a self-contained HTML report here.")
    ] = None,
    save: Annotated[
        Path | None, typer.Option("--save", help="Write run JSON here, for a later compare.")
    ] = None,
    concurrency: Annotated[int, typer.Option(help="Cases in flight at once.")] = 8,
    min_pass_rate: Annotated[
        float | None,
        typer.Option("--min-pass-rate", help="Exit 1 if the pass rate falls below this (0-1)."),
    ] = None,
    domain: Annotated[str | None, typer.Option(help="Only run cases in this domain.")] = None,
) -> None:
    """Judge a dataset and report the results."""
    try:
        dataset = Dataset.from_file(dataset_path)
        rubric = Rubric.from_file(rubric_path)
    except (JudgekitError, FileNotFoundError, ValueError) as exc:
        _fail(str(exc))
        return

    if domain:
        dataset = dataset.filter(domain=domain)
        if not len(dataset):
            _fail(f"no cases in domain {domain!r}")
            return

    judge = Judge(rubric, StubProvider())
    result = asyncio.run(Runner(judge, concurrency=concurrency).run(dataset))

    _print_summary(result, dataset)

    if out:
        out.write_text(
            render_html(result, domains={c.id: c.domain for c in dataset.cases}),
            encoding="utf-8",
        )
        console.print(f"[dim]report written to {out}[/dim]")

    if save:
        save.write_text(render_json(result), encoding="utf-8")
        console.print(f"[dim]run saved to {save}[/dim]")

    if min_pass_rate is not None and result.pass_rate < min_pass_rate:
        _fail(f"pass rate {result.pass_rate:.1%} is below the required {min_pass_rate:.1%}")


@app.command()
def validate(dataset_path: DatasetArg) -> None:
    """Check that a dataset loads and report what is in it."""
    try:
        dataset = Dataset.from_file(dataset_path)
    except (FileNotFoundError, ValueError) as exc:
        _fail(str(exc))
        return

    labelled = len(dataset.labelled)
    with_evidence = sum(1 for c in dataset.cases if not c.evidence.is_empty)
    with_reference = sum(1 for c in dataset.cases if c.reference)

    console.print(f"[green]ok[/green] {dataset.name}@{dataset.version} - {len(dataset)} cases")
    console.print(f"  {with_reference} with a reference answer")
    console.print(f"  {with_evidence} with evidence")
    console.print(f"  {labelled} with human labels")

    if not labelled:
        # Not an error. Judging works without labels; measuring the judge does not.
        console.print(
            "[yellow]note[/yellow] no human labels, so `judgekit calibrate` "
            "cannot measure this judge against anything."
        )


@app.command("compare")
def compare_runs(
    baseline: Annotated[Path, typer.Argument(help="Baseline run JSON.")],
    candidate: Annotated[Path, typer.Argument(help="Candidate run JSON.")],
    max_drop: Annotated[
        float, typer.Option("--max-drop", help="Pass-rate drop to tolerate (0-1).")
    ] = 0.0,
) -> None:
    """Diff two saved runs. Refuses if they are not comparable."""
    try:
        before = load_json(baseline.read_text(encoding="utf-8"))
        after = load_json(candidate.read_text(encoding="utf-8"))
        result = compare(before, after)
    except JudgekitError as exc:
        _fail(str(exc))
        return
    except (OSError, ValueError) as exc:
        _fail(f"could not read runs: {exc}")
        return

    arrow = "+" if result.pass_rate_delta >= 0 else ""
    console.print(
        f"pass rate {result.pass_rate_before:.0%} -> {result.pass_rate_after:.0%} "
        f"({arrow}{result.pass_rate_delta:.1%})"
    )
    console.print(f"mean score {result.mean_score_before:.2f} -> {result.mean_score_after:.2f}")

    for delta in result.regressions:
        console.print(
            f"  [red]regressed[/red] {delta.case_id}: {delta.before:.2f} -> {delta.after:.2f}"
        )
    for delta in result.improvements:
        console.print(
            f"  [green]improved[/green] {delta.case_id}: {delta.before:.2f} -> {delta.after:.2f}"
        )

    if result.only_in_baseline or result.only_in_candidate:
        console.print(
            f"[yellow]note[/yellow] {len(result.only_in_baseline)} case(s) only in the "
            f"baseline, {len(result.only_in_candidate)} only in the candidate"
        )

    if not result.gate(max_pass_rate_drop=max_drop):
        _fail(f"pass rate dropped {-result.pass_rate_delta:.1%}, beyond the {max_drop:.1%} allowed")


@rubric_app.command("list")
def rubric_list(directory: RubricDirOpt = Path("rubrics")) -> None:
    """List the rubrics in a directory with their fingerprints."""
    try:
        rubrics = load_rubrics(directory)
    except JudgekitError as exc:
        _fail(str(exc))
        return

    table = Table(show_header=True, header_style="dim", box=None, padding=(0, 2, 0, 0))
    table.add_column("rubric")
    table.add_column("fingerprint", style="dim")
    table.add_column("criteria", justify="right")
    table.add_column("evidence")

    for key, rubric in sorted(rubrics.items()):
        table.add_row(
            key,
            rubric.fingerprint[:16],
            str(len(rubric.criteria)),
            "yes" if rubric.requires_evidence else "no",
        )
    console.print(table)


@rubric_app.command("lock")
def rubric_lock(directory: RubricDirOpt = Path("rubrics")) -> None:
    """Record the current rubric fingerprints in the lockfile."""
    try:
        rubrics = load_rubrics(directory)
    except JudgekitError as exc:
        _fail(str(exc))
        return

    path = write_lock(directory, rubrics)
    console.print(f"[green]ok[/green] locked {len(rubrics)} rubric(s) in {path}")


@rubric_app.command("verify")
def rubric_verify(directory: RubricDirOpt = Path("rubrics")) -> None:
    """Fail if any rubric changed without a version bump. Run this in CI.

    This is the guard the whole project rests on. A published rubric edited in
    place makes every score recorded on either side of the edit quietly
    incomparable, and nothing else in a normal pipeline would notice.
    """
    try:
        rubrics = load_rubrics(directory)
        violations = verify_lock(rubrics, read_lock(directory))
    except JudgekitError as exc:
        _fail(str(exc))
        return

    if not violations:
        console.print(f"[green]ok[/green] {len(rubrics)} rubric(s) match the lockfile")
        return

    for violation in violations:
        err_console.print(f"  {escape(str(violation))}")
    _fail(f"{len(violations)} rubric violation(s)")


@rubric_app.command("diff")
def rubric_diff(
    left: Annotated[Path, typer.Argument(help="First rubric file.")],
    right: Annotated[Path, typer.Argument(help="Second rubric file.")],
) -> None:
    """Show what differs between two rubrics, and whether scores can be compared."""
    try:
        a = Rubric.from_file(left)
        b = Rubric.from_file(right)
    except JudgekitError as exc:
        _fail(str(exc))
        return

    console.print(f"[bold]{a.id}@{a.version}[/bold] vs [bold]{b.id}@{b.version}[/bold]")

    if a.fingerprint == b.fingerprint:
        console.print("[green]identical[/green] - same fingerprint, scores are comparable")
        return

    fields: list[tuple[str, object, object]] = [
        ("requires_evidence", a.requires_evidence, b.requires_evidence),
        ("scale", a.scale.model_dump(), b.scale.model_dump()),
        ("criteria", [c.id for c in a.criteria], [c.id for c in b.criteria]),
        ("weights", {c.id: c.weight for c in a.criteria}, {c.id: c.weight for c in b.criteria}),
    ]
    for name, left_value, right_value in fields:
        if left_value != right_value:
            console.print(
                f"  [yellow]{name}[/yellow]: "
                f"{escape(str(left_value))} -> {escape(str(right_value))}"
            )

    if a.instructions.strip() != b.instructions.strip():
        console.print("  [yellow]instructions[/yellow]: changed")

    console.print(
        "\n[red]not comparable[/red] - scores under these two rubrics measure different things"
    )


@app.command()
def calibrate(
    dataset_path: DatasetArg,
    rubric_path: RubricOpt,
    min_kappa: Annotated[
        float | None,
        typer.Option("--min-kappa", help="Exit 1 if quadratic kappa falls below this."),
    ] = None,
    concurrency: Annotated[int, typer.Option(help="Cases in flight at once.")] = 8,
) -> None:
    """Measure the judge against the dataset's human labels.

    This is the command the project is named for. Everything else grades
    answers; this grades the grader.
    """
    try:
        dataset = Dataset.from_file(dataset_path)
        rubric = Rubric.from_file(rubric_path)
    except (JudgekitError, FileNotFoundError, ValueError) as exc:
        _fail(str(exc))
        return

    if not dataset.labelled:
        _fail(
            f"{dataset.name} has no human labels, so there is nothing to measure "
            "the judge against. Add `human_label` to some cases first."
        )
        return

    provider = StubProvider()
    result = asyncio.run(Runner(Judge(rubric, provider), concurrency=concurrency).run(dataset))

    try:
        report = calibrate_run(result.judgements, dataset, scale=rubric.scale)
    except ValueError as exc:
        _fail(str(exc))
        return

    table = Table(show_header=True, header_style="dim", box=None, padding=(0, 2, 0, 0))
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_column("reading", style="dim")

    rows: list[tuple[str, str, str]] = [
        ("quadratic kappa", f"{report.quadratic_kappa:.3f}", report.interpretation),
        (
            "cohen kappa",
            f"{report.cohens_kappa:.3f}",
            "exact matches only; harsh on ordinal scales",
        ),
        ("krippendorff alpha", f"{report.krippendorff_alpha:.3f}", "interval reliability"),
        ("spearman", f"{report.spearman:.3f}", "does it rank cases the way humans do?"),
        ("pearson", f"{report.pearson:.3f}", ""),
        ("mean abs error", f"{report.mean_absolute_error:.2f}", "scale points"),
        ("systematic offset", f"{report.systematic_offset:+.2f}", "positive means generous"),
        ("exact agreement", f"{report.exact_agreement:.0%}", ""),
        ("within one point", f"{report.within_one:.0%}", ""),
    ]
    for name, value, reading in rows:
        table.add_row(name, value, escape(reading))

    console.print(table)
    console.print()
    console.print("[dim]confusion matrix (rows: human, columns: judge)[/dim]")
    console.print(escape(report.confusion.render()))
    console.print()
    console.print(
        f"[dim]n={report.n} labelled cases, judge mean {report.judge_mean:.2f} "
        f"vs human mean {report.human_mean:.2f}[/dim]"
    )

    if report.systematic_offset and abs(report.systematic_offset) > 0.5:
        # Worth separating from a ranking problem: they have different fixes.
        direction = "generous" if report.systematic_offset > 0 else "harsh"
        console.print(
            f"[yellow]note[/yellow] the judge is systematically {direction} by "
            f"{abs(report.systematic_offset):.2f} points. If Spearman is high, this is a "
            "threshold problem rather than a rubric problem."
        )

    if min_kappa is not None and not report.meets(min_kappa=min_kappa):
        _fail(f"quadratic kappa {report.quadratic_kappa:.3f} is below the required {min_kappa:.3f}")


@app.command()
def bias(
    dataset_path: DatasetArg,
    rubric_path: RubricOpt,
    threshold: Annotated[
        float, typer.Option(help="Magnitude above which a bias is called concerning.")
    ] = 0.15,
    fail_on_bias: Annotated[
        bool, typer.Option("--fail-on-bias", help="Exit 1 if any bias exceeds the threshold.")
    ] = False,
) -> None:
    """Measure position, verbosity and self-preference bias in the judge."""
    try:
        dataset = Dataset.from_file(dataset_path)
        rubric = Rubric.from_file(rubric_path)
    except (JudgekitError, FileNotFoundError, ValueError) as exc:
        _fail(str(exc))
        return

    judge = Judge(rubric, StubProvider())
    result = asyncio.run(Runner(judge).run(dataset))

    findings = [
        asyncio.run(measure_position_bias(judge, dataset.cases, threshold=threshold)),
        measure_verbosity_bias(result.judgements, dataset, scale=rubric.scale, threshold=threshold),
        measure_self_preference(
            result.judgements, dataset, judge_family=judge.provider.family, threshold=threshold
        ),
    ]

    for finding in findings:
        colour = "red" if finding.concerning else "green"
        mark = "!" if finding.concerning else "ok"
        console.print(
            f"[{colour}]{mark}[/{colour}] [bold]{finding.kind}[/bold] "
            f"{finding.magnitude:+.3f} {finding.unit} (n={finding.n})"
        )
        console.print(f"    [dim]{escape(finding.detail)}[/dim]")

    concerning = [f for f in findings if f.concerning]
    if concerning and fail_on_bias:
        _fail(f"{len(concerning)} bias measurement(s) exceed the {threshold} threshold")


if __name__ == "__main__":  # pragma: no cover
    app()
