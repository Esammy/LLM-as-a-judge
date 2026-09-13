# Measuring the judge

The premise of this project is that using an LLM to grade another LLM has
become standard practice, while *checking whether that judge is any good* has
not. This is the long form of what judgekit measures and why.

---

## The failure this exists to prevent

A team stands up an LLM judge. It produces a number. The number goes on a
dashboard. Decisions get made against it for months.

Then somebody finally sits down and scores fifty outputs by hand, and agreement
between the judge and the reviewers turns out to be poor. Every decision taken
on the strength of that number was taken on the strength of nothing — and worse
than nothing, because the number *looked* like evidence.

Nothing in that story requires anybody to be careless. The judge was plausible,
the number moved in believable ways, and no tool in the pipeline was in a
position to object.

---

## Four distinct ways a judge goes wrong

### 1. It cannot see the evidence

This is the one that motivated the project, and the one least discussed.

Ask a judge "is this figure invented?" while showing it only the answer, and it
cannot possibly know. So it guesses, and it guesses conservatively — correct
figures get marked as fabrications, and the suite reports a quality problem that
does not exist.

The fix is not a better prompt. It is showing the judge the tool calls and
retrieved context that produced the answer. In judgekit, `Evidence` is a
first-class field on a case and a first-class section in the prompt, not
something callers concatenate into a string.

There is a second-order point that matters as much. When no evidence was
captured at all, the honest verdict is **unverifiable**, not **wrong**. judgekit
raises a `no_evidence` flag rather than silently scoring low, because those are
different findings with different fixes: one is a model problem, the other is a
logging problem.

```
revenue-correct-but-unverifiable   3.18   fail   no_evidence
```

A correct answer. A low score. And a flag saying the score should not be trusted
on its own.

### 2. Position bias

Judges prefer whichever answer came first. In a pairwise comparison that alone
can decide the result.

judgekit measures it by scoring **identical content twice**, once presented as
the first candidate and once as the second. The content, rubric, evidence and
temperature are all held constant, so any difference in score is attributable to
slot order and nothing else.

The magnitude is reported in scale points:

```
ok position +0.000 scale points (n=8)
   the same answer scored +0.00 points differently between slots;
   0 of 8 cases moved at all
```

### 3. Verbosity bias

Judges reward length. Published measurements put the inflation at 15–30 points
of preference for the longer answer regardless of whether it is better.

The subtlety is in how you measure it. **Correlating length with score is
wrong**, because long answers frequently *are* better — that correlation mostly
rediscovers a real effect and would flag a perfectly calibrated judge as biased.

judgekit correlates length against the **residual**: judge score minus human
label. That answers the actual question — does the judge reward length *beyond
what the humans thought it was worth?*

### 4. Self-preference

Judges score outputs from their own model family higher. Measuring it needs
cases to record who generated them, in `metadata["generator_family"]`.

When that is missing, judgekit says so:

```
ok self_preference +0.000 scale points (n=0)
   needs cases from the judge's own family (stub) and from others, tagged in
   metadata['generator_family']; found 0 own and 0 other
```

This matters more than it looks. Returning a bare `0.0` would read as *"no
self-preference found"*. **"Not measured" and "measured as unbiased" are
completely different claims about a judge**, and collapsing them is how a tool
launders ignorance into reassurance.

---

## How the detectors are validated

Every detector is tested the same way: run it against a judge with no bias and
confirm it reports none, then inject a *known* amount into the stub provider and
confirm it is found.

| | neutral stub | bias injected |
| --- | --- | --- |
| position | +0.000, 0/8 moved | **+0.922 pts, 8/8 moved, first slot** |
| verbosity | −0.038 | **+0.563 correlation** |

That only works because the stub is neutral *by construction*, and getting there
took a real fix. The first version scored by recall against the reference, which
made it structurally verbosity-biased: a padded answer covers more reference
tokens simply by saying more, so it ranked padding above precision. A provider
biased by construction would have invalidated every one of these measurements.
The fix was an F1 over content tokens, where padding costs precision.

---

## Calibration

Bias tells you *how* a judge is wrong. Calibration tells you *whether to trust
it at all*, by comparing it against human labels on the same cases.

```
metric              value  reading
quadratic kappa     0.702  substantial
cohen kappa         0.360  exact matches only; harsh on ordinal scales
krippendorff alpha  0.759  interval reliability
spearman            0.927  does it rank cases the way humans do?
mean abs error       0.70  scale points
systematic offset   +0.06  positive means generous
exact agreement       50%
within one point      62%
```

### Why quadratic weighting is the headline

Plain Cohen's kappa asks only whether two raters matched *exactly*. On an ordinal
quality scale that is the wrong model: it punishes a judge that said 5 where a
human said 4 exactly as hard as one that said 5 where a human said 1.

The consequence is visible above. **The same judge on the same data scores 0.702
weighted and 0.360 plain** — "substantial" versus "fair". Plain kappa makes
usable judges look broken.

Quadratic weights are `(i−j)² / (k−1)²`, so a near miss costs little and a wild
one costs nearly everything.

### Agreement and accuracy are different questions

Spearman is 0.927 while exact agreement is 50%. The judge ranks cases almost
exactly as the humans do, while frequently landing on a different number.

Both are reported because **the fixes differ**:

- poor ranking, good agreement → the rubric is measuring the wrong thing
- good ranking, poor agreement → the threshold is misplaced

`systematic_offset` separates them directly. A judge sitting a full point low
ranks perfectly and agrees badly, and the fix is recalibrating a threshold, not
rewriting a rubric.

### What chance correction does to narrow datasets

If every human label is 4 or 5, agreeing by accident is easy, so kappa is harsh.
That is correct behaviour, not a bug — but it does mean **a dataset containing no
bad answers cannot tell you much about a judge.** If you want to know whether
yours can spot a bad answer, the dataset has to contain some.

### These bands are conventions, not laws

The "substantial" / "moderate" / "fair" labels come from Landis & Koch (1977).
They are useful for orientation and are no substitute for looking at where the
disagreements actually fall — which is what the confusion matrix is for.

```
human\judge     1     2     3     4     5
          1     0     1     1     0     0
          3     0     0     2     0     0
          4     0     0     1     1     0
          5     0     0     0     1     1
```

### Judges drift

A rubric that agreed with your reviewers in January may not in April. Published
guidance puts drift on a 60–90 day horizon, which makes calibration **recurring
work**, not a one-off. That is why the Kubernetes manifests ship a nightly
calibration CronJob rather than leaving it as something to remember.

---

## Turning it into a gate

```bash
judgekit calibrate datasets/example.jsonl -r rubrics/answer-quality.v2.yaml \
    --min-kappa 0.6
judgekit bias datasets/example.jsonl -r rubrics/answer-quality.v2.yaml \
    --threshold 0.15 --fail-on-bias
```

Both exit 1 when they fail, and both run in this repository's own CI against
this repository's own judge.

---

## Sources

- [LLM-as-judge evaluation guide — Openlayer](https://www.openlayer.com/blog/llm-as-judge-evaluation-guide)
- [LLM-as-Judge Best Practices 2026: Calibration, Bias, Cost](https://futureagi.com/blog/llm-as-judge-best-practices-2026/)
- [LLM-Judge Bias Mitigation (2026)](https://futureagi.com/blog/evaluating-llm-judge-bias-mitigation-2026/)
- [Self-Preference Bias in LLM-as-a-Judge (arXiv 2410.21819)](https://arxiv.org/pdf/2410.21819)
- Landis, J.R. & Koch, G.G. (1977), *The Measurement of Observer Agreement for Categorical Data*
