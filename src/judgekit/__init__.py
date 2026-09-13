"""judgekit — measure the judge, not just the model.

Most evaluation tools score your model's output. This one treats the judge
itself as the thing under test: rubrics are version-pinned, judgements are
made against evidence rather than prose alone, and agreement with human
labels is a number you can fail a build on.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
