"""Exception hierarchy.

Every error judgekit raises inherits from :class:`JudgekitError`, so a caller
embedding the library can catch exactly one thing.
"""

from __future__ import annotations


class JudgekitError(Exception):
    """Base class for every error raised by judgekit."""


class RubricError(JudgekitError):
    """A rubric could not be loaded, parsed or validated."""


class IncomparableScoresError(JudgekitError):
    """Two results were compared that are not legitimately comparable.

    Raised when scores graded under different rubrics, different rubric
    versions, or the same version with different content are put side by side.
    This is a deliberate hard failure: silently comparing them produces a
    number that looks meaningful and is not.
    """


class RubricLockError(JudgekitError):
    """The rubric lockfile does not match the rubrics on disk."""


class ProviderError(JudgekitError):
    """A model provider failed."""


class JudgeParseError(JudgekitError):
    """The judge returned something that could not be read as a judgement."""
