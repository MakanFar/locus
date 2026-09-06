"""Ranker-independent lexical hardness for a (context window, target) pair.

A context is *easy* when the answer is on the surface: the target's first-author
surname is written in the window, or enough of the target's title is. Defining
hardness via a ranker's own success would make "BM25 is at chance on hard" true
by construction.

The title measure is containment -- the fraction of the *title's* content words
present in the window -- not Jaccard. Jaccard divides by |window u title|, so
with ~36-token windows and ~6-token titles it measures window length as much as
title presence: a window containing the whole title scores only ~0.17, below
any threshold that is not itself near zero. Containment is invariant to window
length and is just as ranker-independent.
"""
import re

_WORD = re.compile(r"[a-z]+")
_STOP = frozenset([
    "a", "an", "the", "and", "or", "of", "for", "to", "in", "on", "at", "by", "with",
    "from", "as", "is", "are", "was", "were", "be", "been", "that", "this", "these",
    "those", "it", "its", "we", "our", "their", "his", "her", "they", "he", "she",
    "which", "who", "not", "but", "if", "then", "than", "so", "such", "can", "may",
    "might", "will", "would", "should", "could",
])


def _tokens(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP}


def surname(author) -> str:
    if isinstance(author, dict):
        author = author.get("name") or ""
    parts = str(author).strip().split()
    return parts[-1].lower() if parts else ""


def title_containment(window: str, title: str) -> float:
    """Fraction of the title's content words that appear in the window."""
    a, b = _tokens(window), _tokens(title)
    if not a or not b:
        return 0.0
    return len(a & b) / len(b)


def is_easy(
    window: str, title: str, authors: list | None, tau: float
) -> bool:
    """Is the answer on the surface of this window?

    `authors` may be `None` or absent. The JSONL corpus contract makes it
    optional (`locus.data.ingest`), so a paper dict assembled from a corpus
    that does not carry it must not crash here -- it simply loses the surname
    route and falls through to title containment. On Gu et al.'s test split
    that route is worth 2,239 of 55,488 rank items (hard 78.16% -> 82.20%),
    which is what `ingest`'s docstring warns a corpus without authors gives up.

    A non-list `authors` (e.g. the bare string "Yu, Pengqian") is refused
    rather than silently iterated: iterating a string tests whether "Y" is a
    token in the window, which is never true, so the route would be dead
    rather than absent and nothing would say so. `ingest.validate` rejects the
    same shape at corpus level.
    """
    if authors is not None and not isinstance(authors, (list, tuple)):
        raise TypeError(
            f"authors must be a list, not {type(authors).__name__}; a string "
            "would be iterated character by character and the first-author "
            "surname route would silently never fire"
        )
    win = _tokens(window)
    for author in (authors or ())[:1]:
        s = surname(author)
        if s and s in win:
            return True
    return title_containment(window, title) >= tau
