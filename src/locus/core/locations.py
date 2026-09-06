"""Cluster a citing paper's context windows into locations.

Windows are fixed-width character spans centred on a citation, so two markers
in the same sentence share ~90% of their window and two in different
paragraphs share none. Jaccard over word shingles, thresholded at THETA,
plus union-find. Reference numbers are stripped before shingling because the
same passage recurs with a different number in the target slot.
"""
import hashlib
import re

_WORD = re.compile(r"[a-z]+")


def _key(shingle: str) -> int:
    # Not builtin hash(): that is PYTHONHASHSEED-randomised for str, so two
    # processes can disagree about which shingles collide, and the index is
    # the root of every frozen artefact.
    return int.from_bytes(
        hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).digest(), "big"
    )


def shingles(text: str, n: int = 8) -> frozenset[int]:
    words = _WORD.findall(text.lower())
    if len(words) < n:
        return frozenset()
    return frozenset(
        _key(" ".join(words[i : i + n])) for i in range(len(words) - n + 1)
    )


def _jaccard(a: frozenset[int], b: frozenset[int]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / len(a | b)


def cluster(texts: list[str], theta: float = 0.5, n: int = 8) -> list[int]:
    m = len(texts)
    if m == 0:
        return []
    sh = [shingles(t, n) for t in texts]
    parent = list(range(m))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(m):
        for j in range(i + 1, m):
            if _jaccard(sh[i], sh[j]) >= theta:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[max(ri, rj)] = min(ri, rj)

    labels: list[int] = []
    seen: dict[int, int] = {}
    for i in range(m):
        r = find(i)
        if r not in seen:
            seen[r] = len(seen)
        labels.append(seen[r])
    return labels
