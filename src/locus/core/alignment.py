"""Align a masked context against its raw text to recover reference numbers.

`masked_text` replaces each citation marker with TARGETCIT/OTHERCIT, erasing
the reference number. Walking the literal segments between markers through
`raw` recovers what each marker stood for. Returns None rather than guessing
when the two strings cannot be reconciled.
"""
import collections
import re
from dataclasses import dataclass

_SPLIT = re.compile(r"(TARGETCIT|OTHERCIT)")
_NUM = re.compile(r"\d+")

MAX_PLAUSIBLE_REF = 1000
"""Reference numbers above this are almost certainly scraped prose (years,
sample counts, equation numbers) rather than citations. Papers with >1000
references exist but are vanishingly rare; miscounting those is preferable to
letting a four-digit year through as an anchor."""


def markers_with_numbers(raw: str, masked: str) -> list[tuple[str, str]] | None:
    parts = _SPLIT.split(masked)
    out: list[tuple[str, str]] = []
    pending: list[str] = []
    pos = 0
    for idx, part in enumerate(parts):
        if idx % 2 == 1:
            pending.append(part)
            continue
        seg = part.strip()
        if not seg:
            continue
        nxt = raw.find(seg, pos)
        if nxt < 0:
            return None
        if pending:
            nums = _NUM.findall(raw[pos:nxt])
            if len(nums) != len(pending):
                return None
            out.extend(zip(pending, nums, strict=True))
            pending = []
        pos = nxt + len(seg)
    if pending:
        # trailing markers: consume numbers from the remainder of raw
        nums = _NUM.findall(raw[pos:])
        if len(nums) != len(pending):
            return None
        out.extend(zip(pending, nums, strict=True))
    return out


@dataclass(frozen=True)
class Bibliography:
    number_to_ref: dict[str, str]
    conflicted_numbers: frozenset[str]
    n_entries: int


@dataclass(frozen=True)
class AnchorResolution:
    target_valid: bool
    anchors: frozenset[str]
    n_unresolvable: int
    n_implausible: int
    n_self: int = 0
    """OTHERCITs that resolve to the context's own target. Not anchors, but
    counted so that anchors + n_self + n_unresolvable + n_implausible
    partitions every OTHERCIT and the build report's accounting sums."""


def _plausible(num: str) -> bool:
    return num.isdigit() and int(num) <= MAX_PLAUSIBLE_REF


def build_bibliography(
    entries: list[tuple[str, list[tuple[str, str]]]],
) -> Bibliography:
    """Reconstruct one citing paper's number -> refid map from its TARGETCIT slots.

    `refid` comes from the dataset, not from alignment, so this is not circular:
    a number that denotes two different papers -- or a paper appearing under two
    different numbers -- proves at least one alignment is wrong.
    """
    num_refs: dict[str, set[str]] = collections.defaultdict(set)
    ref_nums: dict[str, set[str]] = collections.defaultdict(set)
    for own_refid, markers in entries:
        targets = [n for typ, n in markers if typ == "TARGETCIT"]
        if len(targets) != 1:
            continue  # cannot anchor a number to this refid
        if not _plausible(targets[0]):
            # Same filter resolve() applies to OTHERCIT. Without it a scraped
            # year can become a trusted bibliography key while the identical
            # number is rejected as implausible on the anchor side.
            continue
        num_refs[targets[0]].add(own_refid)
        ref_nums[own_refid].add(targets[0])

    conflicted = {n for n, refs in num_refs.items() if len(refs) > 1}
    for nums in ref_nums.values():
        if len(nums) > 1:
            conflicted |= nums

    mapping = {
        n: next(iter(refs))
        for n, refs in num_refs.items()
        if n not in conflicted
    }
    return Bibliography(
        number_to_ref=mapping,
        conflicted_numbers=frozenset(conflicted),
        n_entries=len(mapping),
    )


def resolve(
    markers: list[tuple[str, str]],
    bib: Bibliography,
    own_refid: str,
) -> AnchorResolution:
    """Validate a context's markers against its paper's reconstructed bibliography."""
    targets = [n for typ, n in markers if typ == "TARGETCIT"]
    target_valid = (
        len(targets) == 1
        and targets[0] not in bib.conflicted_numbers
        and bib.number_to_ref.get(targets[0]) == own_refid
    )

    anchors: set[str] = set()
    n_unresolvable = 0
    n_implausible = 0
    n_self = 0
    for typ, num in markers:
        if typ != "OTHERCIT":
            continue
        if not _plausible(num):
            n_implausible += 1
            continue
        ref = bib.number_to_ref.get(num)
        if ref is None:
            n_unresolvable += 1
        elif ref == own_refid:
            n_self += 1
        else:
            anchors.add(ref)
    return AnchorResolution(
        target_valid=target_valid,
        anchors=frozenset(anchors),
        n_unresolvable=n_unresolvable,
        n_implausible=n_implausible,
        n_self=n_self,
    )
