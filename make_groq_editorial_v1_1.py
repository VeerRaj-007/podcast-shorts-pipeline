from pathlib import Path

src = Path("groq_editorial_engine_v1.py")
out = Path("groq_editorial_engine_v1_1.py")

if not src.exists():
    raise FileNotFoundError(src)

text = src.read_text(encoding="utf-8")

# Version
if text.count('VERSION = "1.0"') != 1:
    raise RuntimeError("Expected exactly one VERSION = \"1.0\"")

text = text.replace(
    'VERSION = "1.0"',
    'VERSION = "1.1"',
    1,
)

# Replace rank_and_select with a stricter implementation.
start = text.index("def rank_and_select(")
end = text.index(
    "\n\n# ---------------------------------------------------------------------------\n# Main",
    start,
)

new_rank_and_select = '''def rank_and_select(
    candidates: Sequence[Candidate],
    requested_count: int,
    video_end: float,
) -> Tuple[List[Candidate], List[Candidate]]:
    """
    Build a strict publishable pool first, then rank it.

    A candidate may enter `selected` only if it passes ALL deterministic
    validation checks immediately before selection.

    Requested count NEVER changes the quality floor.
    """

    accepted: List[Candidate] = []

    for candidate in candidates:
        # Clear stale validation reasons from previous passes.
        candidate.rejection_reasons = []

        reasons = validate_candidate(
            candidate,
            video_end,
        )

        candidate.rejection_reasons = sorted(
            set(reasons)
        )

        if (
            not candidate.rejection_reasons
            and candidate.critic_status == "passed"
        ):
            accepted.append(candidate)

    accepted.sort(
        key=lambda c: (
            final_score(c),
            c.quality_score,
            c.payoff_score,
            c.boundary_confidence,
            c.confidence,
        ),
        reverse=True,
    )

    selected: List[Candidate] = []

    # Final diversity pass.
    for candidate in accepted:
        # Revalidate AGAIN immediately before serialization eligibility.
        candidate.rejection_reasons = sorted(
            set(
                validate_candidate(
                    candidate,
                    video_end,
                )
            )
        )

        if candidate.rejection_reasons:
            continue

        if candidate.critic_status != "passed":
            continue

        too_close = False

        for existing in selected:
            if interval_overlap(
                candidate,
                existing,
            ) >= 0.45:
                too_close = True
                break

            if story_similarity(
                candidate,
                existing,
            ) >= 0.68:
                too_close = True
                break

        if too_close:
            continue

        selected.append(candidate)

        if len(selected) >= min(
            requested_count,
            TOP_RETURN_LIMIT,
        ):
            break

    return selected, accepted
'''

text = text[:start] + new_rank_and_select + text[end:]

# Add a final output invariant before creating output["clips"].
anchor = '''    selected_ids = {c.candidate_id for c in selected}
'''

replacement = '''    # Final output invariant:
    # nothing may enter `clips` unless it is independently publishable.
    for candidate in selected:
        final_reasons = validate_candidate(
            candidate,
            args.video_end,
        )

        final_reasons.extend(
            candidate.rejection_reasons or []
        )

        final_reasons = sorted(
            set(final_reasons)
        )

        if (
            final_reasons
            or candidate.critic_status != "passed"
        ):
            raise EngineError(
                "Internal invariant violated: "
                f"selected candidate {candidate.candidate_id} "
                "failed final publication validation: "
                f"{final_reasons}"
            )

    selected_ids = {c.candidate_id for c in selected}
'''

if text.count(anchor) != 1:
    raise RuntimeError(
        "Could not find selected_ids anchor."
    )

text = text.replace(
    anchor,
    replacement,
    1,
)

# Add explicit quality status to output metadata.
old = '''        "abstained_from_fillers": len(selected) < args.count,
        "clips": [],
'''

new = '''        "abstained_from_fillers": len(selected) < args.count,
        "quality_status": (
            "fulfilled"
            if len(selected) >= args.count
            else "quality_abstention"
        ),
        "clips": [],
'''

if text.count(old) != 1:
    raise RuntimeError(
        "Could not find output metadata anchor."
    )

text = text.replace(
    old,
    new,
    1,
)

# Update CLI description if present.
text = text.replace(
    "Groq Editorial Engine v1",
    "Groq Editorial Engine v1.1",
)

out.write_text(
    text,
    encoding="utf-8",
)

compile(
    text,
    str(out),
    "exec",
)

print("=" * 72)
print("GROQ EDITORIAL ENGINE v1.1 CREATED")
print("=" * 72)
print()
print(f"Output: {out}")
print()
print("Changes:")
print("  - strict final candidate validation")
print("  - stale rejection reasons cleared before validation")
print("  - candidates revalidated before selection")
print("  - final publication invariant before serialization")
print("  - explicit quality_status")
print()
print("Syntax check: PASSED")