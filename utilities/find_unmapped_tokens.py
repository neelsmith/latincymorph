#!/usr/bin/env python3
"""Run a string of Latin text through the full latincymorph pipeline and
print *only* the tokens stage 3 couldn't map at all -- the genuine
failures (``MorphologicalFormResult.error`` set), not the companion types
(``AbbreviatedAdjective``, ``UngenderedNoun``, ``UngenderedPronoun``,
``UnclassifiedUninflected``) that stand in for a real
``tabulaedspy.MorphologicalForm`` when tabulaedspy's schema demands
something LatinCy didn't tag. Those companion-type tokens are
successes, just not native ones (see ``MorphologicalFormResult.native``);
this script is for finding the tokens that raised
``UnmappableTokenError`` -- a UD feature a *known* analytic type
specifically requires was missing, unrecognized, or rejected by
tabulaedspy's own validator (see
``latincymorph/tabulaedspy_bridge.py``'s module docstring for the
distinction, and ``utilities/analyze_text.py`` for a script that prints
every token's outcome, mapped or not).

Same three stages as ``analyze_text.py``:
Stage 1 (latincymorph.pipeline.analyze_text): load a LatinCy spaCy model
and parse the text into a Doc.
Stage 2 (latincymorph.extraction.extract_sentences): walk the Doc sentence
by sentence, pulling each analyzable token's lemma/UPOS/UD morphology.
Stage 3 (latincymorph.tabulaedspy_bridge.build_morphological_forms): map
each token's morphology onto tabulaedspy's scheme, per
quarto/reference/stringmappings.qmd.

Usage:
    python3 utilities/find_unmapped_tokens.py "Gallia est omnis divisa in partes tres."
    echo "Gallia est omnis divisa in partes tres." | python3 utilities/find_unmapped_tokens.py
    python3 utilities/find_unmapped_tokens.py --model la_core_web_sm "Arma virumque cano."

Requires latincymorph installed with its tabulaedspy extra
(pip install -e ".[tabulaedspy]") and a LatinCy model installed separately
-- see latincymorph/pipeline.py's module docstring for the install command,
and notes/pipeline-overview.md for why that install couldn't be verified
in the environment this pipeline was originally drafted in.
"""
from __future__ import annotations

import argparse
import sys

from latincymorph import DEFAULT_MODEL, analyze_text, build_morphological_forms, extract_sentences


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Latin text through latincymorph's nlp -> extraction -> "
            "tabulaedspy pipeline and print only the tokens stage 3 "
            "couldn't map at all."
        )
    )
    parser.add_argument(
        "text",
        nargs="?",
        help="Latin text to analyze. If omitted, reads from stdin.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"LatinCy model to load (default: {DEFAULT_MODEL})",
    )
    return parser.parse_args(argv)


def read_text(args: argparse.Namespace) -> str:
    if args.text:
        return args.text
    if sys.stdin.isatty():
        raise SystemExit(
            "No text given and no input piped in. Pass Latin text as an "
            "argument, or pipe it in via stdin."
        )
    text = sys.stdin.read().strip()
    if not text:
        raise SystemExit("No text given and stdin was empty.")
    return text


def format_feats(feats: dict) -> str:
    return "|".join(f"{key}={value}" for key, value in feats.items()) if feats else "-"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    text = read_text(args)

    print(f"Loading LatinCy model {args.model!r}...", file=sys.stderr)
    try:
        doc = analyze_text(text, model_name=args.model)
    except OSError as exc:
        print(
            f"Could not load LatinCy model {args.model!r}: {exc}\n\n"
            "Install it first, e.g.:\n"
            '  pip install "la-core-web-lg @ https://huggingface.co/latincy/'
            'la_core_web_lg/resolve/main/la_core_web_lg-3.9.6-py3-none-any.whl"\n'
            "(check https://huggingface.co/latincy/la_core_web_lg/tree/main for the "
            "current wheel filename if that version is out of date)\n\n"
            "See latincymorph/pipeline.py and notes/pipeline-overview.md for details.",
            file=sys.stderr,
        )
        return 1

    sentences = extract_sentences(doc)
    total_tokens = sum(len(sentence) for sentence in sentences)

    unmapped_count = 0
    for sentence in sentences:
        for result in build_morphological_forms(sentence):
            if result.ok:
                continue
            unmapped_count += 1
            token = result.token
            print(
                f"[sentence {token.sent_index}, token {token.token_index}] "
                f"{token.text!r} lemma={token.lemma!r} UPOS={token.upos} "
                f"feats={format_feats(token.feats)} -- {result.error.reason}"
            )

    print(f"\n{unmapped_count} unmapped token(s) out of {total_tokens} total.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
