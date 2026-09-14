#!/usr/bin/env python3
"""Run a string of Latin text through the full latincymorph pipeline and
print what came out at each stage.

Stage 1 (latincymorph.pipeline.analyze_text): load a LatinCy spaCy model
and parse the text into a Doc.
Stage 2 (latincymorph.extraction.extract_sentences): walk the Doc sentence
by sentence, pulling each analyzable token's lemma/UPOS/UD morphology.
Stage 3 (latincymorph.tabulaedspy_bridge.build_morphological_forms): map
each token's morphology onto tabulaedspy's MorphologicalForm schema.

Usage:
    python3 utilities/analyze_text.py "Gallia est omnis divisa in partes tres."
    echo "Gallia est omnis divisa in partes tres." | python3 utilities/analyze_text.py
    python3 utilities/analyze_text.py --model la_core_web_sm "Arma virumque cano."

Requires latincymorph installed with its tabulaedspy extra
(pip install -e ".[tabulaedspy]") and a LatinCy model installed separately
-- see latincymorph/pipeline.py's module docstring for the install command,
and notes/pipeline-overview.md for why that install couldn't be verified
in the environment this pipeline was originally drafted in.
"""
from __future__ import annotations

import argparse
import sys

from latincymorph import (
    DEFAULT_MODEL,
    MorphologicalFormResult,
    analyze_text,
    build_morphological_forms,
    extract_sentences,
)

# Every optional MorphologicalForm property, in the order rendering.py's
# own label_morphology() uses them upstream in tabulaedspy -- kept here
# only for a stable, readable column order, not imported from tabulaedspy
# itself (this script only needs the property *names*, which are part of
# the pydantic model's own field names, not tabulaedspy's internals).
_FORM_FIELDS = (
    "gender",
    "case",
    "number",
    "degree",
    "tense",
    "mood",
    "voice",
    "person",
    "uninflected_type",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Latin text through latincymorph's nlp -> extraction -> "
            "tabulaedspy pipeline and print the result."
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


def format_form(result: MorphologicalFormResult) -> str:
    if not result.ok:
        return f"UNMAPPED ({result.error.reason})"
    form = result.form
    parts = [f"analytic_type={form.analytic_type}"]
    parts.extend(
        f"{field}={value}"
        for field in _FORM_FIELDS
        if (value := getattr(form, field)) is not None
    )
    return " ".join(parts)


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

    print(f"\nInput: {text}")
    print(f"{total_tokens} analyzable token(s) across {len(sentences)} sentence(s)\n")

    mapped_count = 0
    for sent_index, sentence in enumerate(sentences):
        print(f"--- Sentence {sent_index} ---")
        results = build_morphological_forms(sentence)

        header = f"{'text':<15} {'lemma':<15} {'UPOS':<6} {'spaCy morphology':<40} tabulaedspy MorphologicalForm"
        print(header)
        print("-" * len(header))
        for result in results:
            token = result.token
            mapped_count += result.ok
            print(
                f"{token.text:<15} {token.lemma:<15} {token.upos:<6} "
                f"{format_feats(token.feats):<40} {format_form(result)}"
            )
        print()

    print(f"Summary: {mapped_count}/{total_tokens} token(s) mapped to a tabulaedspy MorphologicalForm.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
