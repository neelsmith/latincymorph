"""Stage 3 of the latincymorph pipeline: map the UD morphological features
LatinCy/spaCy attaches to a token onto tabulaedspy's ``MorphologicalForm``
schema (``morphology_scheme.md`` in the tabulaedspy repository), and
instantiate it.

This is the only latincymorph module that imports tabulaedspy -- and
therefore, transitively, ``dspy`` and ``arsgrammatica`` (see tabulaedspy's
own ``pyproject.toml``). ``extraction.py`` (stage 2) has no tabulaedspy
dependency at all, so a caller who only wants raw spaCy morphology never
needs this module, or its heavier dependency chain, installed.

**Mapping fidelity, read before trusting output at scale.** tabulaedspy's
own analysis pipeline (``tabulaedspy.pipeline.analyze_sentence`` /
``analyze_document``) is a *language model*, asked to disambiguate a
token's morphology using its syntactic context -- e.g. a neuter noun that
is morphologically nominative/accusative/vocative gets resolved to
exactly one case by its syntactic role in an ``arsgrammatica`` analysis.
LatinCy's statistical tagger/morphologizer is also context-sensitive (it
consumes surrounding tokens as features), but it commits to *UD* tags, not
tabulaedspy's scheme -- so this module is a second, independent,
deterministic translation from one already-committed tagset to another.
It does not re-implement, and cannot reproduce, tabulaedspy's own
disambiguation step. See ``notes/spacy-to-tabulaedspy-mapping.md`` for the
full mapping tables, their sources, and known gaps (e.g. locative case,
which tabulaedspy's schema has no slot for at all) -- several of these
mappings are documented best-effort judgment calls that still need
checking against the real model's actual output (see that note for why
that verification is still outstanding).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from pydantic import ValidationError
from tabulaedspy import MorphologicalForm

from .extraction import TokenMorphology

# ---------------------------------------------------------------------------
# UD feature-value -> tabulaedspy property-value lookup tables.
#
# Keys are exactly the strings LatinCy's morphologizer writes into
# ``Token.morph`` (per the la_core_web_lg model card); values are exactly
# the strings tabulaedspy.models declares valid (morphology_scheme.md,
# "Valid values for morphological properties").
# ---------------------------------------------------------------------------

_CASE = {
    "Nom": "nominative",
    "Gen": "genitive",
    "Dat": "dative",
    "Acc": "accusative",
    "Abl": "ablative",
    "Voc": "vocative",
    # "Loc" (locative) is deliberately absent: tabulaedspy's Case enum has
    # no locative value. A token tagged Case=Loc fails to map -- see
    # notes/spacy-to-tabulaedspy-mapping.md, "Known gaps".
}

_GENDER = {"Masc": "masculine", "Fem": "feminine", "Neut": "neuter"}
_NUMBER = {"Sing": "singular", "Plur": "plural"}
_DEGREE = {"Pos": "positive", "Cmp": "comparative", "Sup": "superlative"}
_MOOD = {"Ind": "indicative", "Sub": "subjunctive", "Imp": "imperative"}
_VOICE = {"Act": "active", "Pass": "passive"}
_PERSON = {"1": "first", "2": "second", "3": "third"}

# Latin's six-way tense system, in the harmonized UD treebanks LatinCy
# trains on (Perseus, PROIEL, ITTB, LLCT, UDante, CIRCSE, LASLA), is spread
# across *two* UD features -- Tense and Aspect -- following the PROIEL-style
# convention shared with Ancient Greek (see
# https://universaldependencies.org/la/ and .../grc/). A finite or
# non-finite form's tabulaedspy `tense` is the *pair* (Tense, Aspect), not
# Tense alone. See notes/spacy-to-tabulaedspy-mapping.md for the worked
# derivation of this table and why it still needs confirming against real
# model output.
_TENSE_ASPECT: Dict[Tuple[str, Optional[str]], str] = {
    ("Pres", "Imp"): "present",
    ("Pres", None): "present",
    ("Past", "Imp"): "imperfect",
    ("Past", "Perf"): "perfect",
    ("Pqp", "Perf"): "pluperfect",
    ("Pqp", None): "pluperfect",
    ("Fut", "Prosp"): "future",
    ("Fut", None): "future",
    ("Fut", "Perf"): "future_perfect",
}

# UPOS -> tabulaedspy `uninflected_type`, for the analytic types that never
# change form (morphology_scheme.md's "uninflected" table).
_UNINFLECTED_TYPE_BY_UPOS = {
    "ADP": "preposition",
    "CCONJ": "conjunction",
    "SCONJ": "conjunction",
    "PART": "particle",
    "INTJ": "interjection",
    "NUM": "number",
    # SYM/X have no principled home in the 7-value uninflected_type enum;
    # "foreign" is the closest available bucket, not a documented
    # equivalence -- a judgment call, see notes.
    "SYM": "foreign",
    "X": "foreign",
}


class UnmappableTokenError(ValueError):
    """A token's spaCy/UD morphology could not be translated into a valid
    tabulaedspy ``MorphologicalForm``: either a required UD feature was
    missing or had no tabulaedspy equivalent, or the resulting property
    set failed tabulaedspy's own validator (e.g. a supine whose case isn't
    accusative or ablative).

    Carries the offending :class:`~latincymorph.extraction.TokenMorphology`
    and a human-readable ``reason`` so a caller doing corpus-level
    extraction can log and skip a bad token rather than crash the whole
    run -- see :func:`build_morphological_forms`.
    """

    def __init__(self, token: TokenMorphology, reason: str):
        super().__init__(f"{token.text!r} ({token.upos}, {token.feats}): {reason}")
        self.token = token
        self.reason = reason


def _lookup(table: Dict[str, str], key: Optional[str], token: TokenMorphology, feature: str) -> str:
    if key is None:
        raise UnmappableTokenError(token, f"missing required UD feature {feature!r}")
    try:
        return table[key]
    except KeyError:
        raise UnmappableTokenError(
            token, f"unrecognized value {feature}={key!r} (no tabulaedspy equivalent)"
        ) from None


def _tense(token: TokenMorphology) -> str:
    tense_key = token.feats.get("Tense")
    aspect_key = token.feats.get("Aspect")
    if tense_key is None:
        raise UnmappableTokenError(token, "missing required UD feature 'Tense'")
    try:
        return _TENSE_ASPECT[(tense_key, aspect_key)]
    except KeyError:
        raise UnmappableTokenError(
            token,
            f"unrecognized Tense/Aspect combination Tense={tense_key!r}, Aspect={aspect_key!r}",
        ) from None


def _degree_or_default(token: TokenMorphology, *, default: str) -> str:
    raw = token.feats.get("Degree")
    if raw is None:
        return default
    return _lookup(_DEGREE, raw, token, "Degree")


def _nominal_properties(token: TokenMorphology) -> dict:
    return {
        "gender": _lookup(_GENDER, token.feats.get("Gender"), token, "Gender"),
        "case": _lookup(_CASE, token.feats.get("Case"), token, "Case"),
        "number": _lookup(_NUMBER, token.feats.get("Number"), token, "Number"),
    }


def _build_kwargs(token: TokenMorphology) -> dict:
    upos = token.upos
    verb_form = token.feats.get("VerbForm")

    if upos in ("NOUN", "PROPN"):
        return {"analytic_type": "noun", **_nominal_properties(token)}

    if upos in ("PRON", "DET"):
        # DET has no dedicated analytic_type in tabulaedspy's 11-type
        # scheme. Latin UD DET tokens (demonstratives, possessives) are
        # "pronominal adjectives" in the traditional Latin-grammar sense
        # tabulaedspy's scheme follows, so they're folded into "pronoun"
        # here -- a judgment call, see notes/spacy-to-tabulaedspy-mapping.md.
        return {"analytic_type": "pronoun", **_nominal_properties(token)}

    if upos == "ADJ":
        props = _nominal_properties(token)
        props["degree"] = _degree_or_default(token, default="positive")
        return {"analytic_type": "adjective", **props}

    if upos == "ADV":
        # morphology_scheme.md: an adverb derived from an adjective is
        # analytic_type "adverb" with a `degree`; one that isn't is
        # "uninflected" with uninflected_type "adverb". LatinCy's morph
        # string has no direct "derived from an adjective" flag, so this
        # uses Degree's *presence* as a proxy -- see notes for the
        # reasoning and its failure modes (e.g. an adverb whose tagger
        # output omits Degree even though it is gradable).
        raw_degree = token.feats.get("Degree")
        if raw_degree is not None:
            return {"analytic_type": "adverb", "degree": _lookup(_DEGREE, raw_degree, token, "Degree")}
        return {"analytic_type": "uninflected", "uninflected_type": "adverb"}

    if upos in ("VERB", "AUX"):
        if verb_form == "Fin":
            return {
                "analytic_type": "finite verb",
                "tense": _tense(token),
                "mood": _lookup(_MOOD, token.feats.get("Mood"), token, "Mood"),
                "voice": _lookup(_VOICE, token.feats.get("Voice"), token, "Voice"),
                "person": _lookup(_PERSON, token.feats.get("Person"), token, "Person"),
                "number": _lookup(_NUMBER, token.feats.get("Number"), token, "Number"),
            }
        if verb_form == "Inf":
            return {
                "analytic_type": "infinitive",
                "tense": _tense(token),
                "voice": _lookup(_VOICE, token.feats.get("Voice"), token, "Voice"),
            }
        if verb_form == "Part":
            return {
                "analytic_type": "participle",
                "tense": _tense(token),
                "voice": _lookup(_VOICE, token.feats.get("Voice"), token, "Voice"),
                **_nominal_properties(token),
            }
        if verb_form == "Ger":
            return {"analytic_type": "gerund", "case": _lookup(_CASE, token.feats.get("Case"), token, "Case")}
        if verb_form == "Gdv":
            return {"analytic_type": "gerundive", **_nominal_properties(token)}
        if verb_form == "Sup":
            return {"analytic_type": "supine", "case": _lookup(_CASE, token.feats.get("Case"), token, "Case")}
        raise UnmappableTokenError(
            token, f"VERB/AUX token with missing or unrecognized VerbForm={verb_form!r}"
        )

    uninflected_type = _UNINFLECTED_TYPE_BY_UPOS.get(upos)
    if uninflected_type is not None:
        return {"analytic_type": "uninflected", "uninflected_type": uninflected_type}

    raise UnmappableTokenError(token, f"no analytic_type mapping defined for UPOS {upos!r}")


def build_morphological_form(token: TokenMorphology) -> MorphologicalForm:
    """Map one :class:`~latincymorph.extraction.TokenMorphology` onto a
    tabulaedspy ``MorphologicalForm`` and instantiate it.

    Raises :class:`UnmappableTokenError` if a required UD feature is
    missing or has no tabulaedspy equivalent, and wraps tabulaedspy's own
    ``pydantic.ValidationError`` in the same exception type if the mapped
    property set still fails its validator (e.g. an unexpected supine
    case) -- so callers only ever need to catch one exception type.
    """
    kwargs = _build_kwargs(token)
    try:
        return MorphologicalForm(**kwargs)
    except ValidationError as exc:
        raise UnmappableTokenError(
            token, f"tabulaedspy rejected mapped properties {kwargs}: {exc}"
        ) from exc


@dataclass(frozen=True)
class MorphologicalFormResult:
    """One token's mapping outcome: exactly one of ``form`` (success) or
    ``error`` (failure) is set. Returned by :func:`build_morphological_forms`
    so a caller can separate successes from failures without wrapping every
    token in its own try/except.
    """

    token: TokenMorphology
    form: Optional[MorphologicalForm] = None
    error: Optional[UnmappableTokenError] = None

    @property
    def ok(self) -> bool:
        return self.error is None


def build_morphological_forms(tokens: Iterable[TokenMorphology]) -> List[MorphologicalFormResult]:
    """Map an iterable of :class:`TokenMorphology` (e.g. one sentence from
    :func:`latincymorph.extraction.extract_sentences`) to
    :class:`MorphologicalFormResult` records, one per token, in order.

    Never raises: a mapping failure for one token is captured in that
    token's ``.error`` rather than aborting the whole batch, since a
    corpus-level run should be able to report "312 of 320 tokens mapped
    cleanly" rather than dying on the first exception.
    """
    results: List[MorphologicalFormResult] = []
    for token in tokens:
        try:
            results.append(MorphologicalFormResult(token=token, form=build_morphological_form(token)))
        except UnmappableTokenError as exc:
            results.append(MorphologicalFormResult(token=token, error=exc))
    return results
