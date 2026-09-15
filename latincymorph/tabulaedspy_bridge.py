"""Stage 3 of the latincymorph pipeline: map the UD morphological features
LatinCy/spaCy attaches to a token onto tabulaedspy's ``MorphologicalForm``
schema, and instantiate it.

**Authoritative source.** The mapping tables below implement
``quarto/reference/stringmappings.qmd`` (Neel Smith's own reference,
maintained by hand at the repository root -- not this module's docstring
or ``notes/``) as closely as the tabulaedspy schema allows. Where this
module diverges from that document -- because the document doesn't cover a
case that real tokens hit, or because tabulaedspy's schema can't literally
represent what the document asks for -- the divergence is called out
explicitly in a comment at the point it happens, and summarized in
``notes/spacy-to-tabulaedspy-mapping.md``. Treat that document, not this
one, as the place to correct the mapping's linguistic judgment calls.

This is the only latincymorph module that imports tabulaedspy -- and
therefore, transitively, ``dspy`` and ``arsgrammatica`` (see tabulaedspy's
own ``pyproject.toml``). ``extraction.py`` (stage 2) has no tabulaedspy
dependency at all, so a caller who only wants raw spaCy morphology never
needs this module, or its heavier dependency chain, installed.

**Two of tabulaedspy's required properties are not things LatinCy tags at
all**, per stringmappings.qmd's own note under "Degree": LatinCy does not
appear to record degree of comparison for adjectives or adverbs, but
tabulaedspy's schema requires `degree` for both its `adjective` and
`adverb` analytic types. This module resolves the two cases differently,
because they're not actually symmetric -- an adjective still has three
other well-attested properties (gender/case/number) worth keeping, while
a bare adverb has nothing else to keep:

- **Adjectives** missing `degree` become :class:`AbbreviatedAdjective`, a
  local companion type (gender/case/number, no degree slot at all) rather
  than a real ``tabulaedspy.MorphologicalForm`` -- so real data isn't
  thrown away, and no degree value is invented. If LatinCy ever does
  supply `Degree` on an adjective token, this module still produces a
  genuine ``MorphologicalForm(analytic_type="adjective", ...)`` in that
  case, so this isn't a permanent downgrade, just a fallback for what
  LatinCy actually gives us today.
- **Adverbs** are always mapped to ``analytic_type="uninflected"``,
  ``uninflected_type="adverb"`` -- tabulaedspy's own scheme's fallback
  category for adverbs that don't carry gradation -- regardless of
  whatever ``token.morph`` happens to contain. There is nothing else an
  adverb analysis could keep if `degree` isn't available, so unlike
  adjectives there's no local companion type for adverbs; they simply
  never attempt the `adverb` analytic type at all.

**The `uninflected` catch-all.** stringmappings.qmd's own uninflected_type
table only covers `CCONJ`, `ADP`, `ADV`, `NUM`, `INTJ`, `PART`, and `X`
(the last one, per the document's own caveat, is a probable mixed bag of
real foreign words and tagger errors). Any analyzable token whose UPOS
doesn't match one of the specific rules below -- including that
uninflected_type table -- becomes :class:`UnclassifiedUninflected`, a
second local companion type that keeps the token's raw UPOS and UD
features for a human (or a future, more complete mapping rule) to look
at, rather than either raising an error or guessing a specific
tabulaedspy `uninflected_type` value the document doesn't support.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Optional, Union

from pydantic import BaseModel, ValidationError
from tabulaedspy import MorphologicalForm

from .extraction import TokenMorphology

# ---------------------------------------------------------------------------
# UD feature-value -> tabulaedspy property-value lookup tables.
# Source: quarto/reference/stringmappings.qmd, "Values for morphological
# properties".
# ---------------------------------------------------------------------------

_CASE = {
    "Nom": "nominative",
    "Gen": "genitive",
    "Dat": "dative",
    "Acc": "accusative",
    "Abl": "ablative",
    "Voc": "vocative",
    # "Loc" (locative) is not in stringmappings.qmd's Case table, and
    # tabulaedspy's own Case enum has no locative value either -- a token
    # tagged Case=Loc fails to map. See notes/spacy-to-tabulaedspy-mapping.md,
    # "Known gaps".
}
_GENDER = {"Masc": "masculine", "Fem": "feminine", "Neut": "neuter"}
_NUMBER = {"Sing": "singular", "Plur": "plural"}
_DEGREE = {"Pos": "positive", "Cmp": "comparative", "Sup": "superlative"}
_MOOD = {"Ind": "indicative", "Sub": "subjunctive", "Imp": "imperative"}
_VOICE = {"Act": "active", "Pass": "passive"}
_PERSON = {"1": "first", "2": "second", "3": "third"}

# stringmappings.qmd, "Uninflected type" -- keyed by UPOS, used only when a
# token has already been routed to the `uninflected` catch-all below (see
# _build_uninflected_or_unknown). ADV is included for documentation
# completeness even though a real ADV token never reaches this table in
# practice: it's intercepted earlier and always becomes uninflected/adverb
# directly (see the module docstring).
_UNINFLECTED_TYPE_BY_UPOS = {
    "CCONJ": "conjunction",
    "ADP": "preposition",
    "ADV": "adverb",
    "NUM": "number",
    "INTJ": "interjection",
    "PART": "particle",
    "X": "foreign",
}

# stringmappings.qmd, "Analytic type": which VerbForm value routes to which
# tabulaedspy analytic type, checked against token.morph regardless of the
# token's UPOS (a Latin copula tagged AUX with VerbForm=Fin is still a
# finite verb) -- matching the reference document's own phrasing, which
# keys these rules on morph content, not part of speech.
_ANALYTIC_TYPE_BY_VERB_FORM = {
    "Fin": "finite verb",
    "Inf": "infinitive",
    "Part": "participle",
    "Sup": "supine",
    "Ger": "gerund",
    "Gdv": "gerundive",
}

GenderValue = Literal["masculine", "feminine", "neuter"]
CaseValue = Literal["nominative", "genitive", "dative", "accusative", "ablative", "vocative"]
NumberValue = Literal["singular", "plural"]


class AbbreviatedAdjective(BaseModel):
    """An adjective's gender/case/number, without a `degree` -- for an ADJ
    token whose morphology has none, which per
    ``quarto/reference/stringmappings.qmd`` ("Degree") is the LatinCy norm
    rather than the exception. tabulaedspy's own `MorphologicalForm`
    requires `degree` for `analytic_type="adjective"`, so a token like this
    cannot become a real one; this type exists so its other three, usually
    reliable properties aren't discarded and no degree value is invented.
    See :func:`build_morphological_form`, which still returns a genuine
    ``MorphologicalForm`` for the rarer case where `Degree` *is* present.
    """

    analytic_type: Literal["adjective"] = "adjective"
    gender: GenderValue
    case: CaseValue
    number: NumberValue


class UnclassifiedUninflected(BaseModel):
    """A token that isn't one of the analytic types this module can
    confidently identify (noun/pronoun/adjective/a verb-family type), and
    whose UPOS also isn't one of the seven values
    ``quarto/reference/stringmappings.qmd``'s "Uninflected type" table
    covers. Carries the token's own UPOS and raw UD features rather than
    guessing a specific tabulaedspy `uninflected_type` the reference
    document doesn't support, or dropping the token with an error --
    tabulaedspy's `uninflected` is meant as the catch-all analytic type for
    words that don't inflect at all, and a token this module can't place
    more precisely is exactly the case that catch-all exists for.
    """

    analytic_type: Literal["uninflected"] = "uninflected"
    uninflected_type: Literal["unknown"] = "unknown"
    upos: str
    feats: Dict[str, str]


#: Everything stage 3 can produce for one token that isn't an outright
#: mapping failure: either a genuine tabulaedspy object, or one of the two
#: local companion types above for what tabulaedspy's schema can't
#: represent from what LatinCy actually tags.
StageThreeResult = Union[MorphologicalForm, AbbreviatedAdjective, UnclassifiedUninflected]


class UnmappableTokenError(ValueError):
    """A token's spaCy/UD morphology could not be translated at all: a UD
    feature *required by the analytic type this module already committed
    to* (e.g. `Case` for a noun, `Mood` for a finite verb) was missing or
    had a value with no tabulaedspy equivalent, or the resulting property
    set still failed tabulaedspy's own validator (e.g. a supine whose case
    isn't accusative or ablative).

    This is deliberately narrower than "we don't know what analytic type
    this token is" -- that case returns :class:`UnclassifiedUninflected`
    instead of raising (see the module docstring). This error means the
    opposite: the analytic type is known, but a property it specifically
    requires is missing or invalid.

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
    """Tense for **finite verbs and infinitives** -- stringmappings.qmd,
    "Tense". Note this is *not* a simple (Tense, Aspect) lookup table:
    "present" is triggered by ``Tense=Pres`` alone (Aspect isn't checked),
    and "pluperfect" by ``Tense=Pqp`` alone, while the other four values
    each need a specific Tense *and* Aspect pair -- because a finite verb
    genuinely has two parallel forms under e.g. ``Tense=Past`` (imperfect
    "amabam" vs. perfect "amavi") that only `Aspect` distinguishes.
    Implemented as the document's own conditions, checked in the
    document's own order, rather than a dict keyed by the pair -- a literal
    reading of the reference document, not a guess at the underlying UD
    convention.

    **Not used for participles** -- see :func:`_participle_tense`, which
    determines a participle's tense from `Aspect` alone and never
    consults `Tense` at all (the reverse of this function).
    """
    tense = token.feats.get("Tense")
    aspect = token.feats.get("Aspect")

    if tense == "Pres":
        return "present"
    if tense == "Past" and aspect == "Imp":
        return "imperfect"
    if tense == "Fut" and aspect == "Imp":
        return "future"
    if tense == "Past" and aspect == "Perf":
        return "perfect"
    if tense == "Pqp":
        return "pluperfect"
    if tense == "Fut" and aspect == "Perf":
        return "future_perfect"

    raise UnmappableTokenError(
        token, f"unrecognized Tense/Aspect combination Tense={tense!r}, Aspect={aspect!r}"
    )


# Tense for a **participle**, keyed by `Aspect` alone -- corrected per
# stringmappings.qmd's own "Tense" table, which now spells out that
# LatinCy tags a participle's tense through `Aspect`, not `Tense` at all:
# the present participle is `Aspect=Imp` (imperfective aspect), the
# future/prospective participle is `Aspect=Prosp`, and the perfect passive
# participle is `Aspect=Perf`. This is the opposite of finite verbs, where
# `Tense` is primary and `Aspect` only disambiguates -- see _tense()'s own
# docstring. An earlier version of this function keyed on `Tense` instead
# (before Neel had confirmed how LatinCy actually tags participles); that
# was backwards, not just incomplete -- `Tense` is not to be consulted for
# a participle at all, only `Aspect`. See
# notes/spacy-to-tabulaedspy-mapping.md.
_PARTICIPLE_TENSE_BY_ASPECT = {
    "Imp": "present",
    "Prosp": "future",
    "Perf": "perfect",
}


def _participle_tense(token: TokenMorphology) -> str:
    """Tense for a participle -- see :data:`_PARTICIPLE_TENSE_BY_ASPECT`.
    Deliberately distinct from :func:`_tense`: a participle's tense is
    determined by `Aspect` alone, never by `Tense` (backwards from finite
    verbs/infinitives, where `Tense` is primary). A token with
    `VerbForm=Part` is a participle regardless of whatever else its
    morphology string contains, so this function only ever fails for an
    `Aspect` value Latin participles don't actually have (missing, or
    something other than Imp/Prosp/Perf) -- never because `Tense` is
    absent or has some value that would mean something different for a
    finite verb.
    """
    aspect = token.feats.get("Aspect")
    try:
        return _PARTICIPLE_TENSE_BY_ASPECT[aspect]
    except KeyError:
        raise UnmappableTokenError(
            token,
            f"unrecognized participle Aspect={aspect!r} (expected one of "
            f"{sorted(_PARTICIPLE_TENSE_BY_ASPECT)})",
        ) from None


# Tense for an **infinitive**, keyed by `Aspect` alone -- same pattern as
# a participle's tense (see _participle_tense() above): stringmappings.qmd's
# "Tense" table now spells out `Aspect=Imp` -> `present` and `Aspect=Perf`
# -> `perfect` for infinitives too, and `Tense` is never consulted. Unlike
# participles there is no third, `Aspect=Prosp` -> `future` entry here --
# Latin has no synthetic future infinitive (the "future infinitive" is a
# periphrasis, e.g. `amaturus esse`, built from a separate participle
# token plus "esse"), so only these two values are recognized.
_INFINITIVE_TENSE_BY_ASPECT = {
    "Imp": "present",
    "Perf": "perfect",
}


def _infinitive_tense(token: TokenMorphology) -> str:
    """Tense for an infinitive -- see :data:`_INFINITIVE_TENSE_BY_ASPECT`.
    Deliberately distinct from :func:`_tense`, and like
    :func:`_participle_tense`: an infinitive's tense is determined by
    `Aspect` alone, never `Tense`.
    """
    aspect = token.feats.get("Aspect")
    try:
        return _INFINITIVE_TENSE_BY_ASPECT[aspect]
    except KeyError:
        raise UnmappableTokenError(
            token,
            f"unrecognized infinitive Aspect={aspect!r} (expected one of "
            f"{sorted(_INFINITIVE_TENSE_BY_ASPECT)})",
        ) from None


def _nominal_properties(token: TokenMorphology) -> dict:
    return {
        "gender": _lookup(_GENDER, token.feats.get("Gender"), token, "Gender"),
        "case": _lookup(_CASE, token.feats.get("Case"), token, "Case"),
        "number": _lookup(_NUMBER, token.feats.get("Number"), token, "Number"),
    }


def _build_verb_family(token: TokenMorphology, verb_form: str) -> MorphologicalForm:
    analytic_type = _ANALYTIC_TYPE_BY_VERB_FORM[verb_form]

    if analytic_type == "finite verb":
        kwargs = {
            "tense": _tense(token),
            "mood": _lookup(_MOOD, token.feats.get("Mood"), token, "Mood"),
            "voice": _lookup(_VOICE, token.feats.get("Voice"), token, "Voice"),
            "person": _lookup(_PERSON, token.feats.get("Person"), token, "Person"),
            "number": _lookup(_NUMBER, token.feats.get("Number"), token, "Number"),
        }
    elif analytic_type == "infinitive":
        kwargs = {"tense": _infinitive_tense(token), "voice": _lookup(_VOICE, token.feats.get("Voice"), token, "Voice")}
    elif analytic_type == "participle":
        kwargs = {
            "tense": _participle_tense(token),
            "voice": _lookup(_VOICE, token.feats.get("Voice"), token, "Voice"),
            **_nominal_properties(token),
        }
    elif analytic_type == "supine":
        kwargs = {"case": _lookup(_CASE, token.feats.get("Case"), token, "Case")}
    elif analytic_type == "gerund":
        kwargs = {"case": _lookup(_CASE, token.feats.get("Case"), token, "Case")}
    elif analytic_type == "gerundive":
        kwargs = _nominal_properties(token)
    else:  # pragma: no cover -- exhaustive over _ANALYTIC_TYPE_BY_VERB_FORM's values
        raise AssertionError(f"unreachable analytic_type {analytic_type!r}")

    try:
        return MorphologicalForm(analytic_type=analytic_type, **kwargs)
    except ValidationError as exc:
        raise UnmappableTokenError(
            token, f"tabulaedspy rejected mapped properties for {analytic_type!r}: {exc}"
        ) from exc


def _build_adjective(token: TokenMorphology) -> Union[MorphologicalForm, AbbreviatedAdjective]:
    props = _nominal_properties(token)
    raw_degree = token.feats.get("Degree")
    if raw_degree is None:
        # The LatinCy norm per stringmappings.qmd's "Degree" note -- see
        # AbbreviatedAdjective's own docstring.
        return AbbreviatedAdjective(**props)
    degree = _lookup(_DEGREE, raw_degree, token, "Degree")
    try:
        return MorphologicalForm(analytic_type="adjective", degree=degree, **props)
    except ValidationError as exc:
        raise UnmappableTokenError(
            token, f"tabulaedspy rejected mapped properties for 'adjective': {exc}"
        ) from exc


def _build_uninflected_or_unknown(token: TokenMorphology) -> Union[MorphologicalForm, UnclassifiedUninflected]:
    uninflected_type = _UNINFLECTED_TYPE_BY_UPOS.get(token.upos)
    if uninflected_type is not None:
        return MorphologicalForm(analytic_type="uninflected", uninflected_type=uninflected_type)
    return UnclassifiedUninflected(upos=token.upos, feats=dict(token.feats))


def build_morphological_form(token: TokenMorphology) -> StageThreeResult:
    """Map one :class:`~latincymorph.extraction.TokenMorphology` to
    whatever stage 3 can produce for it: a genuine
    ``tabulaedspy.MorphologicalForm`` in the ordinary case, or one of
    :class:`AbbreviatedAdjective` / :class:`UnclassifiedUninflected` when
    tabulaedspy's schema demands something LatinCy's tagger didn't supply
    (see the module docstring for both).

    Dispatch order, mirroring ``quarto/reference/stringmappings.qmd``'s own
    "Analytic type" table:

    1. ``token.morph`` includes a recognized ``VerbForm`` value (`Fin`,
       `Inf`, `Part`, `Sup`, `Ger`, `Gdv`) -> the matching verb-family
       analytic type, checked regardless of ``token.pos_`` (so an AUX
       copula with ``VerbForm=Fin`` is still a finite verb).
    2. ``token.pos_ == "AUX"`` (and step 1 didn't already match) ->
       ``finite verb``, built the same way as a `VerbForm=Fin` token.
       **Note:** stringmappings.qmd's "Analytic type" table now lists
       `AUX -> finite verb` directly, alongside its `VerbForm=Fin` row.
       Because the copula ("sum") also has infinitive and participle
       forms (`VerbForm=Inf`/`Part` -- "esse", "futurus") that are also
       tagged `AUX`, this module reads the new row as a *fallback* for an
       `AUX` token whose morphology doesn't carry a recognized `VerbForm`
       at all, rather than as overriding step 1 -- so an infinitive or
       participle copula token still gets its own analytic type, not
       `finite verb`. Flag to Neel if `AUX` was meant to take priority
       over `VerbForm` instead.
    3. ``token.pos_ == "ADV"`` -> always ``uninflected``/``adverb`` (see
       the module docstring; this preempts stringmappings.qmd's own
       ``ADV -> adverb`` row, which would require a `degree` LatinCy
       doesn't supply).
    4. ``token.pos_`` is ``NOUN`` or ``PROPN`` -> ``noun`` --
       stringmappings.qmd's own "Analytic type" table lists both UPOS
       values directly (`NOUN` or `PROPN` -> `noun`). This was originally
       this module's own inference; Neel has since confirmed it by adding
       `PROPN` to the reference document itself, so it's no longer a
       judgment call this module is making on its own.
    5. ``token.pos_`` is ``PRON`` or ``DET`` -> ``pronoun``. **Note:**
       stringmappings.qmd's table still names only `PRON`; `DET` remains
       this module's own inference (Latin UD `DET` tokens --
       demonstratives, possessives -- are pronominal adjectives in the
       traditional grammar tabulaedspy's scheme follows, and have the
       same gender/case/number property set as `PRON` either way) -- flag
       this to Neel the same way `PROPN` was, if that's not the intended
       reading.
    6. ``token.pos_ == "ADJ"`` -> :func:`_build_adjective` (a genuine
       `adjective` `MorphologicalForm` if `Degree` is present, otherwise
       :class:`AbbreviatedAdjective`).
    7. Anything else -> :func:`_build_uninflected_or_unknown`: a genuine
       `uninflected` `MorphologicalForm` if ``token.pos_`` is one of the
       seven UPOS values stringmappings.qmd's "Uninflected type" table
       covers, otherwise :class:`UnclassifiedUninflected`. **Note:** this
       is where `SCONJ` and `SYM` currently land, since neither appears
       in that table (only `CCONJ` is listed for conjunctions) -- `SCONJ`
       in particular is common in Latin (*cum*, *ut*, *si*, *quod*...) and
       is likely an oversight worth adding to stringmappings.qmd rather
       than a deliberate exclusion.

    Raises :class:`UnmappableTokenError` only once an analytic type *is*
    already determined but a property it specifically requires is missing,
    unrecognized, or fails tabulaedspy's own validator.
    """
    verb_form = token.feats.get("VerbForm")
    if verb_form in _ANALYTIC_TYPE_BY_VERB_FORM:
        return _build_verb_family(token, verb_form)

    if token.upos == "AUX":
        return _build_verb_family(token, "Fin")

    if token.upos == "ADV":
        return MorphologicalForm(analytic_type="uninflected", uninflected_type="adverb")

    if token.upos in ("NOUN", "PROPN"):
        try:
            return MorphologicalForm(analytic_type="noun", **_nominal_properties(token))
        except ValidationError as exc:
            raise UnmappableTokenError(
                token, f"tabulaedspy rejected mapped properties for 'noun': {exc}"
            ) from exc

    if token.upos in ("PRON", "DET"):
        try:
            return MorphologicalForm(analytic_type="pronoun", **_nominal_properties(token))
        except ValidationError as exc:
            raise UnmappableTokenError(
                token, f"tabulaedspy rejected mapped properties for 'pronoun': {exc}"
            ) from exc

    if token.upos == "ADJ":
        return _build_adjective(token)

    return _build_uninflected_or_unknown(token)


@dataclass(frozen=True)
class MorphologicalFormResult:
    """One token's stage-3 outcome: exactly one of ``result`` (success --
    a :data:`StageThreeResult`, so possibly a companion type rather than a
    genuine tabulaedspy object; see ``.native``) or ``error`` (failure) is
    set. Returned by :func:`build_morphological_forms` so a caller can
    separate outcomes without wrapping every token in its own try/except.
    """

    token: TokenMorphology
    result: Optional[StageThreeResult] = None
    error: Optional[UnmappableTokenError] = None

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def native(self) -> bool:
        """Whether ``.result`` is a genuine ``tabulaedspy.MorphologicalForm``
        rather than one of this module's own companion types
        (:class:`AbbreviatedAdjective`, :class:`UnclassifiedUninflected`).
        ``False`` for a failed mapping too (``.result`` is ``None``)."""
        return isinstance(self.result, MorphologicalForm)


def build_morphological_forms(tokens: Iterable[TokenMorphology]) -> List[MorphologicalFormResult]:
    """Map an iterable of :class:`TokenMorphology` (e.g. one sentence from
    :func:`latincymorph.extraction.extract_sentences`) to
    :class:`MorphologicalFormResult` records, one per token, in order.

    Never raises: a mapping failure for one token is captured in that
    token's ``.error`` rather than aborting the whole batch, since a
    corpus-level run should be able to report how many tokens mapped
    cleanly (and how many only as a companion type) rather than dying on
    the first exception.
    """
    results: List[MorphologicalFormResult] = []
    for token in tokens:
        try:
            results.append(MorphologicalFormResult(token=token, result=build_morphological_form(token)))
        except UnmappableTokenError as exc:
            results.append(MorphologicalFormResult(token=token, error=exc))
    return results
