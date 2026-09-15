import pytest
from tabulaedspy import MorphologicalForm

from latincymorph.extraction import TokenMorphology
from latincymorph.tabulaedspy_bridge import (
    AbbreviatedAdjective,
    UnclassifiedUninflected,
    UnmappableTokenError,
    build_morphological_form,
    build_morphological_forms,
)


def tm(text, upos, feats, lemma=None):
    return TokenMorphology(
        text=text,
        lemma=lemma or text.lower(),
        upos=upos,
        feats=feats,
        sent_index=0,
        token_index=0,
    )


# ---------------------------------------------------------------------------
# Nominal types
# ---------------------------------------------------------------------------


def test_noun():
    form = build_morphological_form(
        tm("caelum", "NOUN", {"Case": "Acc", "Gender": "Neut", "Number": "Sing"})
    )
    assert form == MorphologicalForm(analytic_type="noun", gender="neuter", case="accusative", number="singular")


def test_propn_maps_to_noun():
    # stringmappings.qmd's "Analytic type" table now lists NOUN or PROPN
    # -> noun directly (Neel added PROPN after this module first treated
    # it as an inference) -- see
    # tabulaedspy_bridge.build_morphological_form's docstring, point 3.
    form = build_morphological_form(
        tm("Deus", "PROPN", {"Case": "Nom", "Gender": "Masc", "Number": "Sing"})
    )
    assert form.analytic_type == "noun"


def test_pronoun():
    form = build_morphological_form(
        tm("quis", "PRON", {"Case": "Nom", "Gender": "Masc", "Number": "Sing"})
    )
    assert form.analytic_type == "pronoun"


def test_det_maps_to_pronoun():
    # Same situation as PROPN -- see docstring point 4.
    form = build_morphological_form(
        tm("hic", "DET", {"Case": "Nom", "Gender": "Masc", "Number": "Sing"})
    )
    assert form.analytic_type == "pronoun"


# ---------------------------------------------------------------------------
# Adjective: real MorphologicalForm when Degree is present, otherwise
# AbbreviatedAdjective (stringmappings.qmd's "Degree" note: LatinCy
# generally doesn't tag it at all).
# ---------------------------------------------------------------------------


def test_adjective_with_degree_is_a_real_tabulaedspy_form():
    form = build_morphological_form(
        tm("bonus", "ADJ", {"Case": "Nom", "Gender": "Masc", "Number": "Sing", "Degree": "Sup"})
    )
    assert form == MorphologicalForm(
        analytic_type="adjective", gender="masculine", case="nominative", number="singular", degree="superlative"
    )
    assert isinstance(form, MorphologicalForm)


def test_adjective_without_degree_is_abbreviated():
    result = build_morphological_form(
        tm("bonus", "ADJ", {"Case": "Nom", "Gender": "Masc", "Number": "Sing"})
    )
    assert isinstance(result, AbbreviatedAdjective)
    assert result == AbbreviatedAdjective(gender="masculine", case="nominative", number="singular")
    assert not hasattr(result, "degree")


def test_abbreviated_adjective_still_requires_gender_case_number():
    with pytest.raises(UnmappableTokenError, match="Gender"):
        build_morphological_form(tm("bonus", "ADJ", {"Case": "Nom", "Number": "Sing"}))


# ---------------------------------------------------------------------------
# Adverb: always uninflected/adverb, regardless of morph content (never the
# `adverb` analytic type, since degree is never reliably available).
# ---------------------------------------------------------------------------


def test_adverb_with_empty_morph_is_uninflected():
    form = build_morphological_form(tm("cras", "ADV", {}))
    assert form == MorphologicalForm(analytic_type="uninflected", uninflected_type="adverb")


def test_adverb_with_nonempty_morph_is_still_uninflected():
    # Even if some morph feature (e.g. a stray Degree) shows up, ADV never
    # becomes analytic_type="adverb" -- see the module docstring.
    form = build_morphological_form(tm("laetissime", "ADV", {"Degree": "Sup"}))
    assert form == MorphologicalForm(analytic_type="uninflected", uninflected_type="adverb")


# ---------------------------------------------------------------------------
# Verb family, including the corrected Tense/Aspect rules from
# stringmappings.qmd (present and pluperfect check only one feature; the
# other four need a specific Tense+Aspect pair; "future" pairs Fut with
# Imp, not Prosp).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tense_feats,expected_tense",
    [
        ({"Tense": "Pres"}, "present"),
        ({"Tense": "Past", "Aspect": "Imp"}, "imperfect"),
        ({"Tense": "Fut", "Aspect": "Imp"}, "future"),
        ({"Tense": "Past", "Aspect": "Perf"}, "perfect"),
        ({"Tense": "Pqp"}, "pluperfect"),
        ({"Tense": "Fut", "Aspect": "Perf"}, "future_perfect"),
    ],
)
def test_finite_verb_tense_aspect_table(tense_feats, expected_tense):
    feats = {"VerbForm": "Fin", "Mood": "Ind", "Voice": "Act", "Person": "3", "Number": "Sing", **tense_feats}
    form = build_morphological_form(tm("x", "VERB", feats))
    assert form.tense == expected_tense


def test_finite_verb_via_aux():
    # VerbForm is checked against token.morph regardless of UPOS -- an AUX
    # copula with VerbForm=Fin is still a finite verb.
    form = build_morphological_form(
        tm("est", "AUX", {"VerbForm": "Fin", "Tense": "Pres", "Mood": "Ind", "Voice": "Act", "Person": "3", "Number": "Sing"})
    )
    assert form.analytic_type == "finite verb"
    assert form.tense == "present"


# stringmappings.qmd's "Analytic type" table now lists `AUX -> finite
# verb` directly. Read as a fallback (see build_morphological_form's own
# docstring, step 2): an AUX token with no recognized VerbForm still
# becomes a finite verb, but VerbForm still wins when it IS present, so
# an infinitive or participle copula token keeps its own analytic type.


def test_aux_without_verbform_falls_back_to_finite_verb():
    form = build_morphological_form(
        tm("est", "AUX", {"Tense": "Pres", "Mood": "Ind", "Voice": "Act", "Person": "3", "Number": "Sing"})
    )
    assert form.analytic_type == "finite verb"
    assert form.tense == "present"


def test_aux_with_verbform_inf_is_still_infinitive_not_finite_verb():
    # "esse" -- the copula's own infinitive -- is tagged AUX but must not
    # be swallowed by the new AUX->finite verb fallback.
    form = build_morphological_form(tm("esse", "AUX", {"VerbForm": "Inf", "Aspect": "Imp", "Voice": "Act"}))
    assert form.analytic_type == "infinitive"


# stringmappings.qmd's "Voice" section: "If token.pos_ is AUX, voice is
# active." The copula's own tagger sometimes marks an AUX token Voice=Pass
# when it takes part in a passive periphrastic construction (e.g. "est" in
# "amandus est"), even though the copula itself is always grammatically
# active -- so an AUX token's own Voice feature (if any) is never
# consulted, across all three voice-bearing analytic types.


def test_aux_voice_is_always_active_even_when_tagged_passive():
    form = build_morphological_form(
        tm("est", "AUX", {"VerbForm": "Fin", "Tense": "Pres", "Mood": "Ind", "Voice": "Pass", "Person": "3", "Number": "Sing"})
    )
    assert form.analytic_type == "finite verb"
    assert form.voice == "active"


def test_aux_voice_is_active_without_any_voice_feature():
    # Must not raise UnmappableTokenError for a missing Voice -- an AUX
    # token's voice is always "active" regardless of whether it's tagged
    # at all.
    form = build_morphological_form(
        tm("est", "AUX", {"VerbForm": "Fin", "Tense": "Pres", "Mood": "Ind", "Person": "3", "Number": "Sing"})
    )
    assert form.voice == "active"


def test_aux_infinitive_voice_is_always_active_even_when_tagged_passive():
    form = build_morphological_form(tm("esse", "AUX", {"VerbForm": "Inf", "Aspect": "Imp", "Voice": "Pass"}))
    assert form.analytic_type == "infinitive"
    assert form.voice == "active"


def test_non_aux_verb_voice_still_uses_its_own_voice_feature():
    # Regression check: the AUX special case must not affect ordinary
    # VERB tokens, which still need a real Voice feature.
    with pytest.raises(UnmappableTokenError, match="Voice"):
        build_morphological_form(
            tm("x", "VERB", {"VerbForm": "Fin", "Tense": "Pres", "Mood": "Ind", "Person": "3", "Number": "Sing"})
        )


# Infinitive tense is determined by Aspect alone (not Tense) -- same
# pattern as participles (see above). stringmappings.qmd's "Tense" table
# now has "for infinitives, Aspect=Imp -> present" and "for infinitives,
# Aspect=Perf -> perfect"; there is no infinitive future entry (Latin has
# no synthetic future infinitive).


@pytest.mark.parametrize(
    "aspect,expected_tense",
    [
        ("Imp", "present"),
        ("Perf", "perfect"),
    ],
)
def test_infinitive_tense_from_aspect_alone(aspect, expected_tense):
    form = build_morphological_form(tm("x", "VERB", {"VerbForm": "Inf", "Aspect": aspect, "Voice": "Act"}))
    assert form.analytic_type == "infinitive"
    assert form.tense == expected_tense


def test_infinitive_tense_ignores_tense_feature():
    # Tense=Fut would mean something for a finite verb, but must not
    # affect an infinitive's tense -- only Aspect matters.
    form = build_morphological_form(tm("x", "VERB", {"VerbForm": "Inf", "Tense": "Fut", "Aspect": "Perf", "Voice": "Act"}))
    assert form.analytic_type == "infinitive"
    assert form.tense == "perfect"


def test_infinitive_without_aspect_feature_raises():
    with pytest.raises(UnmappableTokenError, match="infinitive Aspect"):
        build_morphological_form(tm("amare", "VERB", {"VerbForm": "Inf", "Tense": "Pres", "Voice": "Act"}))


def test_infinitive_with_unrecognized_aspect_raises():
    with pytest.raises(UnmappableTokenError, match="infinitive Aspect"):
        build_morphological_form(tm("x", "VERB", {"VerbForm": "Inf", "Aspect": "Prosp", "Voice": "Act"}))


def test_infinitive():
    form = build_morphological_form(tm("amare", "VERB", {"VerbForm": "Inf", "Aspect": "Imp", "Voice": "Act"}))
    assert form == MorphologicalForm(analytic_type="infinitive", tense="present", voice="active")


def test_participle():
    form = build_morphological_form(
        tm(
            "amatus",
            "VERB",
            {
                "VerbForm": "Part",
                "Tense": "Past",
                "Aspect": "Perf",
                "Voice": "Pass",
                "Case": "Nom",
                "Gender": "Masc",
                "Number": "Sing",
            },
        )
    )
    assert form == MorphologicalForm(
        analytic_type="participle", tense="perfect", voice="passive", gender="masculine", case="nominative", number="singular"
    )


# Participle tense is determined by Aspect alone (not Tense) -- per
# stringmappings.qmd's "Tense" table, LatinCy tags a participle's tense
# through Aspect=Imp/Prosp/Perf (present/future/perfect), the reverse of
# finite verbs where Tense is primary. VerbForm=Part always maps to
# analytic_type="participle" regardless of whatever Tense value (if any)
# happens to be present -- Tense is never consulted for a participle.


@pytest.mark.parametrize(
    "aspect,expected_tense",
    [
        ("Imp", "present"),
        ("Prosp", "future"),
        ("Perf", "perfect"),
    ],
)
def test_participle_tense_from_aspect_alone(aspect, expected_tense):
    form = build_morphological_form(
        tm(
            "x",
            "VERB",
            {"VerbForm": "Part", "Aspect": aspect, "Voice": "Act", "Case": "Nom", "Gender": "Masc", "Number": "Sing"},
        )
    )
    assert form.analytic_type == "participle"
    assert form.tense == expected_tense


def test_participle_tense_ignores_tense_feature():
    # A Tense value that would mean something specific for a finite verb
    # (e.g. Tense=Past) must not affect a participle's tense -- only
    # Aspect matters. Aspect=Imp here still means "present" even though
    # Tense=Past is also present in the morphology.
    form = build_morphological_form(
        tm(
            "x",
            "VERB",
            {
                "VerbForm": "Part",
                "Tense": "Past",
                "Aspect": "Imp",
                "Voice": "Act",
                "Case": "Nom",
                "Gender": "Masc",
                "Number": "Sing",
            },
        )
    )
    assert form.analytic_type == "participle"
    assert form.tense == "present"


def test_participle_without_aspect_feature_raises():
    # Tense alone is not enough for a participle -- Aspect is required
    # (the reverse of the earlier, corrected assumption that Tense alone
    # was sufficient and Aspect was irrelevant).
    with pytest.raises(UnmappableTokenError, match="participle Aspect"):
        build_morphological_form(
            tm(
                "amaturus",
                "VERB",
                {"VerbForm": "Part", "Tense": "Fut", "Voice": "Act", "Case": "Nom", "Gender": "Masc", "Number": "Sing"},
            )
        )


def test_participle_with_unrecognized_aspect_raises():
    with pytest.raises(UnmappableTokenError, match="participle Aspect"):
        build_morphological_form(
            tm(
                "x",
                "VERB",
                {"VerbForm": "Part", "Aspect": "Hab", "Voice": "Act", "Case": "Nom", "Gender": "Masc", "Number": "Sing"},
            )
        )


def test_gerund():
    form = build_morphological_form(tm("amandi", "VERB", {"VerbForm": "Ger", "Case": "Gen"}))
    assert form == MorphologicalForm(analytic_type="gerund", case="genitive")


def test_gerundive():
    form = build_morphological_form(
        tm("amandus", "VERB", {"VerbForm": "Gdv", "Case": "Nom", "Gender": "Masc", "Number": "Sing"})
    )
    assert form.analytic_type == "gerundive"


def test_supine():
    form = build_morphological_form(tm("amatum", "VERB", {"VerbForm": "Sup", "Case": "Acc"}))
    assert form == MorphologicalForm(analytic_type="supine", case="accusative")


# ---------------------------------------------------------------------------
# Uninflected: real tabulaedspy forms for the UPOS values
# stringmappings.qmd's table covers (CCONJ and SCONJ both -> conjunction);
# UnclassifiedUninflected for anything else (the catch-all).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "upos,expected_type",
    [
        ("CCONJ", "conjunction"),
        ("SCONJ", "conjunction"),
        ("ADP", "preposition"),
        ("NUM", "number"),
        ("INTJ", "interjection"),
        ("PART", "particle"),
        ("X", "foreign"),
    ],
)
def test_uninflected_types_from_stringmappings_table(upos, expected_type):
    form = build_morphological_form(tm("x", upos, {}))
    assert form == MorphologicalForm(analytic_type="uninflected", uninflected_type=expected_type)


@pytest.mark.parametrize("upos", ["SYM"])
def test_unlisted_upos_falls_back_to_unclassified_uninflected(upos):
    # SYM isn't in stringmappings.qmd's uninflected_type table at all
    # (only X comes close, with its own caveat about being a mixed bag).
    # SCONJ used to land here too, before Neel added it alongside CCONJ
    # (see test_uninflected_types_from_stringmappings_table above). AUX
    # used to land here as well when it had no recognized VerbForm, but
    # stringmappings.qmd now maps bare AUX to finite verb instead (see
    # test_aux_without_verbform_falls_back_to_finite_verb and
    # test_aux_without_recognized_verbform_and_missing_properties_raises).
    result = build_morphological_form(tm("x", upos, {}))
    assert isinstance(result, UnclassifiedUninflected)
    assert result.upos == upos


def test_aux_without_recognized_verbform_and_missing_properties_raises():
    # A bare AUX with no VerbForm and none of a finite verb's other
    # required properties either is a genuine mapping failure (the
    # analytic type is known -- finite verb, via the AUX fallback -- but
    # it has nothing to build one from), not a silent UnclassifiedUninflected.
    with pytest.raises(UnmappableTokenError):
        build_morphological_form(tm("x", "AUX", {}))


def test_unclassified_uninflected_keeps_raw_feats():
    # SCONJ used to be a catch-all example here too, before Neel added it
    # alongside CCONJ -- SYM is still genuinely unmapped.
    result = build_morphological_form(tm("%", "SYM", {"Foo": "Bar"}))
    assert result.feats == {"Foo": "Bar"}


# ---------------------------------------------------------------------------
# Failure paths -- these still raise, because the analytic type IS known
# but a property it specifically requires is missing/invalid.
# ---------------------------------------------------------------------------


def test_missing_required_feature_raises():
    with pytest.raises(UnmappableTokenError, match="Gender"):
        build_morphological_form(tm("caelum", "NOUN", {"Case": "Acc", "Number": "Sing"}))


def test_locative_case_is_unmappable():
    with pytest.raises(UnmappableTokenError, match="Case"):
        build_morphological_form(tm("Romae", "NOUN", {"Case": "Loc", "Gender": "Fem", "Number": "Sing"}))


def test_unrecognized_tense_aspect_combination_raises():
    # Tense=Past with no Aspect at all matches neither the imperfect
    # (Past+Imp) nor the perfect (Past+Perf) rule.
    with pytest.raises(UnmappableTokenError, match="Tense/Aspect"):
        build_morphological_form(
            tm(
                "x",
                "VERB",
                {"VerbForm": "Fin", "Tense": "Past", "Mood": "Ind", "Voice": "Act", "Person": "3", "Number": "Sing"},
            )
        )


def test_present_tense_ignores_aspect_per_stringmappings():
    # stringmappings.qmd's "present" rule checks Tense=Pres alone; an
    # incidental Aspect value shouldn't block it.
    form = build_morphological_form(
        tm(
            "x",
            "VERB",
            {"VerbForm": "Fin", "Tense": "Pres", "Aspect": "Perf", "Mood": "Ind", "Voice": "Act", "Person": "3", "Number": "Sing"},
        )
    )
    assert form.tense == "present"


def test_supine_with_invalid_case_reports_tabulaedspy_validation_error():
    # Nominative is a mappable Case value in general, but tabulaedspy's own
    # validator rejects it for a supine (only accusative/ablative allowed).
    with pytest.raises(UnmappableTokenError, match="rejected"):
        build_morphological_form(tm("amatum", "VERB", {"VerbForm": "Sup", "Case": "Nom"}))


# ---------------------------------------------------------------------------
# MorphologicalFormResult.native and the batch helper
# ---------------------------------------------------------------------------


def test_native_is_true_only_for_genuine_tabulaedspy_forms():
    results = build_morphological_forms(
        [
            tm("caelum", "NOUN", {"Case": "Acc", "Gender": "Neut", "Number": "Sing"}),  # native
            tm("bonus", "ADJ", {"Case": "Nom", "Gender": "Masc", "Number": "Sing"}),  # AbbreviatedAdjective
            tm("%", "SYM", {}),  # UnclassifiedUninflected
            tm("Romae", "NOUN", {"Case": "Loc", "Gender": "Fem", "Number": "Sing"}),  # error
        ]
    )
    assert [r.native for r in results] == [True, False, False, False]
    assert [r.ok for r in results] == [True, True, True, False]


def test_build_morphological_forms_separates_successes_and_failures():
    tokens = [
        tm("caelum", "NOUN", {"Case": "Acc", "Gender": "Neut", "Number": "Sing"}),
        tm("Romae", "NOUN", {"Case": "Loc", "Gender": "Fem", "Number": "Sing"}),
    ]
    results = build_morphological_forms(tokens)

    assert len(results) == 2
    assert results[0].ok and results[0].result is not None and results[0].error is None
    assert not results[1].ok and results[1].result is None
    assert isinstance(results[1].error, UnmappableTokenError)
