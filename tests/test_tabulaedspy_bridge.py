import pytest
from tabulaedspy import MorphologicalForm

from latincymorph.extraction import TokenMorphology
from latincymorph.tabulaedspy_bridge import (
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
# One success case per analytic type
# ---------------------------------------------------------------------------


def test_noun():
    form = build_morphological_form(
        tm("caelum", "NOUN", {"Case": "Acc", "Gender": "Neut", "Number": "Sing"})
    )
    assert form == MorphologicalForm(analytic_type="noun", gender="neuter", case="accusative", number="singular")


def test_propn_maps_to_noun():
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
    form = build_morphological_form(
        tm("hic", "DET", {"Case": "Nom", "Gender": "Masc", "Number": "Sing"})
    )
    assert form.analytic_type == "pronoun"


def test_adjective_with_explicit_degree():
    form = build_morphological_form(
        tm("bonus", "ADJ", {"Case": "Nom", "Gender": "Masc", "Number": "Sing", "Degree": "Sup"})
    )
    assert form == MorphologicalForm(
        analytic_type="adjective", gender="masculine", case="nominative", number="singular", degree="superlative"
    )


def test_adjective_defaults_to_positive_degree_when_unmarked():
    form = build_morphological_form(
        tm("bonus", "ADJ", {"Case": "Nom", "Gender": "Masc", "Number": "Sing"})
    )
    assert form.degree == "positive"


def test_adverb_with_degree():
    form = build_morphological_form(tm("laetissime", "ADV", {"Degree": "Sup"}))
    assert form == MorphologicalForm(analytic_type="adverb", degree="superlative")


def test_adverb_without_degree_is_uninflected():
    form = build_morphological_form(tm("cras", "ADV", {}))
    assert form == MorphologicalForm(analytic_type="uninflected", uninflected_type="adverb")


def test_finite_verb():
    form = build_morphological_form(
        tm(
            "creavit",
            "VERB",
            {
                "VerbForm": "Fin",
                "Tense": "Past",
                "Aspect": "Perf",
                "Mood": "Ind",
                "Voice": "Act",
                "Person": "3",
                "Number": "Sing",
            },
        )
    )
    assert form == MorphologicalForm(
        analytic_type="finite verb", tense="perfect", mood="indicative", voice="active", person="third", number="singular"
    )


def test_finite_verb_via_aux():
    form = build_morphological_form(
        tm(
            "est",
            "AUX",
            {
                "VerbForm": "Fin",
                "Tense": "Pres",
                "Mood": "Ind",
                "Voice": "Act",
                "Person": "3",
                "Number": "Sing",
            },
        )
    )
    assert form.analytic_type == "finite verb"
    assert form.tense == "present"


def test_infinitive():
    form = build_morphological_form(
        tm("amare", "VERB", {"VerbForm": "Inf", "Tense": "Pres", "Voice": "Act"})
    )
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


@pytest.mark.parametrize(
    "upos,expected_type",
    [
        ("ADP", "preposition"),
        ("CCONJ", "conjunction"),
        ("SCONJ", "conjunction"),
        ("PART", "particle"),
        ("INTJ", "interjection"),
        ("NUM", "number"),
    ],
)
def test_uninflected_types(upos, expected_type):
    form = build_morphological_form(tm("x", upos, {}))
    assert form == MorphologicalForm(analytic_type="uninflected", uninflected_type=expected_type)


# ---------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------


def test_missing_required_feature_raises():
    with pytest.raises(UnmappableTokenError, match="Gender"):
        build_morphological_form(tm("caelum", "NOUN", {"Case": "Acc", "Number": "Sing"}))


def test_locative_case_is_unmappable():
    with pytest.raises(UnmappableTokenError, match="Case"):
        build_morphological_form(
            tm("Romae", "NOUN", {"Case": "Loc", "Gender": "Fem", "Number": "Sing"})
        )


def test_unrecognized_verbform_raises():
    with pytest.raises(UnmappableTokenError, match="VerbForm"):
        build_morphological_form(tm("mystery", "VERB", {"VerbForm": "Weird"}))


def test_unrecognized_tense_aspect_combination_raises():
    with pytest.raises(UnmappableTokenError, match="Tense/Aspect"):
        build_morphological_form(
            tm(
                "x",
                "VERB",
                {
                    "VerbForm": "Fin",
                    "Tense": "Pres",
                    "Aspect": "Perf",  # not a real combination in the table
                    "Mood": "Ind",
                    "Voice": "Act",
                    "Person": "3",
                    "Number": "Sing",
                },
            )
        )


def test_unmapped_upos_raises():
    with pytest.raises(UnmappableTokenError, match="SPACE"):
        build_morphological_form(tm(" ", "SPACE", {}))


def test_supine_with_invalid_case_reports_tabulaedspy_validation_error():
    # Nominative is a mappable Case value in general, but tabulaedspy's own
    # validator rejects it for a supine (only accusative/ablative allowed).
    with pytest.raises(UnmappableTokenError, match="rejected mapped properties"):
        build_morphological_form(tm("amatum", "VERB", {"VerbForm": "Sup", "Case": "Nom"}))


# ---------------------------------------------------------------------------
# Batch helper
# ---------------------------------------------------------------------------


def test_build_morphological_forms_separates_successes_and_failures():
    tokens = [
        tm("caelum", "NOUN", {"Case": "Acc", "Gender": "Neut", "Number": "Sing"}),
        tm("Romae", "NOUN", {"Case": "Loc", "Gender": "Fem", "Number": "Sing"}),
    ]
    results = build_morphological_forms(tokens)

    assert len(results) == 2
    assert results[0].ok and results[0].form is not None and results[0].error is None
    assert not results[1].ok and results[1].form is None
    assert isinstance(results[1].error, UnmappableTokenError)
