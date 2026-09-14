from latincymorph.extraction import TokenMorphology, extract_sentences, iter_tokens


def test_extract_sentences_skips_punctuation_and_groups_by_sentence(doc_factory):
    doc = doc_factory(
        [
            {"text": "Gallia", "lemma": "Gallia", "pos": "PROPN", "morph": "Case=Nom|Gender=Fem|Number=Sing", "sent_start": True},
            {"text": "est", "lemma": "sum", "pos": "AUX", "morph": "Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin|Voice=Act", "sent_start": False},
            {"text": ".", "lemma": ".", "pos": "PUNCT", "morph": "", "sent_start": False},
            {"text": "Venit", "lemma": "venio", "pos": "VERB", "morph": "Mood=Ind|Number=Sing|Person=3|Tense=Past|Aspect=Perf|VerbForm=Fin|Voice=Act", "sent_start": True},
            {"text": "!", "lemma": "!", "pos": "PUNCT", "morph": "", "sent_start": False},
        ]
    )

    sentences = extract_sentences(doc)

    assert len(sentences) == 2
    assert [t.text for t in sentences[0]] == ["Gallia", "est"]
    assert [t.text for t in sentences[1]] == ["Venit"]

    first = sentences[0][0]
    assert isinstance(first, TokenMorphology)
    assert first.lemma == "Gallia"
    assert first.upos == "PROPN"
    assert first.feats == {"Case": "Nom", "Gender": "Fem", "Number": "Sing"}
    assert first.get("Case") == "Nom"
    assert first.get("Degree") is None
    assert first.sent_index == 0

    second_sentence_token = sentences[1][0]
    assert second_sentence_token.sent_index == 1
    assert second_sentence_token.token_index == 3  # doc-wide index, not per-sentence


def test_iter_tokens_flattens_across_sentences(doc_factory):
    doc = doc_factory(
        [
            {"text": "Gallia", "pos": "PROPN", "morph": "Case=Nom|Gender=Fem|Number=Sing"},
            {"text": ".", "pos": "PUNCT"},
            {"text": "Venit", "pos": "VERB", "morph": "Tense=Past|Aspect=Perf|Mood=Ind|Voice=Act|Person=3|Number=Sing|VerbForm=Fin", "sent_start": True},
        ]
    )

    flattened = list(iter_tokens(doc))
    assert [t.text for t in flattened] == ["Gallia", "Venit"]
