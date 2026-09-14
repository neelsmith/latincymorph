"""Shared test fixtures.

The real LatinCy model (``la_core_web_lg``) is distributed only from Hugging
Face, which this development sandbox's network egress policy blocks (see
``notes/pipeline-overview.md``). Tests therefore build spaCy ``Doc`` objects
by hand -- setting ``pos``, ``morph``, and ``lemmas`` directly via spaCy's
own ``Doc`` constructor, the same mechanism spaCy's own test suite uses for
gold-standard annotations -- rather than running a trained pipeline. This
exercises the exact same ``Token``/``Doc`` API surface
(``extraction.py``/``tabulaedspy_bridge.py`` only ever read ``token.text``,
``token.lemma_``, ``token.pos_``, ``token.morph``, ``token.is_punct``,
``token.is_space``, and ``doc.sents``) without needing the trained weights
at all.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import pytest
import spacy
from spacy.tokens import Doc

_VOCAB_NLP = spacy.blank("la")


def make_doc(token_specs: List[Dict]) -> Doc:
    """Build a ``Doc`` from a list of per-token dicts, each with keys
    ``text`` (required), ``pos``, ``lemma``, ``morph`` (a UD ``FEATS``
    string, e.g. ``"Case=Nom|Gender=Fem|Number=Sing"``), and
    ``sent_start`` (bool; defaults to ``True`` for the first token and
    ``False`` for every other token if no spec sets one explicitly).
    """
    words = [spec["text"] for spec in token_specs]
    pos = [spec.get("pos", "X") for spec in token_specs]
    lemmas = [spec.get("lemma", spec["text"].lower()) for spec in token_specs]
    morphs = [spec.get("morph", "") for spec in token_specs]

    if any("sent_start" in spec for spec in token_specs):
        sent_starts = [bool(spec.get("sent_start", False)) for spec in token_specs]
        sent_starts[0] = True
    else:
        sent_starts = [i == 0 for i in range(len(token_specs))]

    return Doc(
        _VOCAB_NLP.vocab,
        words=words,
        pos=pos,
        lemmas=lemmas,
        morphs=morphs,
        sent_starts=sent_starts,
    )


@pytest.fixture
def doc_factory():
    return make_doc
