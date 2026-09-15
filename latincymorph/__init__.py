"""latincymorph: analyze Latin text with a LatinCy spaCy pipeline and
convert its per-token morphology into tabulaedspy ``MorphologicalForm``
objects.

Three stages (full design in ``notes/pipeline-overview.md``):

1. ``pipeline.analyze_text()`` -- run a LatinCy spaCy model (``nlp``) over
   a string of Latin text.
2. ``extraction.extract_sentences()`` -- walk the resulting ``Doc``
   sentence by sentence and pull out each analyzable token's morphology.
3. ``tabulaedspy_bridge.build_morphological_form()`` -- map that
   morphology onto tabulaedspy's scheme and instantiate the result. Most
   tokens become a genuine ``tabulaedspy.MorphologicalForm``; some become
   one of two local companion types (``AbbreviatedAdjective``,
   ``UnclassifiedUninflected``) for cases tabulaedspy's schema can't
   represent from what LatinCy actually tags -- see
   ``tabulaedspy_bridge``'s own docstring and
   ``notes/spacy-to-tabulaedspy-mapping.md``.

Stages 1-2 need only spaCy/LatinCy installed. Stage 3 additionally needs
tabulaedspy (and, transitively, dspy and arsgrammatica) -- install
latincymorph with the ``tabulaedspy`` extra to get it. Importing this
top-level package pulls stage 3 in for the ``analyze_document()``
convenience function below; import ``latincymorph.extraction`` directly
instead if you only want stages 1-2 without those heavier dependencies.
"""
from __future__ import annotations

from typing import List, Optional

from .extraction import TokenMorphology, extract_sentences, is_analyzable, iter_tokens
from .pipeline import DEFAULT_MODEL, analyze_text, load_model
from .tabulaedspy_bridge import (
    AbbreviatedAdjective,
    MorphologicalFormResult,
    StageThreeResult,
    UnclassifiedUninflected,
    UnmappableTokenError,
    build_morphological_form,
    build_morphological_forms,
)

__all__ = [
    "DEFAULT_MODEL",
    "TokenMorphology",
    "MorphologicalFormResult",
    "StageThreeResult",
    "AbbreviatedAdjective",
    "UnclassifiedUninflected",
    "UnmappableTokenError",
    "analyze_text",
    "load_model",
    "extract_sentences",
    "iter_tokens",
    "is_analyzable",
    "build_morphological_form",
    "build_morphological_forms",
    "analyze_document",
]


def analyze_document(
    text: str, *, nlp=None, model_name: str = DEFAULT_MODEL
) -> List[List[MorphologicalFormResult]]:
    """Run the full three-stage pipeline over ``text`` and return one list
    of :class:`MorphologicalFormResult` per sentence, in document order.

    Equivalent to chaining :func:`analyze_text`, :func:`extract_sentences`,
    and :func:`build_morphological_forms` yourself; call those directly
    instead for more control -- e.g. reusing one loaded ``nlp`` across many
    documents, or skipping stage 3 (and its tabulaedspy/dspy dependency)
    entirely.
    """
    doc = analyze_text(text, nlp=nlp, model_name=model_name)
    return [build_morphological_forms(sentence) for sentence in extract_sentences(doc)]
