"""Stage 2 of the latincymorph pipeline: walk a spaCy ``Doc`` sentence by
sentence and pull out the morphological data stage 3 needs, and nothing
else.

This module only imports spaCy -- not tabulaedspy, dspy, or arsgrammatica --
so raw morphology can be extracted from any spaCy ``Doc`` (one produced by
the real ``la_core_web_lg`` pipeline, a smaller/faster LatinCy model, or a
hand-built ``Doc`` in a test) without installing tabulaedspy's heavier
dependency chain at all.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from spacy.tokens import Doc, Token


@dataclass(frozen=True)
class TokenMorphology:
    """A snapshot of one analyzable token's morphology, detached from the
    spaCy objects that produced it so it can be inspected, serialized, or
    handed to a stage-3 mapping function without keeping the whole ``Doc``
    (and its vocab/model) alive.
    """

    text: str
    lemma: str
    upos: str
    """spaCy/UD coarse part-of-speech tag, e.g. ``"NOUN"``, ``"VERB"``."""
    feats: Dict[str, str]
    """UD morphological features as a plain dict, e.g. ``{"Case": "Nom",
    "Gender": "Fem", "Number": "Sing"}`` -- spaCy's own
    ``Token.morph.to_dict()`` output."""
    sent_index: int
    """0-based index of the sentence this token belongs to within its
    document (``enumerate(doc.sents)`` order)."""
    token_index: int
    """This token's index (``Token.i``) within the whole ``Doc``."""

    def get(self, feature: str) -> Optional[str]:
        """Convenience accessor mirroring ``dict.get`` for one UD feature,
        e.g. ``token_morph.get("Case")``."""
        return self.feats.get(feature)


def is_analyzable(token: "Token") -> bool:
    """Whether a token should be assigned morphology at all.

    Skips punctuation and whitespace tokens. This mirrors the exclusion
    tabulaedspy's own ``is_analyzable()`` applies to ``arsgrammatica``
    tokens -- morphology_scheme.md's "Syntactic context" section pairs a
    morphological analysis with every token that is *not*
    ``tokentype == 'punctuation'`` -- so that a token latincymorph extracts
    morphology for is exactly the kind of token tabulaedspy's own scheme
    expects one for. spaCy whitespace tokens have no analog in that scheme
    at all and are excluded too.
    """
    return not (token.is_punct or token.is_space)


def _token_morphology(token: "Token", sent_index: int) -> TokenMorphology:
    return TokenMorphology(
        text=token.text,
        lemma=token.lemma_,
        upos=token.pos_,
        feats=dict(token.morph.to_dict()),
        sent_index=sent_index,
        token_index=token.i,
    )


def extract_sentences(doc: "Doc") -> List[List[TokenMorphology]]:
    """Walk ``doc.sents`` and return one list of :class:`TokenMorphology`
    per sentence, in document order, skipping non-analyzable tokens
    (:func:`is_analyzable`).

    This is stage 2 of the latincymorph pipeline: cycle through the
    document's tokens sentence by sentence, and extract the morphological
    data only (no instantiation of anything from tabulaedspy happens
    here -- see ``tabulaedspy_bridge`` for stage 3).
    """
    return [
        [_token_morphology(token, sent_index) for token in sent if is_analyzable(token)]
        for sent_index, sent in enumerate(doc.sents)
    ]


def iter_tokens(doc: "Doc") -> Iterator[TokenMorphology]:
    """Flat iterator over every analyzable token's morphology, in document
    order, for callers that don't need the sentence grouping
    :func:`extract_sentences` returns."""
    for sentence in extract_sentences(doc):
        yield from sentence
