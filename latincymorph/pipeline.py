"""Stage 1 of the latincymorph pipeline: load a LatinCy spaCy pipeline and
run it over Latin text.

LatinCy models (``la_core_web_sm``/``md``/``lg``/``trf``) are published as
pip-installable wheels on Hugging Face rather than on PyPI, e.g.::

    pip install "la-core-web-lg @ https://huggingface.co/latincy/la_core_web_lg/resolve/main/la_core_web_lg-any-py3-none-any.whl"

Installing that wheel registers the model as an importable package, so
``spacy.load(name)`` finds it exactly the way a ``python -m spacy download``
model would. This module does not install anything itself; see
``notes/pipeline-overview.md`` for the install command and a note on why it
could not be exercised against the real model weights in this development
environment (huggingface.co is not reachable through the sandbox's network
egress policy -- see that note for how this was worked around for testing).
"""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

import spacy
from spacy.language import Language
from spacy.tokens import Doc

#: Default LatinCy model name. Override per-call with ``model_name=`` if you
#: need a different size (``la_core_web_sm``/``md``/``trf``) or a citation
#: corpus's own custom pipeline.
DEFAULT_MODEL = "la_core_web_lg"


@lru_cache(maxsize=None)
def load_model(model_name: str = DEFAULT_MODEL) -> Language:
    """Load (and cache) a LatinCy spaCy pipeline by name.

    Raises spaCy's own ``OSError`` with its normal "can't find model"
    message if ``model_name`` isn't installed -- see this module's
    docstring for the install command. Results are cached per
    ``model_name`` (``functools.lru_cache``) so repeated calls -- from a
    notebook cell re-run, or a loop processing many documents -- don't
    reload the model's weights every time.
    """
    return spacy.load(model_name)


def analyze_text(
    text: str, *, nlp: Optional[Language] = None, model_name: str = DEFAULT_MODEL
) -> Doc:
    """Run a LatinCy pipeline over ``text`` and return the resulting
    ``Doc``.

    Pass an already-loaded ``nlp`` (e.g. from :func:`load_model`, or your
    own customized pipeline) to reuse it across many calls; otherwise one
    is loaded -- and cached -- by ``model_name``.
    """
    pipeline = nlp if nlp is not None else load_model(model_name)
    return pipeline(text)
