# Pipeline overview

Status: first draft implementation, unit-tested against hand-built spaCy
`Doc` objects. Not yet run against the real `la_core_web_lg` model weights
-- see "What couldn't be verified here" below.

## The three stages

The package implements the three-stage pipeline described in the project
brief, as three modules with a clean dependency boundary between stage 2
and stage 3:

1. **`latincymorph.pipeline`** -- load a LatinCy spaCy pipeline
   (`load_model()`) and run it over a string of Latin text
   (`analyze_text()`), returning a spaCy `Doc`. Needs only `spacy` and an
   installed LatinCy model.
2. **`latincymorph.extraction`** -- walk `doc.sents` and, for each
   sentence, build one `TokenMorphology` per analyzable token
   (`extract_sentences()`), skipping punctuation and whitespace tokens
   (`is_analyzable()`). A `TokenMorphology` is a plain, immutable
   dataclass (`text`, `lemma`, `upos`, `feats: dict`, `sent_index`,
   `token_index`) -- deliberately detached from spaCy's own `Token`/`Doc`
   objects so it's cheap to inspect, log, or serialize. **This module
   imports only `spacy`.**
3. **`latincymorph.tabulaedspy_bridge`** -- map one `TokenMorphology`'s UD
   features onto tabulaedspy's scheme, per `quarto/reference/stringmappings.qmd`
   (Neel's own hand-maintained reference at the repository root -- the
   authoritative source for this stage; see
   `spacy-to-tabulaedspy-mapping.md` for how the code follows it).
   `build_morphological_form()` returns a genuine
   `tabulaedspy.MorphologicalForm` for most tokens, but two local
   companion types stand in when tabulaedspy's schema demands a property
   LatinCy doesn't tag: `AbbreviatedAdjective` (an adjective's
   gender/case/number without the `degree` tabulaedspy requires but
   LatinCy doesn't supply) and `UnclassifiedUninflected` (a token that
   isn't confidently any of the 11 analytic types, carrying its raw UPOS
   and features instead of a guess). `MorphologicalFormResult.native`
   tells a caller which kind it got. `build_morphological_forms()` maps a
   whole sentence at once, collecting per-token results (including
   genuine failures, in `.error`) rather than raising on the first bad
   token. **This is the only module that imports tabulaedspy**, and
   therefore the only one that pulls in tabulaedspy's own dependencies
   (`dspy`, `arsgrammatica`).

`latincymorph.analyze_document(text)` chains all three for convenience.
Import `latincymorph.extraction` directly (not the top-level package) if
you want stages 1-2 without tabulaedspy/dspy/arsgrammatica installed at
all -- e.g. for exploratory tagging work that has nothing to do with
tabulaedspy's own scheme.

Install the `tabulaedspy` extra (`pip install -e ".[tabulaedspy]")` to pull
in stage 3's dependencies; plain `pip install -e .` gets you stages 1-2
only.

## Why stage 3 only targets `MorphologicalForm`, not a full analysis

tabulaedspy's own scheme (`morphology_scheme.md` in that repository) pairs
four things with a syntactically-analyzed token: a *lemma*, a *normalized
form*, a *named entity* structure, and a *morphological form*. Its own
`MorphologicalAnalysis`/`MorphologicallyAnalyzedToken` models bundle all
four together with an `arsgrammatica.TokenAnalysis` (tabulaedspy's syntactic
input, produced by a different tool entirely). (The two local companion
types stage 3 sometimes produces instead -- see above -- are latincymorph's
own, not part of tabulaedspy's scheme at all; they exist only because
tabulaedspy's `MorphologicalForm` itself can't represent a token missing a
property it requires.)

latincymorph only has spaCy's output to work with, and spaCy's tagger
doesn't produce a named-entity *grouping* in tabulaedspy's sense (which of
several adjacent tokens spell one multi-word name), nor an `arsgrammatica`
syntactic analysis to pair anything with. So the bridge targets exactly
`tabulaedspy.MorphologicalForm` -- the one piece of tabulaedspy's schema
that a token's own morphology (independent of syntactic context or
named-entity status) is enough to build. Producing a full
`MorphologicalAnalysis`/`MorphologicallyAnalyzedToken` would need at least
a separate NER-grouping step and, for tabulaedspy's own citation
machinery, `arsgrammatica` `TokenAnalysis` objects this pipeline never
constructs.

## Mapping fidelity -- read before trusting this at scale

tabulaedspy's own `analyze_sentence()`/`analyze_document()` ask a language
model to *disambiguate* a token's morphology using its syntactic context:
morphology_scheme.md's own example is a neuter noun that is morphologically
nominative/accusative/vocative getting resolved to exactly one case by its
syntactic role. LatinCy's tagger/morphologizer is also context-sensitive
(it's a trained sequence model, not a lookup table), but it commits to *UD*
tags, not tabulaedspy's scheme directly.

`tabulaedspy_bridge.py` is therefore a **second, independent, deterministic
translation** from spaCy/UD's already-committed tags to tabulaedspy's
already-committed schema -- it does not, and cannot, re-run or reproduce
tabulaedspy's own context-based disambiguation. If LatinCy's tagger itself
gets a token's case wrong, or leaves a feature unset that tabulaedspy's
schema requires, that error/gap passes straight through. See
`spacy-to-tabulaedspy-mapping.md` for the full per-feature mapping tables,
where each one comes from, and which parts are still educated guesses
rather than confirmed facts.

## What couldn't be verified here

This sandbox's network egress policy blocks `huggingface.co` (confirmed
directly: `curl` to `huggingface.co:443` is rejected by the egress proxy
with "organization policy"), and the LatinCy wheels are hosted only there
(`la_core_web_lg-3.9.6-py3-none-any.whl`, ~297 MB) -- not on PyPI. That
means:

- **The real model was never installed or run in this environment.** No
  code here has been checked against actual `la_core_web_lg` tagger
  output on real Latin text.
- The UD tag/feature values in `tabulaedspy_bridge.py` (e.g. the exact
  `Tense`/`Aspect` combinations, which UPOS values the model emits, which
  features it leaves unset by default) come from the model card
  (`https://huggingface.co/latincy/la_core_web_lg` README, fetched via web
  search/fetch rather than pip-installed) and from the published
  Universal Dependencies feature conventions for the Latin/Ancient Greek
  PROIEL-style treebanks LatinCy trains on -- not from running the model.

**Before relying on this pipeline for real corpus work**, install
`la_core_web_lg` somewhere with unrestricted network access (see the
install command in `latincymorph/pipeline.py`'s module docstring), run it
over a handful of representative sentences, and diff `token.morph` against
the assumptions in `spacy-to-tabulaedspy-mapping.md` -- particularly the
`Tense`/`Aspect` table and the `ADV` degree-presence heuristic, which are
the two most speculative parts of the mapping. `tests/` exercises the
mapping logic thoroughly against hand-built `Doc`s with the *assumed* tag
strings, which is a solid regression net once those strings are confirmed,
but currently proves the code is internally consistent, not that it
matches the model's real output.

## Testing approach

- `tests/conftest.py`'s `make_doc()` builds a spaCy `Doc` by hand (via
  `Doc(vocab, words=..., pos=..., lemmas=..., morphs=..., sent_starts=...)`)
  instead of running a trained pipeline, so `extraction.py` is tested
  against the real spaCy `Token`/`Doc` API without needing model weights.
- `tests/test_tabulaedspy_bridge.py` constructs `TokenMorphology` records
  directly (bypassing spaCy entirely) and checks one success case per
  tabulaedspy analytic type, plus the failure paths (missing feature,
  unmappable value, tabulaedspy's own validator rejecting an
  internally-inconsistent combination like a nominative supine).
- Both suites run against a real installed `tabulaedspy` (from
  `git+https://github.com/neelsmith/tabulaedspy`), so the assertions are
  checked against tabulaedspy's actual `MorphologicalForm` validator, not
  a mock of it.
- Run with `pytest` from the repository root (`pythonpath = ["."]` in
  `pyproject.toml` makes `import latincymorph` work without an install).
