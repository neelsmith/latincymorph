# spaCy/UD -> tabulaedspy mapping tables

Reference for `latincymorph/tabulaedspy_bridge.py`. **This document used to
be the primary source for the mapping** (built from the `la_core_web_lg`
model card and general Universal Dependencies documentation, since the real
model couldn't be installed in the environment this pipeline was first
drafted in -- see `pipeline-overview.md`). It has been superseded:

**The authoritative source is now `quarto/reference/stringmappings.qmd`**
(Neel's own hand-maintained reference, at the repository root -- not part
of `notes/`). `tabulaedspy_bridge.py` implements that document as closely
as tabulaedspy's schema allows; this page records where the code follows
it exactly, where it necessarily extends it, and why. Treat
`stringmappings.qmd` as ground truth for the linguistics; treat this page
as the implementation notes.

## Two properties LatinCy doesn't tag: `degree`

`stringmappings.qmd`'s own note under "Degree": *"latincy does not appear
to tag degree of adjectives and adverbs."* But tabulaedspy's schema
requires `degree` for both its `adjective` and `adverb` analytic types.
The two cases are handled differently, because they aren't symmetric:

- **`adjective`** still has three other properties (`gender`, `case`,
  `number`) that LatinCy *does* tag reliably. Discarding those along with
  the missing `degree` would throw away real data. So an `ADJ` token
  without `Degree` becomes `AbbreviatedAdjective` -- a local companion
  type (`latincymorph.tabulaedspy_bridge.AbbreviatedAdjective`) with
  exactly those three fields and no `degree` slot at all, rather than a
  real `tabulaedspy.MorphologicalForm`. If a `Degree` value *is* present
  (LatinCy might start tagging it in a future release, or already does on
  some subset of tokens), `build_morphological_form()` still produces a
  genuine `MorphologicalForm(analytic_type="adjective", ...)` -- the
  fallback only triggers when the data is actually missing.
- **`adverb`** has nothing else to fall back on -- `degree` is its *only*
  property in tabulaedspy's scheme. So `ADV` tokens are never mapped to
  `analytic_type="adverb"` at all; they always become
  `analytic_type="uninflected"`, `uninflected_type="adverb"` (tabulaedspy's
  own scheme already treats non-comparable adverbs this way), regardless
  of what `token.morph` contains. This is simpler than the adjective case
  because there's no local companion type needed -- the genuine
  `uninflected` form already captures everything LatinCy gives us for an
  adverb.

## The `uninflected` catch-all

`stringmappings.qmd`'s "Uninflected type" table covers eight UPOS values
(`CCONJ` and `SCONJ` both -> `conjunction`, plus `ADP`, `ADV`, `NUM`,
`INTJ`, `PART`, `X`). Any analyzable token that isn't a
noun/pronoun/adjective/verb-family type *and* whose UPOS isn't in that
list becomes `UnclassifiedUninflected` -- a second
local companion type carrying the token's raw `upos` and `feats` rather
than guessing a specific `uninflected_type` the reference document doesn't
support, or raising an error and dropping the token. `MorphologicalFormResult.native`
is `False` for these (and for `AbbreviatedAdjective`), so a caller can
always tell a genuine tabulaedspy object from a fallback.

In practice, this catch-all currently fires for:

- **`SYM`** -- not in the table at all (only `X` is, with its own caveat
  about being a mixed bag).
- **`DET`, `PRON`, `NOUN`, `PROPN`, `ADJ`, `VERB`** *tokens whose
  morphology didn't otherwise resolve*. These are edge cases, not the
  common path (see the next section for how `NOUN`/`PRON` are actually
  dispatched). **`AUX` no longer lands here** -- see "`AUX -> finite
  verb`" below; a bare `AUX` with no recognized `VerbForm` now falls back
  to `finite verb` instead (and raises `UnmappableTokenError`, not this
  catch-all, if it's missing the properties a finite verb needs).

**`SCONJ` no longer lands here either.** It used to -- `stringmappings.qmd`'s
uninflected-type table originally listed only `CCONJ` for conjunctions,
so every Latin subordinating conjunction (*cum*, *ut*, *si*, *quod*...)
fell through to `UnclassifiedUninflected` even though they're extremely
common in ordinary prose. This page previously flagged that as a likely
oversight worth fixing in `stringmappings.qmd` -- Neel has since added
`SCONJ` alongside `CCONJ` there (both -> `conjunction`), so
`_UNINFLECTED_TYPE_BY_UPOS` now maps both, and `SCONJ` tokens get a
genuine `MorphologicalForm(analytic_type="uninflected",
uninflected_type="conjunction")` rather than the fallback. See
`tests/test_tabulaedspy_bridge.py`'s
`test_uninflected_types_from_stringmappings_table` (now covers both
`CCONJ` and `SCONJ`); the older SCONJ-as-catch-all examples in
`test_unlisted_upos_falls_back_to_unclassified_uninflected`,
`test_unclassified_uninflected_keeps_raw_feats`, and
`test_native_is_true_only_for_genuine_tabulaedspy_forms` now use `SYM`
instead, which is still genuinely unmapped.

## `AUX -> finite verb`: a new fallback, not an override

`stringmappings.qmd`'s "Analytic type" table now lists `token.pos_ is AUX
-> finite verb` directly, alongside its existing `VerbForm=Fin -> finite
verb` row. The copula ("sum") isn't only ever finite, though -- it also
has infinitive and participle forms ("esse", "futurus") that spaCy tags
`AUX` too, and the `VerbForm=Fin` row and the new `AUX` row could
therefore genuinely disagree for the same token. `tabulaedspy_bridge.py`
reads the new row as a *fallback*, not an override: `VerbForm` is still
checked first (as it already was, for the reason in the section above),
and `AUX` only takes over when no `VerbForm` value was recognized at all.
So `est` (`AUX`, `VerbForm=Fin`) and `esse` (`AUX`, `VerbForm=Inf`) both
still get their correct, distinct analytic types; only an `AUX` token
with no usable `VerbForm` at all now becomes `finite verb` instead of
falling through to `UnclassifiedUninflected`. **Flag to Neel if `AUX` was
meant to take priority over `VerbForm` instead** -- same kind of judgment
call as `PROPN`/`DET` above, made explicit in
`build_morphological_form()`'s own docstring (step 2).

## `PROPN`: confirmed; `DET`: still this module's own extension

`stringmappings.qmd`'s analytic-type table originally named only `NOUN`
(not `PROPN`) and only `PRON` (not `DET`). Leaving `PROPN`/`DET` genuinely
unhandled would route the *majority* of proper names and demonstratives in
real Latin text through the `uninflected` catch-all above, which is
clearly not right linguistically, so `tabulaedspy_bridge.py` originally
extended the table on inference for both:

- `PROPN` -> `noun` (tabulaedspy's 11-type scheme has no separate "proper
  noun" category, and traditional Latin grammar doesn't usually treat one
  as a distinct part of speech either).
- `DET` -> `pronoun` (Latin UD `DET` tokens -- demonstratives like *hic*,
  possessives like *suus* -- are pronominal adjectives in the traditional
  grammar tabulaedspy's scheme follows, and share `PRON`'s exact property
  set: `gender`, `case`, `number`).

**`PROPN` is no longer an inference.** Neel has since added it to
`stringmappings.qmd`'s "Analytic type" table directly (`NOUN` or `PROPN`
-> `noun`), so the code now follows the reference document exactly for
that case, not a judgment call layered on top of it.

**`DET` -> `pronoun` is still this module's own extension** --
`stringmappings.qmd`'s table still names only `PRON`. Called out
explicitly in `build_morphological_form()`'s own docstring. **Flag to
Neel for `stringmappings.qmd` if this isn't the intended reading**, the
same way `PROPN` was -- it was a necessary inference to make the mapping
usable on real text, not (yet) confirmed against the reference document.

## Tense: `stringmappings.qmd`'s own rules, not a symmetric table

The previous version of this document derived a `(Tense, Aspect)` lookup
table from general PROIEL-convention documentation. `stringmappings.qmd`'s
own "Tense" table is more specific, and not symmetric -- two of the six
values check only one UD feature:

| Rule (checked in this order) | tabulaedspy `tense` |
| --- | --- |
| `Tense=Pres` | `present` |
| `Tense=Past` and `Aspect=Imp` | `imperfect` |
| `Tense=Fut` and `Aspect=Imp` | `future` |
| `Tense=Past` and `Aspect=Perf` | `perfect` |
| `Tense=Pqp` | `pluperfect` |
| `Tense=Fut` and `Aspect=Perf` | `future_perfect` |

Two corrections from the old speculative table worth noting explicitly:
**present** and **pluperfect** each match on `Tense` alone -- `Aspect`
isn't checked at all, so e.g. `Tense=Pres` with an incidental
`Aspect=Perf` still yields `present`. And **future**, *for a finite verb*,
pairs `Fut` with `Aspect=Imp`, not `Aspect=Prosp` as the old table
guessed. (`Aspect=Prosp` does turn out to appear in `stringmappings.qmd`
after all, just not here -- see the participle bug fix below, where it's
what marks a *participle's* future tense.) `tabulaedspy_bridge._tense()`
implements this as the document's own conditions, checked in the
document's own order, rather than a dict keyed by the `(Tense, Aspect)`
pair. Any combination not covered by these six rules raises
`UnmappableTokenError` (e.g. `Tense=Past` with no `Aspect` at all).

**Bug fix 1 (superseded by bug fix 2 below): a participle's tense
shouldn't need `Aspect` at all.** `_tense()` above is written for finite
verbs (and infinitives), where `Aspect` really does distinguish two
parallel forms under `Tense=Past`/`Tense=Fut` (imperfect vs. perfect,
future vs. future_perfect). Originally `_build_verb_family` called the
shared `_tense()` for participles too, which meant a real participle
token would raise `UnmappableTokenError` and fail to map *at all*
whenever its `Aspect` feature was absent or didn't happen to match
`_tense()`'s finite-verb-oriented rule -- exactly backwards from
`stringmappings.qmd`'s own table, where `VerbForm=Part` unconditionally
means `participle`. First fix: a separate `_participle_tense()` keyed by
`Tense` alone, ignoring `Aspect` entirely -- present active
(`Tense=Pres`), the always-passive past participle (`Tense=Past`, always
"perfect"), future active (`Tense=Fut`).

**Bug fix 2: that first fix had it backwards -- LatinCy tags a
participle's tense through `Aspect`, not `Tense`, at all.** After digging
into how LatinCy actually tags participles, Neel updated
`stringmappings.qmd`'s "Tense" table to spell this out explicitly: for a
participle, `Aspect=Imp` (imperfective aspect) means `present`,
`Aspect=Prosp` ("prospective" aspect) means `future`, and `Aspect=Perf`
means `perfect` -- `Tense` is not consulted for a participle at all. This
is the reverse of finite verbs, where `Tense` is primary and `Aspect`
only disambiguates. (`Aspect=Prosp` -- "prospective" -- is exactly the
value an earlier draft of this document had speculatively guessed
belonged to finite-verb `future`, before Neel's own reference document
ruled that out; it turns out `Prosp` is real, just reserved for
participles instead.) `_participle_tense()` now keys on `Aspect` alone,
via `_PARTICIPLE_TENSE_BY_ASPECT` -- a participle's `Tense` feature,
whatever it is or isn't, is simply never consulted. See
`tests/test_tabulaedspy_bridge.py`'s `test_participle_tense_from_aspect_alone`,
`test_participle_tense_ignores_tense_feature`,
`test_participle_without_aspect_feature_raises`, and
`test_participle_with_unrecognized_aspect_raises`.

**Infinitives turn out to work the same way as participles.** After the
participle discovery above, Neel deciphered LatinCy's infinitive tagging
too, and added two more rows to `stringmappings.qmd`'s "Tense" table:
"for infinitives, `Aspect=Imp`" -> `present`, and "for infinitives,
`Aspect=Perf`" -> `perfect`. Same pattern as participles -- `Tense` is
never consulted for an infinitive, only `Aspect` -- so
`tabulaedspy_bridge.py` gets a second, parallel function,
`_infinitive_tense()`, keyed by `_INFINITIVE_TENSE_BY_ASPECT`. The one
difference from participles: there's no infinitive equivalent of
`Aspect=Prosp -> future`, because Latin has no synthetic future
infinitive at all -- the traditional "future infinitive" (*amaturus
esse*) is periphrastic, spelled with a separate future participle token
plus "esse", so `_INFINITIVE_TENSE_BY_ASPECT` only has two entries
(`Imp`/`Perf`), and an infinitive tagged `Aspect=Prosp` is a genuine
mapping failure, not a third recognized value. The original `test_infinitive`
(which predates this discovery) used to pass a bare `Tense=Pres` with no
`Aspect` at all; that would now fail, so it's been updated to
`Aspect=Imp` alongside the new tests --
`test_infinitive_tense_from_aspect_alone`,
`test_infinitive_tense_ignores_tense_feature`,
`test_infinitive_without_aspect_feature_raises`, and
`test_infinitive_with_unrecognized_aspect_raises` (the last of which
specifically checks that `Aspect=Prosp` -- valid for a participle, not an
infinitive -- is rejected here).

## `AUX` voice is always `active`

`stringmappings.qmd`'s "Voice" section now opens with: *"If `token.pos_`
is `AUX`, voice is `active`. Otherwise: [the `Voice=Act`/`Voice=Pass`
table]."* The copula's own tagger apparently sometimes marks an `AUX`
token `Voice=Pass` when it takes part in a passive periphrastic
construction (e.g. `est` in `amandus est`), even though the copula itself
is always grammatically active -- so an `AUX` token's own `Voice` feature,
whatever it is or isn't, must never be consulted; its voice is `active`
unconditionally. This is the same pattern as the participle/infinitive
tense discoveries above: a property that's determined a different way (or
not consulted at all) for one particular `token.pos_`/analytic-type
combination, that the rest of the mapping's generic rule would get wrong.

`tabulaedspy_bridge.py` gets a small `_voice()` helper -- `"active"` for
any `AUX` token, otherwise the ordinary `_VOICE` lookup (`Act` ->
`active`, `Pass` -> `passive`) -- used everywhere `voice` is built:
finite verb, infinitive, and participle, all three, since `AUX` can be
any of those three analytic types (see the `AUX -> finite verb` section
above) and the rule applies regardless of which one a given `AUX` token
resolves to. See `tests/test_tabulaedspy_bridge.py`'s
`test_aux_voice_is_always_active_even_when_tagged_passive`,
`test_aux_voice_is_active_without_any_voice_feature`,
`test_aux_infinitive_voice_is_always_active_even_when_tagged_passive`,
and the regression check `test_non_aux_verb_voice_still_uses_its_own_voice_feature`.

## Morphological property values

Unchanged from `stringmappings.qmd`'s own tables, direct lookups (see
`_CASE`, `_GENDER`, `_NUMBER`, `_DEGREE`, `_MOOD`, `_VOICE`, `_PERSON` in
`tabulaedspy_bridge.py`):

| Property | UD value -> tabulaedspy value |
| --- | --- |
| Case | Nom->nominative, Gen->genitive, Dat->dative, Acc->accusative, Abl->ablative, Voc->vocative |
| Gender | Masc->masculine, Fem->feminine, Neut->neuter |
| Number | Sing->singular, Plur->plural |
| Degree | Pos->positive, Cmp->comparative, Sup->superlative (only reached when `Degree` actually is present -- see above) |
| Mood | Ind->indicative, Sub->subjunctive, Imp->imperative |
| Voice | Act->active, Pass->passive (except `AUX` tokens, always `active` -- see "`AUX` voice is always `active`" above) |
| Person | "1"->first, "2"->second, "3"->third |

### Known gap: locative case

`stringmappings.qmd`'s Case table doesn't mention locative either, and
tabulaedspy's own `Case` enum has no locative value at all (six values:
nominative, genitive, dative, accusative, ablative, vocative). There is no
principled single substitution -- Latin locative forms are historically
syncretic with genitive, dative, or ablative depending on declension, so
guessing one would be inventing data. A token tagged `Case=Loc` raises
`UnmappableTokenError`. Unlike the `PROPN`/`DET` gaps above (and the
former `SCONJ` gap, now fixed), this one is a genuine schema limit
(tabulaedspy has no slot for it at all), not
something a mapping choice can paper over -- it would need a change to
tabulaedspy's own `morphology_scheme.md` to fix.

## What still needs checking against the real model

This mapping is now built from Neel's own reference document rather than
speculative model-card reading, but it still hasn't been run against real
`la_core_web_lg` output -- `huggingface.co` remains unreachable from both
the cloud sandbox this pipeline was drafted in and the network this
repository's own `.venv` runs on (see `pipeline-overview.md`). Once the
real model is available, particularly worth checking:

1. **How often the `uninflected` catch-all actually fires** on real
   text, and on which UPOS values -- now that `SCONJ` is fixed (see
   above), `SYM` frequency is the main remaining open question.
2. **Whether `Degree` genuinely never appears** on `ADJ`/`ADV` tokens, or
   only sometimes -- confirms whether `AbbreviatedAdjective` is the common
   case or a rare fallback.
3. **Whether `DET` tokens carry `Gender`/`Case`/`Number`** as assumed (so
   the `pronoun` mapping actually succeeds for them) rather than coming
   through with sparser morphology.
4. The Tense/Aspect rules above, and the locative-case gap's real-world
   frequency.

Fold confirmed findings back into `tabulaedspy_bridge.py` and this
document, and add confirmed sentences as regression fixtures in `tests/`.
