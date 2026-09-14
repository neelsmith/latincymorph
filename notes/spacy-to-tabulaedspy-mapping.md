# spaCy/UD -> tabulaedspy mapping tables

Reference for `latincymorph/tabulaedspy_bridge.py`. tabulaedspy's own
target values are from `tabulaedspy/morphology_scheme.md`
(neelsmith/tabulaedspy); the UD tag/feature strings on the spaCy side are
from the `la_core_web_lg` model card
(`https://huggingface.co/latincy/la_core_web_lg`) and standard Universal
Dependencies feature documentation (universaldependencies.org). "Confirmed"
below means checked against a primary source document; it does not mean
checked against the running model (not possible in this environment --
see `pipeline-overview.md`).

## analytic_type dispatch (by UPOS, then VerbForm for verbs)

| spaCy UPOS | tabulaedspy `analytic_type` | Confidence |
| --- | --- | --- |
| `NOUN`, `PROPN` | `noun` | Confirmed (direct UD-to-scheme correspondence) |
| `PRON` | `pronoun` | Confirmed |
| `DET` | `pronoun` | **Judgment call.** tabulaedspy's 11 types have no "determiner"; Latin UD `DET` tokens (demonstratives like *hic*, possessives like *suus*) are pronominal adjectives in the traditional grammar tabulaedspy's scheme follows. Worth revisiting once real tagger output shows how often/where LatinCy actually emits `DET` vs. `PRON` vs. `ADJ` for these lemmas. |
| `ADJ` | `adjective` | Confirmed |
| `ADV` | `adverb` (if `Degree` present) else `uninflected`/`adverb` | **Heuristic**, see below |
| `VERB`, `AUX` + `VerbForm=Fin` | `finite verb` | Confirmed |
| `VERB`, `AUX` + `VerbForm=Inf` | `infinitive` | Confirmed |
| `VERB`, `AUX` + `VerbForm=Part` | `participle` | Confirmed |
| `VERB`, `AUX` + `VerbForm=Ger` | `gerund` | Confirmed |
| `VERB`, `AUX` + `VerbForm=Gdv` | `gerundive` | Confirmed |
| `VERB`, `AUX` + `VerbForm=Sup` | `supine` | Confirmed |
| `ADP` | `uninflected` / `preposition` | Confirmed |
| `CCONJ`, `SCONJ` | `uninflected` / `conjunction` | Confirmed (UD splits coordinating/subordinating; tabulaedspy's scheme has one `conjunction` bucket) |
| `PART` | `uninflected` / `particle` | Confirmed |
| `INTJ` | `uninflected` / `interjection` | Confirmed |
| `NUM` | `uninflected` / `number` | Note: this assumes LatinCy tags cardinal/ordinal numerals uninflected as `NUM` with no useful `Case`/`Gender`/`Number` features. Latin ordinals and some cardinals *do* decline like adjectives; if LatinCy's morphologizer attaches `Case`/`Gender`/`Number` to a `NUM` token, this mapping currently ignores that and files it as uninflected `number` anyway. Worth revisiting against real output. |
| `SYM`, `X` | `uninflected` / `foreign` | **Judgment call**, weakest mapping in the table -- `foreign` is the closest of the scheme's 7 `uninflected_type` values, not a documented equivalent. |
| anything else (`SPACE`, `PUNCT` reach here only if `is_analyzable()` is bypassed) | unmappable, raises `UnmappableTokenError` | -- |

### The `ADV` heuristic

morphology_scheme.md distinguishes adverbs *derived from an adjective*
(analytic_type `adverb`, carries a `degree`) from adverbs that are not
(analytic_type `uninflected`, `uninflected_type=adverb`) -- example given:
*laete*/*laetissime* (from *laetus*) are `adverb`; a non-derived adverb is
`uninflected`. UD's `FEATS` has no "derived from an adjective" flag as
such. This mapping uses **presence of a `Degree` feature on an `ADV`
token** as the proxy: if LatinCy's morphologizer attached `Degree` at all,
treat it as a comparable (adjective-derived) adverb; if not, treat it as
uninflected. This will misfire if the tagger ever attaches `Degree=Pos` to
a non-derived adverb, or omits `Degree` on a derived one it's unsure
about -- there is no way to check this without running the real model.

## Morphological property values

Straight lookup tables (see `_CASE`, `_GENDER`, etc. in
`tabulaedspy_bridge.py`); each spaCy/UD value maps to exactly one
tabulaedspy value, all confirmed directly against the model card's feature
list and `morphology_scheme.md`'s valid-values table:

| Property | UD value -> tabulaedspy value |
| --- | --- |
| Case | Nom->nominative, Gen->genitive, Dat->dative, Acc->accusative, Abl->ablative, Voc->vocative |
| Gender | Masc->masculine, Fem->feminine, Neut->neuter |
| Number | Sing->singular, Plur->plural |
| Degree | Pos->positive, Cmp->comparative, Sup->superlative |
| Mood | Ind->indicative, Sub->subjunctive, Imp->imperative |
| Voice | Act->active, Pass->passive |
| Person | "1"->first, "2"->second, "3"->third |

### Known gap: locative case

UD Latin marks `Case=Loc` for locative forms (*Romae*, *domi*, *humi*, a
handful of town/place names and a few common nouns). tabulaedspy's `Case`
enum (`morphology_scheme.md`, "Valid values for morphological properties")
has only six values -- nominative, genitive, dative, accusative, ablative,
vocative -- with **no locative**. There is no principled single
substitution (locative forms are historically syncretic with genitive,
dative, or ablative depending on declension, so guessing one would be
inventing data). `tabulaedspy_bridge.py` deliberately leaves `Loc` out of
the `_CASE` table, so a token tagged `Case=Loc` raises `UnmappableTokenError`
rather than silently picking a case. If tabulaedspy's own scheme is
extended to add a locative value, add it to `_CASE` and this becomes a
non-issue; until then, a corpus-level run should expect a handful of these
failures on any text using locatives and should not treat them as bugs.

### Tense: the `(Tense, Aspect)` pair

Latin's six tabulaedspy tenses (`present`, `imperfect`, `future`,
`perfect`, `pluperfect`, `future_perfect`) don't correspond 1:1 to a single
UD `Tense` value. The `la_core_web_lg` model card lists `Tense` values
`Pres`/`Past`/`Fut`/`Pqp` (four) and a separate `Aspect` feature with values
`Imp`/`Perf`/`Prosp`/`Inch` -- this is the PROIEL-style tense/aspect split
used for the Ancient Greek and Latin PROIEL treebanks (and carried through
the harmonized UD treebanks LatinCy trains on: Perseus, PROIEL, ITTB, LLCT,
UDante, CIRCSE, plus LASLA corpus data), documented at
`https://universaldependencies.org/la/` and `.../grc/`. The combination
table used here:

| Tense | Aspect | tabulaedspy tense |
| --- | --- | --- |
| Pres | Imp (or unset) | present |
| Past | Imp | imperfect |
| Past | Perf | perfect |
| Pqp | Perf (or unset) | pluperfect |
| Fut | Prosp (or unset) | future |
| Fut | Perf | future_perfect |

**This table is the single most speculative piece of this whole mapping.**
It is built from the general PROIEL tense/aspect convention as documented
by the UD project, cross-checked against the exact four/four feature-value
lists the model card gives (which fit this convention's shape neatly), but
it has **not** been checked against one real `la_core_web_lg` prediction,
because the model could not be installed here (see `pipeline-overview.md`).
`Aspect=Inch` (inchoative) has no entry at all -- it wasn't obviously
assignable to any of the six tabulaedspy tenses from documentation alone,
so a token with `Aspect=Inch` currently raises `UnmappableTokenError`
rather than guessing. Any `(Tense, Aspect)` pair not in the table above
also raises, by design, rather than silently defaulting -- that failure
mode is the signal to come back and extend this table once real output is
available.

## Suggested follow-up once network access allows it

1. Install `la_core_web_lg` (see `latincymorph/pipeline.py`'s docstring for
   the command) somewhere with access to `huggingface.co`.
2. Run `latincymorph.pipeline.analyze_text()` over a few dozen sentences
   spanning verse and prose, different tenses/moods, at least one locative
   and one gradable adverb.
3. For every token, print `(token.pos_, token.morph.to_dict())` and diff
   against this document's tables -- especially: does `DET` actually show
   up, and for what lemmas; does `NUM` ever carry `Case`/`Gender`/`Number`;
   does `Aspect=Inch` occur and on what tense; do `ADV` tokens without
   `Degree` include any that plausibly should have one.
4. Fold confirmed findings back into `tabulaedspy_bridge.py`'s tables and
   into this document, and add the confirmed sentences as regression
   fixtures in `tests/`.
