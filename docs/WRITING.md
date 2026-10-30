# Writing guide — feels docs

How the docs on this repo get written. Ari's rules; they win over everything else.

## Voice

- **Active voice everywhere.** The pipeline converts, the system scales, the algorithm queries. Never "is performed by".
- **Ari's rewrites win word-for-word.** Never "improve" her phrasing. If she wrote it, it stays.

## No negative framing

Banned: don't / doesn't / cannot / never / no / not / without / unlike, and "only"
as in "we only use X". Every sentence states what IS true. Honest limits are
phrased as measured facts ("rattle energy is reported for the observable band
only; the 5–25 Hz band requires 100 Hz on-robot sampling").

## Filler

- **No marketing taglines or slogans.** No punchy three-beat lines, no cute
  phrasing, no copy that sounds like a startup landing page. Descriptive titles
  only. The page title already does the job — no replacement tagline.
- If a section could be copy-pasted onto a different robot unchanged, delete it.
- No "why it matters" sections. No circular "because" justifications.
- Delete sections that restate a `--help` flag or what the tool already enforces.

## Terminology

- The system's own vocabulary only — never invented terms. Ari defines the
  vocabulary, not codebase grep.
- Canonical stage names: **lift → swirl → predict**.
- Banned terms: "tare" → "baseline". "confound set" → plain description.
- **No unmeasured weight claims.** Weights were never measured. Labels like
  `solid_150g` are names, not measurements. No "150g", no "same weight", no
  "same mass" as measured claims.

## Headers

Descriptive phrases that say something ("Mapping Swirl Dynamics to a 10-D
Signature"), not single-word labels. Cut headers that don't earn their place.

## Brevity

Minimum text that gets the point across. Per-paragraph audit — required?
already said? accurate (every number backed by a real run)? citable? adds
value? — else remove.

## Formatting

Bold only for the term being defined. No standout boxes.
