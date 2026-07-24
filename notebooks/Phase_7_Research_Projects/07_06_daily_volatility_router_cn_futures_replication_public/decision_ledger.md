# Decision Ledger

## 2026-07-16 - Public manuscript boundary

- Created a separate public mirror for the daily Chinese-futures replication.
- Restricted empirical inputs to rounded, aggregate values in
  `evidence/public_evidence.json`.
- Excluded return series, event dates, product attribution, position data, execution
  schedules, source fingerprints, and any later strategy-development work.
- Locked the manuscript claim to transportability: the tested volatility-only router is
  not robust in this sample, while volatility remains informative about shock intensity.
- Kept the fixed Q4 boundary and both pre-specified market proxies; no percentile,
  product, sector, sleeve, or universe search was introduced.

## 2026-07-16 - Evidence and figures

- Generated all five manuscript figures with `scripts/build_public_figures.py`.
- Verified that the figure builder reads only the aggregate public evidence file.
- Reported the primary-proxy endpoint result together with event influence, proxy
  disagreement, conditional sleeve rankings, and cross-sectional market anatomy.
- Treated the result as exploratory because only 66 holding months are evaluable and no
  untouched confirmation period exists.

## 2026-07-16 - Manuscript release candidate

- Compiled `paper/paper_a_volatility_not_sufficient.tex` into a 14-page A4 PDF.
- Rendered and inspected every page for clipping, overlap, figure legibility, table
  layout, and pagination.
- Removed the contents pages and an unnecessary bibliography page break after visual QA.
- Required the disclosure-boundary and Module 7 architecture tests to pass before the
  manuscript is treated as a release candidate.

## Research decision

This project does not proceed to a new volatility-percentile search. The negative result
is retained as evidence that a mathematically portable scalar need not be an economically
portable decision state. Any future router must first justify pre-specified measures of
shock direction, breadth, or stage before portfolio optimization or final validation.
