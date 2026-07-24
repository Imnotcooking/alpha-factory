# Module 7 Publication Review Gate

Completing an analysis does not make its paper publication-ready. Every internal report
or public manuscript must pass this gate before Module 7 starts a new research pathway.

## 1. Evidence And Claims

- Recompute or reload the frozen evidence and reconcile every headline number.
- Separate sample fact, statistical inference, economic interpretation and new hypothesis.
- Remove stale lifecycle statements and claims that exceed the evidence boundary.
- Confirm that editorial changes did not alter factor definitions, thresholds or portfolios.

## 2. Language And Terminology

- Use one primary language consistently throughout the title, body, tables and figures.
- Define unavoidable abbreviations and foreign technical terms at first use.
- Use one translation for each recurring concept; do not alternate casually between languages.
- Check headings, captions, metadata, table of contents and references separately.

## 3. Figures And Layout

- Regenerate every figure from its source script; never edit chart pixels by hand.
- Localize titles, axes, legends, annotations and units to the paper's primary language.
- Render every PDF page and inspect it visually at normal reading scale.
- Reject clipped text, unreadable chart type, orphan headings, excessive accidental whitespace,
  missing headers/footers, broken glyphs and inconsistent page furniture.

## 4. Reproducibility

- Record the data/config hashes, build command, PDF page count and final PDF SHA-256.
- Run the project tests and any report-specific localization or evidence checks.
- Require a clean TeX log apart from explicitly documented environment dependencies.

## 5. Confidentiality And Publication Boundary

- Internal reports must carry the confidentiality notice on every numbered page.
- Public papers must be rebuilt only from their allowlisted aggregate evidence contract.
- Recheck formulas, thresholds, instrument attribution, fees, capacity and directory paths
  against the public/private boundary before release.

## 6. Sign-Off

Each paper must contain a project-local `report/editorial_review.md` recording issues found,
changes made, verification evidence, residual limitations and reviewer status. Research may
resume only after the current paper queue has passed this gate or the manager explicitly
accepts a documented exception.
