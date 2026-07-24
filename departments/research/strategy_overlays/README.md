# Private Strategy Risk Overlays

This registry owns causal exposure controls applied after a strategy target has
been constructed. Overlay files use stable `ovl_*` IDs and may scale or gate
finished positions, but they may not create alpha direction or choose between
strategy sleeves.

- Factors in `departments/research/factors/` produce alpha.
- Routers in `departments/research/routers/` allocate among named sleeves.
- Risk overlays here modify the exposure of the resulting strategy target.
- Strategy YAML files declare which factors, routers, and overlays are used.

Every overlay must declare `OVERLAY_ID`, `OVERLAY_METADATA`, and
`OVERLAY_CONTRACT`, expose `apply(targets, parameters=...)`, and declare an
`OVERLAY_PARAMETERS` schema. The schema makes defaults and any permitted search
range auditable; it does not itself authorize optimisation. The input and output
position grid must be identical. Unless the contract explicitly permits it, an
overlay cannot flip a sign or increase gross exposure.

Overlay optimisation remains blocked until an overlay-specific economic
objective profile and failure gates are approved. A declared parameter range is
an interface contract, not evidence that the overlay improves a strategy.

The Research Dashboard displays this registry inside **Router Library > Risk
Overlays** because both component types govern capital allocation. This is a UI
grouping only: `rtr_*` routers and `ovl_*` overlays retain separate registries,
contracts, loaders, and execution stages in the backend.
