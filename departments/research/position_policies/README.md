# Position Policy Registry

Position policies convert an already-defined alpha state into admissible
positions or orders. They may enforce liquidity, capacity, concentration, or
cost-to-edge rules, but they may not invent return direction or choose among
strategy sleeves.

Files use stable `pos_NNN_*` IDs and declare `POSITION_POLICY_METADATA` plus
`POSITION_POLICY_CONTRACT`. Factor correlation and IC tests must use the factor
score before a position policy is applied.
