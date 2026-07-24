# Supervised regression task

This package defines the target-dependent task contract, validation policy,
and experiment ledger shared by supervised regressors. It does not imply a
particular algorithm: tree ensembles live under `ml.tree_based`, while future
linear or neural regressors may implement the same contract from their own
algorithm-family packages.

Regime and representation models do not inherit this base because they do not
learn from a supervised target column.
