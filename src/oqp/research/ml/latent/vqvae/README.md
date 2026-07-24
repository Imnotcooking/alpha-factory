# Canonical reusable VQ-VAE core

New code imports this package explicitly:

```python
from oqp.research.ml.latent.vqvae import VQVAEConfig, VQVAETrainer
```

The retired `oqp.research.latent.vqvae` namespace is no longer an import path.
Historical module-qualified type labels remain frozen inside the v1
parameter-hash wire format so existing bundle hashes stay valid; those labels
are data, not compatibility imports.

This package trains and applies a vector-quantised autoencoder to an already
prepared finite matrix. It intentionally does not own:

- feature discovery or column dropping;
- missing-value imputation or scaling;
- temporal-window construction;
- market-regime names or portfolio decisions;
- research-fold selection or dashboard output.

Those responsibilities belong to explicit research or operational adapters.
In particular, a VQ code is an anonymous discrete representation until a
separately versioned semantics policy interprets it.

The fitted model authenticates its exact feature order and neural parameters.
It retains immutable tensor snapshots rather than a public live PyTorch graph;
inference reconstructs the canonical private graph, so hooks or module swaps
cannot silently mutate the registered model.
Persistence uses a JSON manifest plus an `allow_pickle=False` NumPy archive so
artifacts do not depend on importable historical Python class paths. Loading
requires the model ID, parameter hash, and whole-bundle hash from an
independent registry; the digest stored beside a bundle is not treated as its
own source of trust. Bundle directories are immutable: publish a new versioned
directory instead of replacing an existing fitted model in place.
