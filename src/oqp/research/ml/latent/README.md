# Latent representation models

This package contains self-supervised representation learners. The reusable
VQ-VAE core accepts an already prepared finite matrix and returns discrete
codes, latent vectors, and reconstructions. Temporal-window construction and
state semantics belong to explicit consumers rather than the base model.

Use `oqp.research.ml.latent` and its purpose-named subpackages directly:
`vqvae` owns the immutable core, `encoders` owns the historical
joblib-oriented adapters, `temporal` owns reusable window transforms, and
`diagnostics` owns codebook diagnostics. The retired `oqp.research.latent`
namespace is no longer an import path.
