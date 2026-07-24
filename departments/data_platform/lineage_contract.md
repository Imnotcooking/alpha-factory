# Dataset Lineage Contract

Last reviewed: 2026-07-17

Every reproducible generated dataset should have a JSON manifest beside the
dataset or in its immutable artifact run directory. Research protocols may add
stronger fields, but should not omit the minimum contract below.

## Required Manifest Fields

```json
{
  "schema_version": "oqp_dataset_manifest_v1",
  "dataset_id": "stable-logical-name",
  "run_id": "content-or-config-derived-run-id",
  "created_at_utc": "2026-07-17T00:00:00Z",
  "producer": "package.module:function",
  "code_commit": "git-sha-or-dirty-marker",
  "source_provider": "provider-or-local-source",
  "source_artifacts": [
    {"path": "relative/path", "sha256": "...", "size_bytes": 0}
  ],
  "transform_config": {},
  "config_sha256": "...",
  "schema_sha256": "...",
  "date_range": {"min": "YYYY-MM-DD", "max": "YYYY-MM-DD"},
  "row_count": 0,
  "asset_count": 0,
  "output_artifacts": [
    {"path": "relative/path", "sha256": "...", "size_bytes": 0}
  ],
  "quality_summary": {},
  "parent_manifests": []
}
```

## Rules

1. Hash canonical serialized configuration, not a Python object representation.
2. Use repository-relative paths when the artifact is inside the workspace.
3. Store retrieval time and request scope for API-sourced data.
4. Include all parent manifests for multi-stage derived data.
5. Do not rewrite an authenticated manifest after downstream work references it.
6. Do not include credentials, signed URLs, or private account identifiers.
7. Mark a dirty worktree honestly when no immutable commit identifies the run.

The daily-regime research package already uses stronger authenticated
manifests. This contract is the common minimum for other data pipelines, not a
reason to weaken those controls.
