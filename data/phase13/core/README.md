# Phase 13 Core dataset inputs

`mmlu_pro_dc_selection_v1.json` records only public MMLU-Pro sample identities. It fixes the
Engineering and Physics identity sets used by Dynamic Cheatsheet release
`5cfe3c37e8e52b1d858d0f3df46e7f17c50991b9`; it does not freeze a Main trajectory order.

Materialize the pinned official datasets locally after accepting the GPQA gate:

```bash
python -m memcontam.cli phase13 materialize-core-datasets \
  --output data/phase13/core/materialized
python -m memcontam.cli phase13 validate-core-datasets \
  --root data/phase13/core/materialized \
  --trajectory-seed 1729
```

The seed above is an operator validation example, not a Main seed allocation. Bundles written to
the documented `materialized` path contain gated GPQA questions and are ignored by Git. Their
sealed manifest binds the official repository revisions, downloaded source hashes, local artifact
hashes, and row counts; validation also reports task-specific ordering hashes.

Inside this repository, the CLI rejects every output location except the ignored `materialized`
path. An output path outside the repository is allowed for ephemeral validation or protected
storage.
