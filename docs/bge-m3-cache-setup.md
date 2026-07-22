# BGE-M3 Cache Setup for F1C

This guide prepares the local BGE-M3 cache required by the F1C pinned real-retriever
gate. It uses a `conda` environment named `memcontam`.

This setup is not needed for replay-only F1A/F1B work. It is required before any
F1C pass, V2 overall certification, or later scientific admission that depends on
the pinned real retriever. The verifier must load the model from cache only; it must
not download weights during verification or substitute fake embeddings.

## Required Model

```text
model: BAAI/bge-m3
revision: 5617a9f61b028005a4858fdac845db406aefb181
dimension: 1024
normalization: enabled
cache policy: local cache only
```

## Conda Environment

Run all commands from the repository root. Create the environment if it does not
exist, then activate it and install the repo.

```bash
conda create -n memcontam python=3.11 -y  # skip if it already exists
conda activate memcontam
python --version
python -c "import sys; print(sys.executable)"
python -m pip install -e '.[dev]'
```

After pulling repository changes later, rerun the final `pip install -e '.[dev]'`
command in the active environment.

## Hugging Face Cache Setup

Install or verify the Hugging Face CLI.

```bash
hf --version
```

If `hf` is unavailable, install it in the active environment.

```bash
python -m pip install -U huggingface_hub[cli]
hf --version
```

Use one cache root for both download and verification.

```bash
export HF_HOME="$HOME/.cache/huggingface"
```

Download the exact snapshot into the standard Hugging Face cache. Do not use
`--local-dir`; the verifier loads through the cache used by `sentence-transformers`.

```bash
hf download BAAI/bge-m3 \
  --revision 5617a9f61b028005a4858fdac845db406aefb181
```

Verify the cached snapshot.

```bash
hf cache verify BAAI/bge-m3 \
  --revision 5617a9f61b028005a4858fdac845db406aefb181 \
  --fail-on-missing-files
```

Confirm the exact revision can be resolved offline.

```bash
HF_HUB_OFFLINE=1 python - <<'PY'
from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id="BAAI/bge-m3",
    revision="5617a9f61b028005a4858fdac845db406aefb181",
    local_files_only=True,
)
print(path)
PY
```

The printed path should include:

```text
models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181
```

## F1C Verification

Run the project verifier. It denies network access during the run, so a pass proves
the model was available from the local cache.

```bash
python scripts/verify_bge_m3_fidelity.py
```

Expected success output includes:

```json
{"overall": "pass"}
```

Then run the F1C regression slice.

```bash
python -m pytest -q \
  tests/test_bge_m3_fidelity.py \
  tests/test_live_embedding_policy.py \
  tests/test_openai_compatible_client.py
```

## Troubleshooting

If the verifier reports `missing_cached_bge_m3`, check that download and verification
use the same cache root.

```bash
printf '%s\n' "$HF_HOME"
```

Then export the same value before both `hf download` and F1C verification.

```bash
export HF_HOME="$HOME/.cache/huggingface"
```

Do not download with `sudo`, another user account, or `--local-dir` only. In a
container, mount the host cache at the same `HF_HOME` path or set `HF_HOME` inside the
container to the mounted cache.

If F1C remains blocked, implementation and replay-only QA may still proceed, but no
overall V2 certification or scientific admission can claim a passed pinned retriever
until these commands pass.
