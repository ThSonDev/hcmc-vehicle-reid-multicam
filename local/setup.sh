#!/usr/bin/env bash
# Bootstrap the local pipeline environment (idempotent).
# Requires: uv (https://docs.astral.sh/uv/) and Python 3.9.x.
#
#   cd local && ./setup.sh
#
# Creates .venv and the data/ weights/ results/ logs/ temp/ directories inside local/,
# where every script anchors its paths (config.ROOT). When done, put videos in data/ and
# weights in weights/, then:
#   docker compose up -d     # Kafka
#   python run.py            # run the whole pipeline (with .venv active)
set -euo pipefail
# Anchor to this script's folder (the local/ project dir) regardless of where it is called from.
cd "$(dirname "$0")"

PYTHON_VERSION="3.9.19"

if ! command -v uv >/dev/null 2>&1; then
  echo "x 'uv' not found. Install: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

echo "==> Creating virtualenv .venv (Python ${PYTHON_VERSION})"
uv venv --python "${PYTHON_VERSION}"

echo "==> Installing dependencies (requirements.txt)"
uv pip install -r requirements.txt

# supervision & torchreid are installed separately with --no-deps so they don't break the pinned legacy stack.
echo "==> Installing supervision (--no-deps)"
uv pip install supervision==0.25.0 --no-deps

# torchreid is the vendored fork at ../osnet/deep-person-reid (shared with the Airflow stack; not moved into local/).
echo "==> Installing torchreid editable from ../osnet/deep-person-reid (--no-deps, builds Cython)"
uv pip install -e ../osnet/deep-person-reid --no-deps --no-build-isolation

echo "==> Creating runtime directories"
mkdir -p data weights results logs temp

echo ""
echo "Done. Quick check:"
.venv/bin/python -c "import torch, supervision, torchreid; \
print('  torch', torch.__version__, '| cuda', torch.cuda.is_available()); \
print('  supervision', supervision.__version__, '| torchreid', torchreid.__version__)"
echo ""
echo "Next: put videos in data/ , weights in weights/ , then 'docker compose up -d' and 'python run.py'."
