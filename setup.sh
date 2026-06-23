#!/usr/bin/env bash
# Bootstrap the local pipeline environment (idempotent).
# Requires: uv (https://docs.astral.sh/uv/) and Python 3.9.x.
#
#   ./setup.sh
#
# When done, put videos in data/ and weights in weights/, then:
#   docker compose up -d        # Kafka
#   .venv/bin/python run.py     # run the whole pipeline
set -euo pipefail
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

echo "==> Installing torchreid editable from osnet/deep-person-reid (--no-deps, builds Cython)"
uv pip install -e osnet/deep-person-reid --no-deps --no-build-isolation

echo "==> Creating runtime directories"
mkdir -p data weights results logs temp

echo ""
echo "Done. Quick check:"
.venv/bin/python -c "import torch, supervision, torchreid; \
print('  torch', torch.__version__, '| cuda', torch.cuda.is_available()); \
print('  supervision', supervision.__version__, '| torchreid', torchreid.__version__)"
echo ""
echo "Next: put videos in data/ , weights in weights/ , then 'docker compose up -d' and '.venv/bin/python run.py'."
