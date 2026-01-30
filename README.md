## Python version

Local development requires **Python 3.9.x** (tested with 3.9.19).

You can use any tool to manage Python versions (pyenv, system Python, conda, etc.).

# Install dependencies for local
```
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install supervision==0.25.0 --no-deps
pip install -e osnet/deep-person-reid --no-deps --no-build-isolation
```