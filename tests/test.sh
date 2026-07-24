set -e
pip install -e .
pip install "flash-linear-attention>=0.5.0" matplotlib
python tests/check_optimized_fwd.py
python tests/test_fwd.py
