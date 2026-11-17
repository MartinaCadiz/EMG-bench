"""Write a small fake dataset for smoke-testing the pipeline.

    python -m Pretrain.make_synthetic data/synthetic
"""

import sys

from .data import make_synthetic

if __name__ == "__main__":
    print(make_synthetic(sys.argv[1]))
