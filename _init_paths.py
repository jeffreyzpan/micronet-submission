"""Set up paths for TernaryNet."""

import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

base_dir = osp.join(osp.dirname(__file__), '..')

# Add lib to PYTHONPATH
add_path(base_dir)
