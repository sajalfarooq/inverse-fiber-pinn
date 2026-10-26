import runpy
from pathlib import Path

def test_retained_release_hashes():
    namespace=runpy.run_path(str(Path(__file__).resolve().parents[1]/"verify.py"))
    assert namespace["verify"]()>0
