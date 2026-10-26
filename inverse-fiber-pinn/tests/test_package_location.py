from pathlib import Path
import optics_portfolio

def test_uses_own_packaged_source():
    expected=Path(__file__).resolve().parents[1]/"src/optics_portfolio"
    assert Path(optics_portfolio.__file__).resolve().parent==expected
