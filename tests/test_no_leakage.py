"""Engineering-rigor check: the modelling table must never contain the
`duration` leakage column. Needs db/crosssell.db (run load.py first) -
skips automatically in environments (e.g. CI) where the dataset isn't
present, rather than failing on a missing file."""
import pytest

from db import DB_PATH, get_modelling_table

pytestmark = pytest.mark.skipif(not DB_PATH.exists(), reason="run load.py first to build db/crosssell.db")


def test_no_leakage_column():
    assert "duration" not in get_modelling_table().columns


def test_target_is_binary():
    assert set(get_modelling_table()["y"].unique()) <= {0, 1}
