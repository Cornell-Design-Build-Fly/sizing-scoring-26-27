import numpy as np

from src.opt.topline_opt import _update_top_candidate_archive
from src.vectors import DesignVector


def test_top_candidate_archive_is_unique_and_bounded() -> None:
    variable_count = len(DesignVector.bounds())
    population = np.zeros((600, variable_count), dtype=float)
    population[:, 0] = np.arange(600, dtype=float)
    objectives = np.arange(600, dtype=float)
    archive: dict = {}

    _update_top_candidate_archive(
        population,
        objectives,
        generation=1,
        archive=archive,
        limit=500,
    )
    assert len(archive) == 500
    retained_objectives = sorted(row["objective"] for row in archive.values())
    assert retained_objectives == list(np.arange(500, dtype=float))

    _update_top_candidate_archive(
        population[:10],
        objectives[:10],
        generation=2,
        archive=archive,
        limit=500,
    )
    assert len(archive) == 500
