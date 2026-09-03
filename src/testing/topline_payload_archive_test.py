import unittest

import numpy as np

from src.opt.topline_opt import BAD_OBJECTIVE, _update_payload_archive
from src.opt.view_results import _score_color_scale
from src.vectors import DesignVector


class PayloadArchiveTest(unittest.TestCase):
    def _vector(self, containers: int, wing_span: float) -> np.ndarray:
        vector = np.array(
            [sum(bound) / 2.0 for bound in DesignVector.bounds()],
            dtype=float,
        )
        names = DesignVector.opt_names()
        vector[names.index("extra_shipping_containers")] = containers
        vector[names.index("wing_span")] = wing_span
        return vector

    def test_keeps_only_best_design_for_each_simulator_count(self):
        archive = {}
        first_population = np.vstack(
            [
                self._vector(1, 1.0),
                self._vector(1, 1.1),
                self._vector(2, 1.2),
                self._vector(3, 1.3),
            ]
        )
        updates = _update_payload_archive(
            first_population,
            np.array([-4.0, -5.0, -3.0, BAD_OBJECTIVE]),
            generation=1,
            archive=archive,
        )

        self.assertEqual(updates, 3)
        self.assertEqual(set(archive), {1, 2})
        self.assertEqual(archive[1]["score"], 5.0)
        self.assertEqual(archive[1]["wing_span"], 1.1)

        second_population = np.vstack(
            [
                self._vector(1, 1.4),
                self._vector(2, 1.5),
            ]
        )
        updates = _update_payload_archive(
            second_population,
            np.array([-4.5, -6.0]),
            generation=2,
            archive=archive,
        )

        self.assertEqual(updates, 1)
        self.assertEqual(archive[1]["wing_span"], 1.1)
        self.assertEqual(archive[2]["score"], 6.0)
        self.assertEqual(archive[2]["wing_span"], 1.5)
        self.assertEqual(archive[2]["generation"], 2)

    def test_score_color_scale_starts_at_zero_and_greys_negatives(self):
        color_map, color_norm = _score_color_scale(np.array([-2.0, 1.5, 4.0]))

        self.assertEqual(color_norm.vmin, 0.0)
        self.assertEqual(color_norm.vmax, 4.0)
        self.assertEqual(color_map(color_norm(-1.0))[:3], (0.55, 0.55, 0.55))


if __name__ == "__main__":
    unittest.main()
