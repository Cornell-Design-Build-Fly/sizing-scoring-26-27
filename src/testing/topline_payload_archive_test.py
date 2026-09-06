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

    def test_keeps_only_best_design_for_each_container_count(self):
        archive = {}
        first_population = np.vstack(
            [
                self._vector(3, 1.0),
                self._vector(3, 1.1),
                self._vector(6, 1.2),
                self._vector(9, 1.3),
            ]
        )
        updates = _update_payload_archive(
            first_population,
            np.array([-4.0, -5.0, -3.0, BAD_OBJECTIVE]),
            generation=1,
            archive=archive,
        )

        self.assertEqual(updates, 3)
        self.assertEqual(set(archive), {3, 6})
        self.assertEqual(archive[3]["score"], 5.0)
        self.assertEqual(archive[3]["wing_span"], 1.1)

        second_population = np.vstack(
            [
                self._vector(3, 1.4),
                self._vector(6, 1.5),
            ]
        )
        updates = _update_payload_archive(
            second_population,
            np.array([-4.5, -6.0]),
            generation=2,
            archive=archive,
        )

        self.assertEqual(updates, 1)
        self.assertEqual(archive[3]["wing_span"], 1.1)
        self.assertEqual(archive[6]["score"], 6.0)
        self.assertEqual(archive[6]["wing_span"], 1.5)
        self.assertEqual(archive[6]["generation"], 2)

    def test_optimizer_uses_current_payload_and_propeller_fields(self):
        from src.opt.topline_opt import (
            ToplineConfig, _differential_evolution_kwargs, _pd_ratio,
        )
        names = DesignVector.opt_names()
        vector = self._vector(0, 1.2)
        vector[names.index("prop_diameter_in")] = 20.
        vector[names.index("prop_pitch_in")] = 10.
        vector[names.index("motor_kv")] = 400.
        self.assertEqual(_pd_ratio(vector), 0.5)
        options = _differential_evolution_kwargs(ToplineConfig())
        self.assertEqual(len(options["constraints"]), 1)
        self.assertEqual(
            [name for name, integer in zip(names, options["integrality"]) if integer],
            ["extra_shipping_containers"],
        )

    def test_score_color_scale_starts_at_zero_and_greys_negatives(self):
        color_map, color_norm = _score_color_scale(np.array([-2.0, 1.5, 4.0]))

        self.assertEqual(color_norm.vmin, 0.0)
        self.assertEqual(color_norm.vmax, 4.0)
        self.assertEqual(color_map(color_norm(-1.0))[:3], (0.55, 0.55, 0.55))


if __name__ == "__main__":
    unittest.main()
