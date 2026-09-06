import numpy as np
import pytest
from src.aero.static_derivatives import calibrated_static_derivatives
from src.vectors import DesignVector

@pytest.mark.parametrize('alpha', [-2., 0., 4., 10.])
def test_cg_translation_uses_same_lift_slope(alpha):
    d = DesignVector()
    front = calibrated_static_derivatives(d, alpha, 3., .1)
    aft = calibrated_static_derivatives(d, alpha, 3., .1 + .2*d.wing_chord)
    assert aft[0] == front[0]
    assert aft[1]-front[1] == pytest.approx(.2*front[0]*np.cos(np.deg2rad(alpha)))
    assert aft[2] < front[2]

def test_elevator_changes_shared_tail_contribution():
    d = DesignVector()
    neutral = calibrated_static_derivatives(d, 4., 0., .1)
    deflected = calibrated_static_derivatives(d, 4., 8., .1)
    assert deflected[0] < neutral[0]
    assert deflected[1] > neutral[1]
    assert deflected[2] == neutral[2]

def test_fuselage_geometry_cache_respects_changed_dimensions():
    from dataclasses import replace
    d = DesignVector()
    baseline = calibrated_static_derivatives(d, 4., 0., .1)
    wider = calibrated_static_derivatives(replace(d, fuselage_width=d.fuselage_width*1.2), 4., 0., .1)
    assert wider[0] > baseline[0]
    assert wider[2] < baseline[2]
    assert calibrated_static_derivatives(d, 4., 0., .1) == baseline
