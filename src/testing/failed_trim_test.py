import importlib
import numpy as np
import pytest
from src.aero.main_aero import aero_main
from src.aero.cruise_analysis_fast import _trim_violation
from src.vectors import DesignVector, ParameterVector


def test_angle_and_force_misses_are_graded():
    assert _trim_violation(15., 20., 10., 10., 100.) == 0.
    assert _trim_violation(16., 20., 10., 10., 100.) < _trim_violation(20., 20., 10., 10., 100.)
    assert _trim_violation(0., 21., 10., 10., 100.) < _trim_violation(0., 25., 10., 10., 100.)
    assert _trim_violation(0., 0., 11., 10., 100.) == pytest.approx(.1)
    assert _trim_violation(0., 0., 9., 10., 100.) == pytest.approx(.1)


def test_closer_thrust_balance_reduces_failed_trim_penalty(monkeypatch):
    module = importlib.import_module('src.aero.cruise_analysis_fast')
    def drag(design, parameters, velocity, *args):
        force = 10. + .1*(velocity-25.)**2
        return {'test': force/(.5*parameters.rho*velocity**2*design.wing_area)}
    monkeypatch.setattr(module, 'drag_coefficients', drag)
    penalties = []
    for shortfall in [.1, 1., 5.]:
        result = aero_main(DesignVector(), ParameterVector(), (0.,0.,10.-shortfall),
                          (0.,0.,300.), 1, (.1,0.,0.), np.eye(3), 5.)
        assert not result.can_fly
        assert np.isinf(result.lap_time)
        assert 10. < result.penalty < 20.
        assert result.penalty_trim == result.penalty
        assert 'thrust-drag=' in result.flight_profile_reason
        penalties.append(result.penalty)
    assert penalties[0] < penalties[1] < penalties[2]
    cc = module.cruise_analysis_fast(DesignVector(), ParameterVector(), (0.,0.,11.), (.1,0.,0.), 5.,1)
    assert cc.converged
    assert cc.trim_violation is None
