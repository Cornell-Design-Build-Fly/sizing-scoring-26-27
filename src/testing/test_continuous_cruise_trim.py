import numpy as np
import pytest

from src.aero import cruise_analysis_fast as fast
from src.aero.cruise_analysis_continuous import cruise_analysis_continuous
from src.prop.prop_classes import MPS_TO_MPH
from src.prop.prop_helper_functions import make_battery_from_design, make_motor_from_design, motor_check
from src.vectors import DesignVector, ParameterVector


class AnalyticPropDatabase:
    """Known smooth prop law to check root accuracy independently of RPM spacing."""

    def evaluate(self, *, diameter_in, pitch_in, velocity_mph, rpm):
        velocity, rpm = np.broadcast_arrays(velocity_mph, rpm)
        return 2e-7 * rpm**2 - 0.0005 * velocity**2, 1e-8 * rpm**2

    def contains(self, *, diameter_in, pitch_in, velocity_mph, rpm):
        return np.ones(np.broadcast_arrays(velocity_mph, rpm)[0].shape, dtype=bool)


@pytest.fixture
def inputs():
    return dict(
        design_vector=DesignVector(wing_span=1.8, wing_chord=0.32, motor_kv=650),
        parameter_vector=ParameterVector(), cg=(0.05, 0.0, 0.0), mass=10.0,
        mission=2, prop_database=AnalyticPropDatabase(),
    )


def test_continuous_rpm_and_throttle_balance_thrust_at_speed_cap(inputs):
    result = cruise_analysis_continuous(**inputs)
    assert result.converged
    speed = float(result.operating_point.velocity)
    assert speed == pytest.approx(fast.CRUISE_SPEED_BOUNDS_MPS[1])
    expected_rpm = np.sqrt((result.thrust_n + 0.0005 * (speed * MPS_TO_MPH)**2) / 2e-7)
    assert result.propeller_rpm == pytest.approx(expected_rpm, abs=1e-5)
    assert abs(result.propeller_rpm / 100 - round(result.propeller_rpm / 100)) > 0.01
    motor = make_motor_from_design(inputs['design_vector'], inputs['parameter_vector'])
    battery = make_battery_from_design(inputs['design_vector'], inputs['parameter_vector'])
    expected_point = motor_check(1e-8 * expected_rpm**2, expected_rpm, motor, battery)
    assert result.throttle == pytest.approx(expected_point.throttle)
    assert 0 < result.throttle < 0.55  # Missed by every former fixed retry.


@pytest.mark.parametrize('throttle_bounds', [(0.0, 0.1), (0.8, 1.0)])
def test_throttle_bounds_reject_unreachable_trim(inputs, throttle_bounds):
    assert not cruise_analysis_continuous(**inputs, throttle_bounds=throttle_bounds).converged


def test_no_extrapolation_outside_propeller_data(inputs):
    class Unsupported(AnalyticPropDatabase):
        def contains(self, **kwargs):
            return np.zeros_like(super().contains(**kwargs))
    inputs['prop_database'] = Unsupported()
    assert not cruise_analysis_continuous(**inputs).converged


def test_shared_speed_bound_is_used(inputs, monkeypatch):
    monkeypatch.setattr(fast, 'CRUISE_SPEED_BOUNDS_MPS', (3.0, 25.0))
    result = cruise_analysis_continuous(**inputs)
    assert result.converged
    assert result.operating_point.velocity == pytest.approx(25.0)
    assert fast.ALPHA_BOUNDS_DEG[0] <= result.operating_point.alpha <= fast.ALPHA_BOUNDS_DEG[1]
    assert fast.ELEVATOR_BOUNDS_DEG[0] <= result.elevator_deflection <= fast.ELEVATOR_BOUNDS_DEG[1]


def test_no_trim_above_motor_current_limit(inputs, monkeypatch):
    from src.aero import cruise_analysis_continuous as module
    motor = make_motor_from_design(inputs['design_vector'], inputs['parameter_vector'])
    from dataclasses import replace
    monkeypatch.setattr(module, 'make_motor_from_design', lambda *args: replace(motor, max_current=0.001))
    assert not cruise_analysis_continuous(**inputs).converged


def test_fixed_thrust_solver_accepts_root_at_upper_endpoint(inputs, monkeypatch):
    result = cruise_analysis_continuous(**inputs)
    # Exclude the second, slower root of the constant-thrust drag curve.
    monkeypatch.setattr(fast, 'CRUISE_SPEED_BOUNDS_MPS', (29.0, 30.0))
    fixed = fast.cruise_analysis_fast(
        inputs['design_vector'], inputs['parameter_vector'], (0.0, 0.0, result.thrust_n),
        inputs['cg'], inputs['mass'], inputs['mission'],
    )
    assert fixed.converged
    assert fixed.operating_point.velocity == pytest.approx(30.0)
    assert fixed.elevator_deflection == pytest.approx(result.elevator_deflection)
