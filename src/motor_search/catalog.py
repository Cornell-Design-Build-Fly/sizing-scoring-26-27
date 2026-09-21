"""Exact catalog interpolation without computing unusable extrapolated points."""
import numpy as np

from src.prop.continuous_prop_database import ContinuousPropDatabase


class InterpolationOnlyDatabase(ContinuousPropDatabase):
    def evaluate(self, diameter_in, pitch_in, velocity_mph, rpm):
        surface = self.catalog.get_by_geometry(diameter_in, pitch_in)
        velocity, rpm = np.broadcast_arrays(velocity_mph, rpm)
        points = np.column_stack((velocity.ravel(), rpm.ravel()))
        thrust = np.asarray(surface.thrust_interpolator(points)).reshape(velocity.shape)
        torque = np.asarray(surface.torque_interpolator(points)).reshape(velocity.shape)
        if thrust.ndim == 0:
            return float(thrust), float(torque)
        return thrust, torque
