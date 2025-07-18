import scipy.optimize
import numpy as np

from Data.AdvanceMethod import AdvanceMethod
from Particle import Particle


class ParticleMethodA(Particle):

    def __init__(self, t0: float, a: float, eps: float, s: float, m: float,
                 M0: float, I: float, varpi: float, ascnode: float, centre_mass: float, name="Particle", perturb=True):
        super().__init__(t0, a, eps, s, m, M0, I, varpi, ascnode, centre_mass, name, perturb)

    def __init_sub(self):
        """constructor for subclass"""
        # no extra attributes need to be set
        pass

    @staticmethod
    def convert_particle(p: Particle):
        """
        Converts object type from Particle to ParticleMethodB

        :param p: the particle to convert
        """
        p.__class__ = ParticleMethodA
        p.__init_sub()

    def get_advance_method(self) -> AdvanceMethod:
        """Returns the advance method of the particle. Overrides method from parent class"""
        return AdvanceMethod.A

    def advance_orbit(self, dt: float):
        """
        Update the particles attributes after dt seconds have elapsed. Overwrites method in parent class

        :param dt: time elapsed in seconds
        """
        self.E = self.calc_E1(dt)
        self.kepler_to_cartesian()

    def calc_E1(self, dt):
        """
        calculates E1 from E0 after change over time step

        :param dt: the time step
        :return: the value of E1
        """
        # E0 also used as initial guess
        E0 = self.E

        return scipy.optimize.fsolve(
            ParticleMethodA._root_function_kepler_difference_equation,
            np.array([E0]),
            fprime=ParticleMethodA._root_function_kepler_difference_equation_prime,
            args=(E0, dt, self.eps, self._omega)
        )[0]

    @staticmethod
    def _root_function_kepler_difference_equation(E1: float, E0: float, dt: float, eps: float, omega: float):
        """Root function for solving for E1 in the difference of Kepler's equations"""
        return (E1 - E0) - eps * (np.sin(E1) - np.sin(E0)) - omega * dt

    @staticmethod
    def _root_function_kepler_difference_equation_prime(E1: float, E0: float, dt: float, eps: float, omega: float):
        """Derivative of the root function for solving for E1 in the difference of Kepler's equations"""
        return E1 - eps * np.sin(E1)

    def update_velocity(self, v_vec_new: np.ndarray = None):
        """
        Updates the velocity and recalculates the particle's (Kepler) attributes
        because they will have changed. Overwrites method in parent class

        :param v_vec_new: the new velocity
        """
        # set new velocity
        self.v_vec = v_vec_new

        # update orbital elements
        self.cartesian_to_kepler()
        self.update_eccentric_anomaly_from_r_and_eps()
