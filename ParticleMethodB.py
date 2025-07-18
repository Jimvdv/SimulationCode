import scipy.optimize
import numpy as np

from Data.AdvanceMethod import AdvanceMethod
from Particle import Particle


class ParticleMethodB(Particle):

    def __init__(self, t0: float, a: float, eps: float, s: float, m: float,
                 M0: float, I: float, varpi: float, ascnode: float, centre_mass: float, name="Particle", perturb=True):
        super().__init__(t0, a, eps, s, m, M0, I, varpi, ascnode, centre_mass, name, perturb)

        # variables used for f and g functions
        self._r0_vec = 0
        self._r0 = 0
        self._v0_vec = 0

    def __init_sub(self):
        """Constructor for subclass"""
        # set variables r0_vec, r0 and v0_vec to initial values
        self.update_initial_values()

    @staticmethod
    def convert_particle(p: Particle):
        """
        Converts object type from Particle to ParticleMethodB

        :param p: the particle to convert
        """
        p.__class__ = ParticleMethodB
        p.__init_sub()

    def get_advance_method(self) -> AdvanceMethod:
        """Returns the advance method of the particle. Overrides method from parent class"""
        return AdvanceMethod.B

    def update_initial_values(self):
        """updates the reference values for the f and g functions"""
        self._r0_vec = self.r_vec
        self._v0_vec = self.v_vec
        self._r0 = self.r

    def calc_dE(self, dt):
        """
        calculates the change in eccentric anomaly after a time step

        :param dt: the time step
        :return: the value of dE
        """

        # eccentric anomaly only used for initial guess
        E0 = self.E

        C1 = np.dot(self.r_vec, self.v_vec) / (self._omega * self.a ** 2)
        C2 = (1 - self.r / self.a)
        C3 = dt * self._omega

        return scipy.optimize.fsolve(
            ParticleMethodB._root_function_dE,
            np.array([E0]),
            fprime=ParticleMethodB._root_function_dE_prime,
            args=(C1, C2, C3)
        )[0]

    @staticmethod
    def _root_function_dE(dE, C1, C2, C3):
        """root function for solving for dE to get the change in eccentric anomaly"""
        t1 = (1 - np.cos(dE)) * C1
        t2 = C2 * np.sin(dE)
        t3 = dE - C3

        return t1 - t2 + t3

    @staticmethod
    def _root_function_dE_prime(dE, C1, C2, C3):
        """Derivative of the root function for solving for dE to get the change in eccentric anomaly"""
        return 1 + (C1 * np.sin(dE)) - (C2 * np.cos(dE))

    def advance_orbit(self, dt: float):
        """
        Update the particles attributes after dt seconds have elapsed

        :param dt: time elapsed in seconds
        """
        dE = self.calc_dE(dt)

        self._move_time_step_fg_functions(dE, dt)
        self.update_initial_values()

    def _move_time_step_fg_functions(self, dE, dt):
        """
        moves the particle along its Kepler orbit for dt time

        :param dt: time step
        """
        self.r_vec = self._f(dE) * self._r0_vec + self._g(dE, dt) * self._v0_vec
        self.v_vec = self._f_derivative(dE) * self._r0_vec + self._g_derivative(dE) * self._v0_vec

    def update_velocity(self, v_vec_new: np.ndarray = None):
        """
        Updates the velocity and recalculates the particle's (Kepler) attributes
        because they will have changed

        :param v_vec_new: the new velocity
        """
        self.v_vec = v_vec_new

        # updating angles is not needed for the simulation but they are calculated anyway when particle's data is stored
        self.cartesian_to_kepler(update_angles=self._store_data)

        # update the eccentric anomaly for when it should be stored
        if self._store_data:
            self.update_eccentric_anomaly_from_r_and_eps()

        self.update_initial_values()

    def _f(self, dE) -> float:
        """
        the f function

        :return: value of the f function
        """
        return self.a / self._r0 * (np.cos(dE) - 1) + 1

    def _g(self, dE, dt) -> float:
        """
        the g function

        :param dt: time step
        :return: value of the g function
        """
        return dt + (np.sin(dE) - dE) / self._omega

    def _f_derivative(self, dE) -> float:
        """
        the derivative of the f function

        :return: value of the derivative of f
        """
        return - (self.a ** 2) * self._omega / (self.r * self._r0) * np.sin(dE)

    def _g_derivative(self, dE) -> float:
        """
        the derivative of the g function

        :return: value of the derivative of g
        """
        return self.a / self.r * (np.cos(dE) - 1) + 1
