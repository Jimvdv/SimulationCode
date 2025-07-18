from __future__ import annotations

import copy
from typing import List, Tuple

import scipy.optimize
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

# data
from Data import Constants
from Data.AdvanceMethod import AdvanceMethod

# exceptions and warnings
import warnings
from Exceptions.InvalidAdvanceMethodError import InvalidAdvanceMethodError
from Exceptions.AdvanceMethodWarning import AdvanceMethodWarning
from Exceptions.IncorrectParticleDictionaryError import IncorrectParticleDictionaryError
from Exceptions.StoreDataError import StoreDataError

from Exceptions.UnboundedOrbitError import UnboundedOrbitError


class Particle:
    """Models a particle in orbit around the sun using Kepler orbits"""

    def __init__(self, t0: float, a: float, eps: float, s: float, m: float,
                 M0: float, I: float, varpi: float, ascnode: float,
                 centre_mass: float, name="Particle", perturb=True):
        """
        :param t0: time of creation (s)
        :param a: semi-major axis (m)
        :param eps: eccentricity
        :param s: radius (m)
        :param m: mass (kg)
        :param M0: initial mean anomaly
        :param I: inclination
        :param varpi: argument of periapsis
        :param ascnode: longitude of ascending node
        :param centre_mass: mass of central body the particle orbits
        :param name: name of the particle
        :param perturb: if True, particle gravitationally perturbs other non-central particles. if False, it will not
            perturb other particles. Note that in this case it can still be perturbed by other non-central particles.
        """

        self.name = name
        self.t0 = t0
        self.a = a
        self.s = s
        self.m = m

        self._centre_mass = centre_mass
        self._effective_centre_mass = centre_mass
        self._mu = self.m / self._centre_mass

        # mean motion
        self._omega = float(self.omega)

        # position, velocity and angular momentum vectors
        self.r_vec = np.array([0, 0, 0])
        self.v_vec = np.array([0, 0, 0])
        self.L_vec = np.array([0, 0, 0])

        self.L = 0

        # to make sure eps_vec already has length eps
        self.eps_vec = eps * np.array([1, 0, 0])
        self.eps = eps

        # setting eccentric and initial mean anomaly
        self.E = 0
        self.M0 = M0
        # ... and calc E from M0
        self.update_eccentric_anomaly_from_M(self.M0, self.eps)

        # angles
        self.I = I
        self.varpi = varpi
        self.ascnode = ascnode

        self.perturb = perturb

        # calculate r_vec and v_vec, and then L_vec. Also makes sure Cartesian and Kepler coordinates agree
        self.kepler_to_cartesian()
        self.cartesian_to_kepler()

        # for saving particle data. both attributes will be
        # set to the correct value when the simulation is created
        self._N = 0
        self._store_data = False

        # dictionary in which all values of some variables (a, eps, E, M, I, varpi, ascnode, r_vec, v_vec) are stored
        # in arrays. For example the values of a at every iteration can be obtained with Particle.stored_data['a']
        self.values = {}

    @staticmethod
    def create_particle_from_dict(particle_dict: dict, advance_method: AdvanceMethod = None, rad=False) -> Particle:
        """
        Creates a particle from a dictionary with variables

        :param particle_dict: dictionary with particle attributes. See Data/ParticleDicts for template
        :param advance_method: the advance method. If not specified it will return
        :param rad: set to True if the angles given are in radians. if not, degrees are assumed.
        :return: the created particle
        """

        if 'M0' in particle_dict and 'l0' in particle_dict:
            raise IncorrectParticleDictionaryError(
                f"only one of the keys 'M0' (mean anomaly) and 'l0' (mean longitude) should be specified"
            )

        if 'longitude of periapsis' in particle_dict:
            if 'argument of periapsis' in particle_dict:
                raise IncorrectParticleDictionaryError(
                    f"either 'argument of periapsis' or 'longitude of periapsis' must be given, but not both"
                )

            particle_dict['argument of periapsis'] = \
                particle_dict['longitude of periapsis'] - particle_dict['longitude of ascending node']
            del particle_dict['longitude of periapsis']

        if 'l0' in particle_dict:
            particle_dict['M0'] = particle_dict['l0'] \
                                  - particle_dict['argument of periapsis'] \
                                  - particle_dict['longitude of ascending node']

        # adds optional attribute to particle dictionary when its not yet included and sets it to default value
        for optional_attribute, default_value in Constants.OPTIONAL_ATTRIBUTES.items():
            if optional_attribute not in particle_dict:
                particle_dict[optional_attribute] = default_value

        # not needed anymore
        del particle_dict['l0']

        dict_set = set(particle_dict)
        attribute_set = set(Constants.ATTRIBUTE_LIST)

        if dict_set != attribute_set:
            # when the particle dictionary is invalid, i.e. it does not have all the required attributes
            difference = attribute_set - dict_set
            if len(difference) > 0:
                error_text = f"particle dictionary missing the values for: {difference}"
            else:
                difference = dict_set - attribute_set
                error_text = f"particle dictionary has too many keys: {difference}"

            raise IncorrectParticleDictionaryError(error_text)

        # uses constructor to create particle
        p = Particle._create_particle_from_dict(particle_dict, rad)

        # immediately convert particle to correct advance method if specified
        if advance_method is not None:
            p.convert(advance_method)
        return p

    @staticmethod
    def _create_particle_from_dict(d: dict, rad=False) -> Particle:
        """"Used in method above to convert a particle dictionary to Particle object"""

        conversion = 1
        if not rad:
            conversion = Constants.DEGREES_TO_RADIANS

        return Particle(
            d['t0'],
            d['semimajor axis'],
            d['eccentricity'],
            d['radius'],
            d['mass'],
            d['M0'] * conversion,
            d['inclination'] * conversion,
            d['argument of periapsis'] * conversion,
            d['longitude of ascending node'] * conversion,
            d['centre mass'],
            d['name'],
            d['perturb']
        )

    def set_centre_mass(self, centre_mass, effective_central_mass: bool = None):
        self._centre_mass = centre_mass
        self._effective_centre_mass = centre_mass
        self._mu = self.m / self._centre_mass

        if effective_central_mass:
            self._effective_centre_mass = self._centre_mass / (1 + self._mu) ** 2

        self._omega = self.omega
        self.kepler_to_cartesian()
        self.cartesian_to_kepler()

    def convert(self, advance_method=None):
        """
        Converts particle to ParticleMethodA or ParticleMethodB

        :param advance_method: the advance method. Must be of type str or AdvanceMethod
        """
        # imported here to avoid error because of circular import
        from ParticleMethodA import ParticleMethodA
        from ParticleMethodB import ParticleMethodB

        if advance_method is None or advance_method == AdvanceMethod.NONE:
            # warn user if particle will not be converted
            warnings.warn(
                "advance method not specified or NONE. particle not converted.", AdvanceMethodWarning
            )
            return
        elif isinstance(advance_method, str):
            # if a string is given, convert it to Enum AdvanceMethod
            advance_method: AdvanceMethod = Particle.advance_method_from_string(advance_method)
        elif not isinstance(advance_method, AdvanceMethod):
            # if not of right type
            raise InvalidAdvanceMethodError(
                f"advance_method must be of type {str} or {AdvanceMethod}"
            )

        # converting particle to subclass if possible
        if advance_method == AdvanceMethod.A:
            ParticleMethodA.convert_particle(self)
        elif advance_method == AdvanceMethod.B:
            ParticleMethodB.convert_particle(self)
        else:
            # if none if the options above
            raise InvalidAdvanceMethodError("advance method specified is invalid")

    @property
    def r(self) -> float:
        """length of r_vec"""
        return la.norm(self.r_vec)

    @property
    def v(self) -> float:
        """length of v_vec"""
        return la.norm(self.v_vec)

    @property
    def c(self) -> float:
        """semi-focal separation"""
        return self.a * self.eps

    @property
    def b(self) -> float:
        """semi-minor axis"""
        return self.a * np.sqrt(1 - self.eps ** 2)

    @property
    def kepler_energy(self) -> float:
        """the total energy of a particle in a kepler orbit"""
        return - Constants.GRAVITATIONAL_CONSTANT * self._effective_centre_mass * self.m / (2 * self.a)

    @property
    def periapsis(self) -> float:
        """periapsis"""
        return self.a - self.c

    @property
    def apoapsis(self) -> float:
        """apoapsis"""
        return self.a + self.c

    @property
    def T(self) -> float:
        """orbital period"""
        return 2 * np.pi / self._omega

    @property
    def semi_latus_rectum(self) -> float:
        """semi-latus rectum"""
        return self.L ** 2 / (Constants.GRAVITATIONAL_CONSTANT * self._effective_centre_mass * self.m ** 2)

    @property
    def M(self):
        """the mean anomaly"""
        return Particle._root_function_kepler_equation(self.E, 0, self.eps)

    @property
    def omega(self) -> float:
        """the mean motion calculated using Kepler's third law"""
        return np.sqrt(Constants.GRAVITATIONAL_CONSTANT * self._effective_centre_mass / self.a ** 3)

    @property
    def mu(self):
        """gets the particle mass / centre mass ration"""
        return self._mu

    def get_mean_longitude(self):
        """mean longitude"""
        return (self.M + self.get_longitude_of_periapsis()) % (2 * np.pi)

    def get_longitude_of_periapsis(self):
        """longitude_of_periapsis"""
        return (self.varpi + self.ascnode) % (2 * np.pi)

    def get_nu(self) -> float:
        """Calculates the true anomaly from eccentric anomaly and returns the value"""
        top = np.cos(self.E) - self.eps
        bot = 1 - self.eps * np.cos(self.E)
        if 0 <= self.E <= np.pi:
            return np.arccos(top / bot)
        else:
            return 2 * np.pi - np.arccos(top / bot)

    def get_n_vec(self) -> np.ndarray:
        """Calculates n_vec, a vector point towards the ascending node, and returns it"""
        # https://en.wikipedia.org/wiki/Argument_of_periapsis
        # same as k x L_vec (k is unit vector in z direction)
        return np.array([-self.L_vec[1], self.L_vec[0], 0])

    def get_advance_method(self) -> AdvanceMethod:
        """Returns the advance method of the particle. Overridden subclass"""
        return AdvanceMethod.NONE

    def get_H_cbm(self, r_vec_central_body: np.ndarray, i: int = None) -> float:
        """
        Gets the value of (H_cbm)_particle
        :param r_vec_central_body: position of the central body.
        :param i: iteration of the simulation. If None, current value will be calculated

        :return: value of H_cbm
        """
        if r_vec_central_body is None:
            return 0

        p = self
        if i is not None:
            p = self.get_particle_at_iteration(i)

        dr_vec = p.r_vec - r_vec_central_body
        H_cbm = p._effective_centre_mass / p.r - p._centre_mass / la.norm(dr_vec)

        return Constants.GRAVITATIONAL_CONSTANT * p.m * H_cbm

    def get_H_int(self, particles: List[Particle], i: int = None) -> float:
        """
        Gets the value of (H_int)_particle
        :param particles: list of particles in the system
        :param i: iteration of the simulation. If None, current value will be calculated

        :return: value of (H_int)_particle
        """
        p = self
        if i is not None:
            p = self.get_particle_at_iteration(i)

        H = 0
        for p_pert in particles:
            if p_pert != self:
                r_vec_pert = p_pert.values['r'][i]
                dr_vec = p.r_vec - r_vec_pert
                H += p_pert.m / la.norm(dr_vec)

        return -Constants.GRAVITATIONAL_CONSTANT * p.m * H

    def get_H_Kep(self, i: int = None) -> float:
        """
        Gets the value of (H_Kep)_particle
        :param i: iteration of the simulation. If None, current value will be calculated

        :return: value of (H_Kep)_particle
        """
        p = self
        if i is not None:
            p = self.get_particle_at_iteration(i)

        return - Constants.GRAVITATIONAL_CONSTANT * p._effective_centre_mass * p.m / (2 * p.a)

    def get_Hs(self, particles: List[Particle], r_vec_central_body: np.ndarray, i: int = None) \
            -> Tuple[float, float, float]:
        """
        Gets the value of (H_Kep)_particle, (H_int)_particle, (H_cbm)_particle
        :param particles: list of particles in the system
        :param r_vec_central_body: position of the central body.
        :param i: iteration of the simulation. If None, current value will be calculated

        :return: values of (H_Kep)_particle, (H_int)_particle, (H_cbm)_particle
        """
        return self.get_H_Kep(i), self.get_H_int(particles, i), self.get_H_cbm(r_vec_central_body, i)

    @staticmethod
    def advance_method_from_string(advance_method: str) -> AdvanceMethod:
        """
        Gets the advance method from a string

        :param advance_method: string of the advance method
        :return: advance method as AdvanceMethod enum
        """
        if isinstance(advance_method, AdvanceMethod):
            # nothing needs to be done
            return advance_method

        #  AdvanceMethod.X.name.upper() just returns "X", the name of the Enum (always in upper case)
        if advance_method is None or advance_method.upper() in AdvanceMethod.NONE.name.upper():
            return AdvanceMethod.NONE

        elif advance_method.upper() in AdvanceMethod.A.name.upper():
            return AdvanceMethod.A

        elif advance_method.upper() in AdvanceMethod.B.name.upper():
            return AdvanceMethod.B

        else:
            raise InvalidAdvanceMethodError(
                f"advance method should be {AdvanceMethod.A.name.upper()} or {AdvanceMethod.B.name.upper()}"
            )

    def initiate_values_dict(self, N):
        """Initiates the values dictionary for storing data"""
        self._N = N
        self._store_data = True

        # scalar values
        self.values['a'] = np.zeros(self._N)
        self.values['eps'] = np.zeros(self._N)
        self.values['E'] = np.zeros(self._N)
        self.values['I'] = np.zeros(self._N)
        self.values['varpi'] = np.zeros(self._N)
        self.values['ascnode'] = np.zeros(self._N)

        # 3 x N matrix to store vectors
        self.values['r'] = np.zeros(3 * self._N).reshape(self._N, 3)
        self.values['v'] = np.zeros(3 * self._N).reshape(self._N, 3)

    def store_data_at_iteration(self, i: int):
        """
        Stores the current values of several attributes in the values dictionary

        :param i: the current iteration
        """
        if i >= len(list(self.values.values())[0]):
            # if i is too big which happens when simulation is terminated at last iteration but data is still stored
            i -= 1
            warnings.warn(
                f"simulation terminated on last iteration. Last value of data arrays is value of iteration N + 1, not N"
            )

        self.values['a'][i] = self.a
        self.values['eps'][i] = self.eps
        self.values['E'][i] = self.E
        self.values['I'][i] = self.I
        self.values['varpi'][i] = self.varpi
        self.values['ascnode'][i] = self.ascnode
        self.values['r'][i] = self.r_vec
        self.values['v'][i] = self.v_vec

    def slice_values_arrays(self, n0: int, n1: int):
        """
        Slices the data arrays

        :param n0: new start iteration
        :param n1: new end iteration
        """
        self._N = n1 - n0
        for key in self.values.keys():
            self.values[key] = self.values[key][n0:n1]

    def advance_orbit(self, dt: float):
        """
        Update the particles attributes after dt seconds have elapsed. Implemented in subclass

        :param dt: time step in seconds
        """
        pass

    def update_velocity(self, v_vec_new: np.ndarray = None):
        """
        Updates the velocity and recalculates the particle's (Kepler) attributes
        because they will have changed. Overridden in subclass

        :param v_vec_new: the new velocity
        """
        pass

    def get_v_dot_interaction(self, perturbator: Particle) -> float:
        """
        Calculates the velocity rate of change due to gravitational interaction with another body.

        :param perturbator: the other body
        :return: velocity rate of change due to gravitational interaction with other non-central bodies
        """
        dr_vec = self.r_vec - perturbator.r_vec
        dr = la.norm(dr_vec)
        return -Constants.GRAVITATIONAL_CONSTANT * perturbator.m / (dr ** 3) * dr_vec

    def get_v_dot_central_body_movement(self, r_vec_central_body):
        """
        Calculates the correction term for the case when the centre mass is assumed not to be at the origin

        :param r_vec_central_body: position vector of the central body
        :return: velocity rate of change due to central body movement
        """
        dr_vec = self.r_vec - r_vec_central_body
        temp = \
            self._effective_centre_mass * self.r_vec / (self.r ** 3) \
            - self._centre_mass * dr_vec / (la.norm(dr_vec) ** 3)

        return Constants.GRAVITATIONAL_CONSTANT * temp

    def update_eccentric_anomaly_from_r_and_eps(self):
        """Calculates and updates the eccentric anomaly from r_vec and eps_vec"""
        dot_product = np.dot(self.r_vec, self.eps_vec)
        eps = self.eps
        r = self.r

        top = eps * eps * r + dot_product
        bot = eps * r + eps * dot_product

        E = np.arccos(top / bot)

        # cross product to determine direction of the vectors in order to get the right angle
        cross_product = np.cross(self.eps_vec, self.r_vec)
        if np.dot(np.array([0, 0, 1]), cross_product) < 0:
            E = 2 * np.pi - E

        self.E = E

    def update_eccentric_anomaly_from_M(self, M, eps):
        """
        Calculates (and updates) the eccentric anomaly E  from M = E - eps*sin(E)

        :param M: mean anomaly
        :param eps: eccentricity
        """
        self.E = scipy.optimize.fsolve(
            Particle._root_function_kepler_equation,
            np.array([M]),
            fprime=Particle._root_function_kepler_equation_prime,
            args=(M, eps)
        )[0]

    @staticmethod
    def _root_function_kepler_equation(E: float, M: float, eps: float) -> float:
        """Root function for solving for E in Kepler's equation"""
        return E - eps * np.sin(E) - M

    @staticmethod
    def _root_function_kepler_equation_prime(E: float, M: float, eps: float) -> float:
        """Derivative of the root function for solving for E in Kepler's equation"""
        return 1 - eps * np.cos(E)

    def cartesian_to_kepler(self, update_angles=True):
        """"
        Calculates and updates the attributes L_vec, eps_vec, a, c, I, varpi and ascnode
        when r_vec and v_vec are known or have changed.

        :param update_angles: True if angles need to be updates, False otherwise

        """
        temp = Constants.GRAVITATIONAL_CONSTANT * self._effective_centre_mass * self.m

        # angular momentum
        self.L_vec = self.m * np.cross(self.r_vec, self.v_vec)
        self.L = la.norm(self.L_vec)

        # eccentricity
        self.eps_vec = np.cross(self.v_vec, self.L_vec) / temp - self.r_vec / self.r
        self.eps = la.norm(self.eps_vec)

        if self.eps >= 1:
            # in case the particle's orbit is unbounded after perturbation
            raise UnboundedOrbitError(f"Eccentricity of {self.name} exceeds 1. No elliptical/circular orbit", self)

        # semi-latus rectum and sami-major axis
        l = self.L ** 2 / (temp * self.m)
        self.a = l / (1 - self.eps ** 2)
        self._omega = self.omega

        if update_angles:
            # only update the angles when specified
            self.update_angles()

    def update_angles(self):
        """Calculates and updates the angles varpi, ascnode and I"""

        # inclination = angle between orbital plane and xy-plane = angle between angular momentum and z-axis
        self.I = Particle.angle_between_vectors(self.L_vec, np.array([0, 0, 1]))

        # vector in direction of ascending node
        n_vec = self.get_n_vec()

        if la.norm(n_vec) == 0:
            self.varpi = 0
            self.ascnode = 0
            # TODO: if I = 0 then n_vec = 0. so ascnode and varpi are not defined
        else:
            # argument of periapsis = angle between n_vec and eps_vec
            self.varpi = Particle.angle_between_vectors(n_vec, self.eps_vec)
            if self.eps_vec[2] < 0:
                self.varpi = 2 * np.pi - self.varpi

            # longitude of ascending node is angle between n_vec and x-axis
            self.ascnode = Particle.angle_between_vectors(n_vec, np.array([1, 0, 0]))
            if n_vec[1] < 0:
                self.ascnode = 2 * np.pi - self.ascnode

    def kepler_to_cartesian(self):
        """Calculates Cartesian coordinates from Kepler coordinates"""

        # rotation matrix
        R = self._get_rotation_matrix()

        # r vector in xy-plane
        temp_r = np.array([
            self.a * np.cos(self.E) - self.c,
            self.b * np.sin(self.E),
            0
        ])

        # r vector in 3D space
        self.r_vec = np.dot(R, temp_r)

        # v vector in xy-plane
        temp_v = np.array([
            -self.a * np.sin(self.E),
            self.b * np.cos(self.E),
            0])

        # v vector in 3D space
        self.v_vec = (self._omega * self.a / self.r) * np.dot(R, temp_v)

    def _get_rotation_matrix(self) -> np.ndarray:
        """
        Gets the rotation matrix from the angles I, varpi and ascnode

        :return: the rotation matrix
        """

        return Particle._get_rotation_matrix_zxz(self.varpi, self.I, self.ascnode)

    @staticmethod
    def _get_rotation_matrix_zxz(angle_z1: float, angle_x: float, angle_z2: float) -> np.ndarray:
        """
        Creates ta rotation matrix for the transform:
            rotation about z-axis -> rotation about x-axis -> rotation about z-axis

        :param angle_z1: angle of first rotation about z-axis (varpi)
        :param angle_x: angle of rotation about x-axis (I)
        :param angle_z2: angle of second rotation about z-axis (ascnode)
        :return: the complete rotation matrix
        """

        R_z1 = Particle._get_rotation_matrix_z(angle_z1)
        R_x = Particle._get_rotation_matrix_x(angle_x)
        R_z2 = Particle._get_rotation_matrix_z(angle_z2)

        return np.matmul(R_z2, np.matmul(R_x, R_z1))

    @staticmethod
    def _get_rotation_matrix_x(angle: float) -> np.ndarray:
        """
        Rotation matrix for rotating about x-axis

        :param angle: angle of rotation
        :return: rotation matrix about x-axis
        """
        return np.array([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)]
        ])

    @staticmethod
    def _get_rotation_matrix_z(angle: float) -> np.ndarray:
        """
        Rotation matrix for rotating about z-axis

        :param angle: angle of rotation
        :return: rotation matrix about z-axis
        """
        return np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])

    def get_particle_at_iteration(self, i, time=0.0) -> Particle:
        """
        Gets the particle as Particle instance at a specified iteration of the simulation

        :param i: the iteration
        :param time: the time
        :return: the particle at iteration i
        """

        # first calculate M from E using Kepler's equation
        E = self.values['E'][i]
        eps = self.values['eps'][i]
        M = Particle._root_function_kepler_equation(E, 0, eps)

        particle = Particle(
            time,
            self.values['a'][i],
            self.values['eps'][i],
            self.s,
            self.m,
            M,
            self.values['I'][i],
            self.values['varpi'][i],
            self.values['ascnode'][i],
            self._centre_mass,
            self.name,
            self.perturb,
        )

        try:
            use_effective_mass = self._centre_mass != self._effective_centre_mass
        except AttributeError:
            self._effective_centre_mass = self._centre_mass
            use_effective_mass = False

        particle.set_centre_mass(self._centre_mass, use_effective_mass)
        return particle

    def rescale_distance_values(self, factor):
        """
        Rescales variables with units of distance

        :param factor: the factor to rescale by
        """
        self.a *= factor
        self.r_vec *= factor

        if self._store_data:
            self.values['a'] *= factor
            self.values['r'] *= factor

    def plot_orbit(self, ax: plt.Axes, iteration=None, dot_size=None, name_label=None, **kwargs):
        """
        Plots the orbit of a particle as ellipse with the current position represented by a dot

        :param ax: matplotlib.pyplot axis of the figure
        :param iteration: the iteration of the simulation to plot. If not specified it will plot the current orbit.
        :param dot_size: Size of the dot. If None, no dot will be drawn.
        :param name_label: if True, particle names will be displayed. If name is of type dict it will be passed as
            **kwargs to the plt.text method
        :param kwargs: to specify properties like colour, label etc as in matplotlib.pyplot.plot
        """

        if iteration is None:
            self._plot_orbit(ax, dot_size=dot_size, name_kwargs=name_label, **kwargs)
        else:
            self._plot_orbit_at_iteration(ax, iteration, dot_size=dot_size, name_kwargs=name_label, **kwargs)

    def _plot_orbit(self, ax: plt.Axes, dot_size=None, dot_color=None, name_kwargs=None, **kwargs):
        """
        Plots the orbit of a particle as ellipse with the current position represented by a dot

        :param ax: matplotlib.pyplot axis of the figure
        :param dot_size: Size of the dot. If None, no dot will be drawn.
        :param kwargs: to specify properties like colour, label etc as in matplotlib.pyplot.plot
        """
        n_points = 1000
        theta = np.linspace(0, 2 * np.pi, n_points)

        # to make sure only the key 'color' is in kwargs
        if 'c' in kwargs:
            kwargs['color'] = kwargs['c']
            del kwargs['c']

        x = self.a * (np.cos(theta) - self.eps)
        y = self.b * np.sin(theta)
        z = np.zeros(n_points)

        R = Particle._get_rotation_matrix_zxz(self.varpi, self.I, self.ascnode)
        x, y, z = np.dot(R, np.array([x, y, z]))

        # ignore z to get projection onto xy-plane
        ellipse, = ax.plot(x, y, **kwargs)

        x_dot = self.a * (np.cos(self.E) - self.eps)
        y_dot = self.b * np.sin(self.E)
        x_dot, y_dot, z_dot = np.dot(R, np.array([x_dot, y_dot, 0]))

        if dot_size is not None:
            # to make dot same colour as ellipse
            kwargs_copy = copy.deepcopy(kwargs)

            if dot_color is not None:
                kwargs_copy['color'] = dot_color
            else:
                kwargs_copy['color'] = ellipse.get_color()

            ax.plot(x_dot, y_dot, '.', ms=dot_size, **kwargs_copy)

        if name_kwargs is True:
            name_kwargs = {'text': self.name}

        if name_kwargs is not None and isinstance(name_kwargs, dict):
            if 'text' not in name_kwargs.keys():
                name_kwargs['text'] = self.name

            if 'align' not in name_kwargs:
                name_kwargs['align'] = 'outer'

            if name_kwargs['align'] == 'inner':
                align_x = {1: 'right', -1: 'left'}
                align_y = {1: 'top', -1: 'bottom'}
                in_out = -1
            elif name_kwargs['align'] == 'outer':
                align_x = {1: 'left', -1: 'right'}
                align_y = {1: 'bottom', -1: 'top'}
                in_out = 1
            else:
                raise RuntimeError(f"{name_kwargs['align']} invalid value for 'align'")

            del name_kwargs['align']

            # set text and delete from name_kwarg dictionary
            s = name_kwargs['text']
            del name_kwargs['text']

            # to make sure only the key 'color' is in name_kwargs
            if 'c' in name_kwargs:
                name_kwargs['color'] = name_kwargs['c']
                del name_kwargs['c']

            if 'color' not in name_kwargs:
                name_kwargs['color'] = ellipse.get_color()

            if 'xytext' not in name_kwargs:
                if 'd' not in name_kwargs:
                    d = 7
                else:
                    d = name_kwargs['d']
                angle = self.get_mean_longitude()
                d *= in_out
                name_kwargs['xytext'] = (d * np.cos(angle), d * np.sin(angle))

            if 'd' in name_kwargs:
                del name_kwargs['d']

            if 'textcoords' not in name_kwargs:
                name_kwargs['textcoords'] = 'offset pixels'

            if 'ha' not in name_kwargs:
                name_kwargs['ha'] = align_x[np.sign(x_dot)]

            if 'va' not in name_kwargs:
                name_kwargs['va'] = align_y[np.sign(y_dot)]

            ax.annotate(s, (x_dot, y_dot),
                        bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.05'), **name_kwargs)

    def _plot_orbit_at_iteration(self, ax: plt.Axes, iteration, dot_size=None, name_kwargs=None, **kwargs):
        """
        Plots the orbit of a particle as ellipse with the current position represented by a dot at a specified iteration

        :param ax: matplotlib.pyplot axis of the figure
        :param iteration: the iteration of the simulation to plot
        :param dot_size: Size of the dot. If None, no dot will be drawn
        :param kwargs: to specify properties like colour, label etc as in matplotlib.pyplot.plot
        """
        if not self._store_data:
            raise StoreDataError("No particle data stored. Cannot plot orbit")

        particle = self.get_particle_at_iteration(iteration)
        particle._plot_orbit(ax, dot_size=dot_size, name_kwargs=name_kwargs, **kwargs)

    def plot_periapsis_evolution(self, ax: plt.Axes, npoints=1000, **kwargs):
        """
        Plots the positions of the particles periapsis at different iterations of the simulation.

        :param ax: matplotlib.pyplot axis of the figure
        :param npoints: number of points to plot
        :param kwargs: to specify properties like colour, label etc as in matplotlib.pyplot.plot
        :return:
        """

        x_list = []
        y_list = []

        iterations = np.array(np.linspace(0, self._N, npoints), dtype=int)
        iterations[-1] -= 1

        for i in iterations:
            p = self.get_particle_at_iteration(i)
            x, y, z = (p.eps_vec / p.eps) * p.periapsis
            x_list.append(x)
            y_list.append(y)

        ax.plot(x_list, y_list, **kwargs)

    @staticmethod
    def set_orbit_plot_axes_lim(ax, recentre=True, resize: float = 1) -> (float, float):
        """
        Sets the axis limits so figure will be square for plotting orbits

        :param ax: matplotlib.pyplot axis of the figure
        :param recentre: if True, it will recentre the central orbit such that it is in the middle if the figure
        :param resize: factor that the axes limits will be multiplied by to change the size
        """
        xy_lim = list(ax.get_xlim()) + list(ax.get_ylim())

        x_middle = (xy_lim[1] + xy_lim[0]) / 2
        y_middle = (xy_lim[3] + xy_lim[2]) / 2

        x_distance = xy_lim[1] - xy_lim[0]
        y_distance = xy_lim[3] - xy_lim[2]

        d = np.max([x_distance, y_distance]) / 2

        if recentre:
            x_middle, y_middle = 0, 0
            d = np.max(np.abs(xy_lim))

        d *= resize

        x_lim = (x_middle - d, x_middle + d)
        y_lim = (y_middle - d, y_middle + d)

        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        return 2 * d

    @staticmethod
    def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Calculates the (smallest) angle between two vectors

        :param v1: first vector
        :param v2: second vector
        :return: the angle between the vectors
        """
        temp = np.dot(v1, v2) / (la.norm(v1) * la.norm(v2))
        return np.arccos(temp)

    def vector_magnitudes_to_string(self) -> str:
        """Returns a string with all the vector magnitudes of the particle"""
        return (
            f"\tr:            {self.r:.3e}"
            f"\n\tv:            {self.v:.3e}"
            f"\n\tL:            {self.L:.3e}"
            f"\n\teps:          {self.eps:.5f}"
        )

    def vectors_to_string(self) -> str:
        """Returns a string with all the vectors of the particle"""
        return (
            f"\tr_vec:        {self.r_vec}"
            f"\n\tv_vec:        {self.v_vec}"
            f"\n\tL_vec:        {self.L_vec}"
            f"\n\teps_vec:      {self.eps_vec}"
        )

    def orbital_elements_to_string(self) -> str:
        """Returns a string with the orbital elements of the particle"""
        return (
            f"\tE:            {self.E * Constants.RADIANS_TO_DEGREES:.1f}"
            f"\n\tM:            {self.M * Constants.RADIANS_TO_DEGREES:.1f}"
            f"\n\ta:            {self.a:.3e}"
            f"\n\teps:          {self.eps:.3e}"
            f"\n\tI:            {self.I * Constants.RADIANS_TO_DEGREES:.1e}"
            f"\n\tascnode:      {self.ascnode * Constants.RADIANS_TO_DEGREES:.1f}"
            f"\n\tvarpi:        {self.varpi * Constants.RADIANS_TO_DEGREES:.1f}"
        )

    def other_attributes_to_string(self) -> str:
        """Returns a string of other relevant attributes not included in the methods above of the particle"""
        uses_effective_mass = (self._centre_mass != self._effective_centre_mass)

        return (
            f"\tt0:           {self.t0}"
            f"\n\tm:            {self.m:.3e}"
            f"\n\ts:            {self.s:.3e}"
            f"\n\tcentre_mass:  {self._centre_mass:.5e}"
            f"\n\teff. mass:    {uses_effective_mass}: {self._effective_centre_mass:.5e}"
            f"\n\tperturb:      {self.perturb}"
        )

    def __str__(self) -> str:
        """Creates string representation for a particle"""

        return (
            f"{self.name} (advance method: {self.get_advance_method().name})"
            f"\n{self.vector_magnitudes_to_string()}"
            f"\n\n{self.orbital_elements_to_string()}"
            f"\n\n{self.vectors_to_string()}"
            f"\n\n{self.other_attributes_to_string()}"
        )
