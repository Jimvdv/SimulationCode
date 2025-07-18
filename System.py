from __future__ import annotations

import copy
from typing import List, Dict

import numpy as np
import numpy.linalg as la

# progressbar
import tqdm
import time

# data
from Data import Constants
from Data.AdvanceMethod import AdvanceMethod

# exceptions and warnings
import warnings
from Exceptions.AdvanceMethodWarning import AdvanceMethodWarning
from Exceptions.CalculateEnergyError import CalculateEnergyError
from Exceptions.StoreDataError import StoreDataError
from Exceptions.UnboundedOrbitError import UnboundedOrbitError

from Particle import Particle


class System:
    """Models a system of particles orbiting a massive central body, like the Solar System"""

    def __init__(self, particle_list: List[Particle], centre_mass: float = None,
                 central_body_name: str = None, from_dict=False):
        """
        :param particle_list: list of the particles in the system
        :param centre_mass: the mass of the central body
        :param central_body_name: the name of the central body
        """

        # sort the particles list according to increasing semi-major axis
        self._particle_list = sorted(particle_list, key=lambda p: p.a)

        # dictionary containing all the particles. A particle can be obtained with System.particles[<particle_name>]
        self._particles = {p.name: p for p in self._particle_list}

        # attributes that will be set when Simulation is created
        self._central_body_movement = None
        self._N = 0
        self._store_data = None

        # list of masses in the system. The last bit is to reshape from (3,) to (3,1)
        self._masses = np.array([p.m for p in self._particle_list])[:, np.newaxis]

        # central body variables
        self._central_body_name = ""
        self._centre_mass = 0
        self._r_central_body_values = np.array([])
        self._v_central_body_values = np.array([])

        # to store current energy and energies at every iteration
        self.energy = 0
        self.energies = np.array([])

        # list of unbounded particles
        self.unbounded_particles = []

        if centre_mass is not None:
            self._centre_mass = centre_mass
            if central_body_name is not None:
                self._central_body_name = central_body_name
            else:
                # default name in case it is not specified
                self._central_body_name = "Central Body"
        else:
            # default to Solar mass if not specified
            self._centre_mass = Constants.SOLAR_MASS
            self._central_body_name = "Sun"

        if not from_dict:
            # if System was not created using system_from_dicts
            self.set_particle_centre_mass()

    @property
    def central_body(self):
        """Dictionary to access central body variables"""

        return {
            'name': self._central_body_name,
            'mass': self._centre_mass,
            'r': self._get_r_central_body(),
            'v': self._get_v_central_body(),
            'r values': self._r_central_body_values,
            'v values': self._v_central_body_values
        }

    @staticmethod
    def system_from_dicts(particle_dicts: List[Dict], central_body_dict: dict = None, rad=False) -> System:
        """
        Creates a system of particles from a list of particle dictionaries

        :param particle_dicts: the list of particle dictionaries
        :param central_body_dict: particle dictionary of the central body. at least 'name' and 'mass'
            should be keys in the dictionary.
        :param rad: set to True if the angles given are in radians. if not, degrees are assumed.

        :return: the created system
        """
        particle_list = []
        for dic in particle_dicts:
            if central_body_dict is not None:
                dic["centre mass"] = central_body_dict["mass"]
            particle_list.append(Particle.create_particle_from_dict(dic, rad=rad))

        if central_body_dict is not None:
            return System(
                particle_list,
                centre_mass=central_body_dict["mass"],
                central_body_name=central_body_dict["name"]
            )
        else:
            return System(particle_list, from_dict=True)

    def initialize_system(self, N: int, central_body_movement: bool):
        """
        Initializes some attributes for when Simulation is created

        :param N: number of iterations
        :param central_body_movement: True if central body is assumed to move, False if fixed at origin
        """
        self._N = N
        self._central_body_movement = central_body_movement

        if self._central_body_movement:
            self._r_central_body_values = np.zeros(3 * self._N).reshape(self._N, 3)
            self.central_body['r values'] = self._r_central_body_values

            self._v_central_body_values = np.zeros(3 * self._N).reshape(self._N, 3)
            self.central_body['v values'] = self._v_central_body_values

            # store first value
            self._r_central_body_values[0] = self._get_r_central_body()
            self._v_central_body_values[0] = self._get_v_central_body()

    @property
    def n_particles(self):
        """The number of particles in the system"""
        return len(self._particle_list)

    @property
    def particle_list(self) -> List[Particle]:
        return self._particle_list

    @property
    def particles(self) -> Dict[str, Particle]:
        return self._particles

    def has_unbounded_particles(self) -> bool:
        """Check whether there are unbounded particles in the system"""
        return len(self.unbounded_particles) > 0

    def set_particle_centre_mass(self, effective_central_mass: bool = None):
        """Sets the centre_mass attribute for each particle and updates their other variables accordingly"""

        for particle in self._particle_list:
            particle.set_centre_mass(self._centre_mass, effective_central_mass)

    def set_particle_perturb(self, interaction_perturbation: bool = None):
        """
        Sets the perturb attribute for the particles in the system

        :param interaction_perturbation: True if particles gravitationally interact with each other, False if not
        """

        if interaction_perturbation is not None:
            for p in self._particle_list:
                p.perturb = interaction_perturbation

    def initiate_particles_values_dict(self, store_data: bool = None):
        """
        Initiates the values dictionary each particle to store their data at every iteration

        :param store_data: set to True if data of all particles need to be stored, False if not.
        """
        self._store_data = store_data

        if self._store_data:
            for p in self._particle_list:
                p.initiate_values_dict(self._N)

    def set_particle_advance_method(self, advance_method=None):
        """
        Converts each particle to the desired advance method

        :param advance_method: the advance method
        """
        if advance_method is None or advance_method == AdvanceMethod.NONE:
            # warn user if particle will not be converted
            warnings.warn(
                "advance method not specified or AdvanceMethod.NONE. particles not converted.",
                AdvanceMethodWarning
            )
        else:
            for p in self._particle_list:
                p.convert(advance_method)

    def get_sliced_values_system(self, *args):
        """
        Gets a copy of the system with the data arrays sliced to desired size

        :param args: 2 arguments, 'start' and 'end', if only 1 arguments is given its the end index
            - if end is float <= 1, args interpreted as percentages of total length of arrays (number of iterations)
            - if end is int > 1, args interpreted as indices
        :return: the sliced system copy
        """
        if len(args) == 2:
            start = args[0]
            end = args[1]
        elif len(args) == 1:
            start = 0
            end = args[0]
        else:
            raise TypeError(f"expected 1 or 2 arguments, but {len(args)} were given")

        if end <= 1:
            start = int(start * self._N)
            end = int(end * self._N)

        sys_copy = copy.deepcopy(self)
        sys_copy.slice_values_arrays(start, end)
        return sys_copy

    def slice_values_arrays(self, n0: int, n1: int):
        """
        Slices the data arrays

        :param n0: new start iteration
        :param n1: new end iteration
        """
        self._N = n1 - n0

        for p in self._particle_list:
            p.slice_values_arrays(n0, n1)

        # for central body
        self._r_central_body_values = self._r_central_body_values[n0:n1]
        self.central_body['r values'] = self._r_central_body_values

        self._v_central_body_values = self._v_central_body_values[n0:n1]
        self.central_body['v values'] = self._v_central_body_values

        if len(self.energies > 0):
            self.energies = self.energies[n0:n1]

    def advance_orbits(self, iteration: int, dt: float):
        """
        Advances particle orbits and stores current data if store_data is True

        :param iteration: current iteration
        :param dt: time step
        """
        for particle in self._particle_list:
            if self._store_data:
                particle.store_data_at_iteration(iteration)
            particle.advance_orbit(dt)

    def store_particle_data(self, iteration: int):
        """
        Store the particle data for the current iteration if store_data is True. Doesn't store data for particles in
        an unbounded orbit. Note that particle data is already stored while running the simulation in the advance_orbit
        method above.

        :param iteration: current iteration
        """
        if self._store_data:
            for particle in self._particle_list:
                if particle not in self.unbounded_particles:
                    particle.store_data_at_iteration(iteration)

    def _get_r_central_body(self):
        """Returns the position vector of the central body calculated using the centre of mass"""
        positions = np.array([p.r_vec for p in self._particle_list])
        return - np.sum(self._masses * positions, axis=0) / self._centre_mass

    def _get_v_central_body(self):
        """Returns the velocity vector of the central body calculated using the centre of mass"""
        velocities = np.array([p.v_vec for p in self._particle_list])
        return - np.sum(self._masses * velocities, axis=0) / self._centre_mass

    def calculate_perturbation_velocities(self, dt: float, iteration: int):
        """
        Calculates the velocities for each particle when accounting for gravitational interaction between the particles

        :param dt: time step in seconds
        :param iteration: current iteration
        """

        # to keep track of when there is an unbounded particle
        is_bounded = True

        for p1 in self._particle_list:
            # initialise the change in velocity vector for p1
            v_vec_dot = np.zeros(3)
            for p2 in self._particle_list:
                if p2.perturb and p1 != p2:
                    # TODO: v_dot_interaction is now calculated N(N-1) times, only N(N-1)/2 times needed
                    v_vec_dot += p1.get_v_dot_interaction(p2)

            if self._central_body_movement:
                r_vec_cb = self._get_r_central_body()
                v_vec_cb = self._get_v_central_body()

                # +1 to make sure the nth entry in the central body data arrays corresponds to the same time
                # as the nth array in the particle data arrays. The data of the central body for the first iteration
                # has already been stored when initialized
                try:
                    self._r_central_body_values[iteration + 1] = r_vec_cb
                    self._v_central_body_values[iteration + 1] = v_vec_cb
                except IndexError:
                    # data arrays are full, next (and last) iteration not stored
                    pass

                v_vec_dot += p1.get_v_dot_central_body_movement(r_vec_cb)

            # calculate new velocity and update velocity for p1
            dv_vec = v_vec_dot * dt
            v_vec_new = p1.v_vec + dv_vec

            try:
                p1.update_velocity(v_vec_new)
            except UnboundedOrbitError as e:
                # if particle orbit is unbounded, save particle data (at next iteration, so +1)
                # and store particle in unbounded_particles list
                unbounded_particle: Particle = e.unbounded_particle
                unbounded_particle.store_data_at_iteration(iteration + 1)
                self.unbounded_particles.append(unbounded_particle)

                is_bounded = False

        return is_bounded

    def calculate_energies(self, progressbar=True, override=False) -> np.ndarray:
        """
        Calculates the total energy at each iteration of the simulation and stores it in energies attribute

        :return: numpy array with the total energy values at each iteration
        """
        if len(self.energies) != 0 and not override:
            raise CalculateEnergyError("energies already calculated. To calculate again set override to True")

        energies = np.zeros(self._N)

        print(f"\nCALCULATING TOTAL ENERGIES...")

        print_progressbar = tqdm.tqdm if progressbar else (lambda x: x)

        # to fix bug where progressbar is on top
        time.sleep(0.1)

        for i in print_progressbar(range(self._N)):
            energies[i] = self.calculate_energy(i)

        self.energies = energies
        return energies

    def calculate_energy_old(self, iteration=None) -> float:
        """
        Calculates the total energy of the system (= the total hamiltonian) at specified iteration at stored it in
        energy attribute

        :param iteration: the iteration. If not specified it wil calculate the energy of the current state of the system
        :return: the total energy
        """
        (
            m_particles,
            a_particles,
            r_vec_particles,
            r_vec_cb,
            v_vec_cb
        ) = self._get_values_at_iteration(iteration)

        # Kepler energy
        energy = - self._centre_mass / 2 * np.sum(m_particles / a_particles)

        # potential energy interaction non-central particles
        for i in range(self.n_particles):
            for j in range(i + 1, self.n_particles):
                mi, mj = m_particles[i], m_particles[j]
                ri, rj = r_vec_particles[i], r_vec_particles[j]
                dr = la.norm(ri - rj)

                energy += - mi * mj / dr

        # potential energy central body movement
        if self._central_body_movement:
            for m, r_vec in zip(m_particles, r_vec_particles):
                dr = la.norm(r_vec - r_vec_cb)
                r = la.norm(r_vec)
                energy += self._centre_mass * m * (1 / r - 1 / dr)

        energy = Constants.GRAVITATIONAL_CONSTANT * energy

        # kinetic energy central body
        if self._central_body_movement:
            energy += self._centre_mass * np.dot(v_vec_cb, v_vec_cb) / 2

        self.energy = energy
        return energy

    def calculate_energy(self, iteration: int = None) -> float:
        """
        Calculates the total energy of the system (= the value of total hamiltonian) at specified iteration at stored
        it in energy attribute

        :param iteration: the iteration. If not specified it wil calculate the energy of the current state of the system
        :return: the total energy
        """
        if self._central_body_movement:
            r_vec_cb = self._get_r_central_body() if iteration is None else self._r_central_body_values[iteration]
            v_vec_cb = self._get_v_central_body() if iteration is None else self._v_central_body_values[iteration]
        else:
            r_vec_cb = None
            v_vec_cb = None

        n = self.n_particles
        H_Kep_arr, H_int_arr, H_cbm_arr = np.zeros(n), np.zeros(n), np.zeros(n)
        for i, p in enumerate(self):
            H_Kep, H_int, H_cbm = p.get_Hs(self.particle_list, r_vec_cb, iteration)
            H_Kep_arr[i] = H_Kep
            H_int_arr[i] = H_int
            H_cbm_arr[i] = H_cbm

        H_Kep = np.sum(H_Kep_arr)
        H_int = np.sum(H_int_arr) / 2  # division by two because of double counting

        H_cbm = 0
        E_kin_cb = 0
        if self._central_body_movement:
            H_cbm = np.sum(H_cbm_arr)
            E_kin_cb = self._centre_mass * np.dot(v_vec_cb, v_vec_cb) / 2

        return H_Kep + H_int + H_cbm + E_kin_cb

    def get_rescaled_system(self, factor: float = None) -> System:
        """
        Creates a copy of the system with distance values (r, r_vec and a) rescaled by the given factor

        :param factor: the factor to rescale by. If not specified it if defaults to 1/AU
        :return: the rescaled copy of the System
        """
        factor = factor if factor is not None else 1 / Constants.ASTRONOMICAL_UNIT

        system_copy = copy.deepcopy(self)
        system_copy.rescale_distance_values(factor)
        return system_copy

    def rescale_distance_values(self, factor: float = None):
        """
        Rescales variables with units of distance

        :param factor: the factor to rescale by
        """
        factor = factor if factor is not None else 1 / Constants.ASTRONOMICAL_UNIT

        self._r_central_body_values *= factor

        for particle in self._particle_list:
            particle.rescale_distance_values(factor)

    def _get_values_at_iteration(self, i=None) -> tuple:
        """
        Gets the values of m, a, r_vec for all particles in the system and the values of
        r_vec_central_body and v_vec_central body at the specified iteration

        :param i: the iteration. If not specified the current values will be returned
        :return: a tuple containing the arrays m_values, a_values, r_values with the value of m, a, r_vec for each
            non-central particle and the values of r_cb, v_cb, the position and velocity of the central body
        """
        m_values = np.array([p.m for p in self._particle_list])
        r_cb = np.array([])
        v_cb = np.array([])

        if i is None:
            a_values = np.array([p.a for p in self._particle_list])
            r_values = np.array([p.r_vec for p in self._particle_list])
            if self._central_body_movement:
                r_cb = self._get_r_central_body()
                v_cb = self._get_v_central_body()
        else:
            if not self._store_data:
                raise StoreDataError(f"cannot calculate total energy at iteration {i}. Particle data not stored")
            a_values = np.array([p.values['a'][i] for p in self._particle_list])
            r_values = np.array([p.values['r'][i] for p in self._particle_list])
            if self._central_body_movement:
                r_cb = self._r_central_body_values[i]
                v_cb = self._v_central_body_values[i]
        return m_values, a_values, r_values, r_cb, v_cb

    def __str__(self) -> str:
        return self.get_description(particle_details=False)

    def __iter__(self):
        for particle in self._particle_list:
            yield particle

    def get_description(self, particle_details=True) -> str:
        """Returns a string with information about all the particles in the system"""

        description = (
            f"Central body:"
            f"\n\t- {self._central_body_name} (m = {self._centre_mass:.3e})"
            f"\n\nParticles in system ({len(self._particle_list)}):"
        )

        for particle in self._particle_list:
            description += f"\n\t- {particle.name}"

        if particle_details:
            description += "\n\n\n---------------------------- PARTICLES ----------------------------\n"
            for particle in self._particle_list:
                description += str(particle)
                description += "\n\n"

        return description
