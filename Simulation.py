from __future__ import annotations

import copy
from typing import List, Union, Dict, Tuple

import numpy as np
import time

import os
from pathlib import Path
import pickle

# progressbar
import tqdm

# GUI
import tkinter as tk
from tkinter import filedialog

# data
from Data import Constants
from Data.AdvanceMethod import AdvanceMethod

# exceptions and warnings
import warnings
from Exceptions.DescriptionWarning import DescriptionWarning
from Exceptions.InvalidAdvanceMethodError import InvalidAdvanceMethodError
from Exceptions.InvalidTimeUnitError import InvalidTimeUnitError
from Exceptions.SimulationError import SimulationError
from Exceptions.SimulationNotFoundError import SimulationNotFoundError

from System import System
from Particle import Particle


class Simulation:
    """Class that handles the simulation"""

    def __init__(self, system: System, dt: float, simulated_time: float, time_unit='year', t0=0,
                 central_body_movement: bool = None, interaction_perturbation: bool = None,
                 effective_central_mass: bool = None, advance_method=None, store_data=True,
                 progressbar=True, print_info=True):
        """
        :param system: the system that will be simulated.
        :param dt: time step
        :param simulated_time: the total time to simulate. Note that the actual simulated time could differ a bit when
            it is not an integer multiple of dt
        :param time_unit: time unit. has to be one of: 'seconds'/'S', 'days'/'D', 'years'/'Y'
        :param t0: start time
        :param central_body_movement: if True, central body is assumed to move. if False central body is fixed in origin
        :param advance_method: the method used to advance the non-central particles along their Kepler orbit. Has to be
            one of AdvanceMethod.A (using rotation matrix to translate between Kepler and Cartesian coordinates),
            AdvanceMethod.B (using f and g functions) or AdvanceMethod.NONE
        :param store_data: if True, particle data (position, eccentricity etc.) for every iteration will be stored
        :param progressbar: if True, a progressbar will be shown while running simulation
        :param print_info: if True, information about the simulation will be shown for some methods
        """

        # number of iterations
        self.N = int(simulated_time / dt)

        # setting System and system attributes
        self._system = system
        self._central_body_movement = central_body_movement
        self._system.initialize_system(self.N, self._central_body_movement)

        self._interaction_perturbation = interaction_perturbation
        self._system.set_particle_perturb(self._interaction_perturbation)

        # set particle data storage arrays
        self._store_data = store_data
        self._system.initiate_particles_values_dict(self._store_data)

        # set particle H_cbm adjustments
        self._effective_central_mass = effective_central_mass
        self._system.set_particle_centre_mass(effective_central_mass)

        # set particle advance method if specified
        self._advance_method = Particle.advance_method_from_string(advance_method)
        self._system.set_particle_advance_method(self._advance_method)

        # setting time related variables
        self.TIME_CONVERSION = 1
        self.time_unit = time_unit
        self._set_time_unit_and_conversion()

        self.dt = dt * self.TIME_CONVERSION
        self.dt_in_unit = dt
        self.time_in_unit = self.dt_in_unit * self.N

        # start time and current simulation time and iteration
        self.t0_in_unit = t0
        self.t0 = t0 * self.TIME_CONVERSION
        self._time = self.t0
        self._iteration = 0

        # time arrays
        self.t_array = np.linspace(0, self.dt * self.N, self.N)
        self.t_array_in_unit = self.t_array / self.TIME_CONVERSION

        # the file path where the simulation is stored and name of simulation
        self._save_path = None
        self.name = None

        # whether to print information for some methods
        self._print_info = print_info
        self._progressbar = progressbar

        # to keep track of the run time
        self.runtime = None
        self.runtime_formatted = None

        # description of simulation at the start
        self.initial_description = self.get_description(particle_details=True)

        # function that will run every iteration. Optional parameter in Simulation.run method
        self._function = None

    @property
    def system(self) -> System:
        """The system"""
        return self._system

    @property
    def time(self) -> float:
        return self._time

    @property
    def current_time_in_unit(self) -> float:
        return self._time / self.TIME_CONVERSION

    @property
    def simulated_time(self):
        return self.dt * self.N

    @property
    def simulated_time_in_unit(self):
        return self.simulated_time / self.TIME_CONVERSION

    @property
    def central_body_movement(self) -> bool:
        return self._central_body_movement

    @property
    def advance_method(self) -> AdvanceMethod:
        return self._advance_method

    @property
    def effective_central_mass(self) -> bool:
        return self._effective_central_mass

    @property
    def fullname(self) -> str:
        return self._get_full_name()

    @property
    def details_name(self) -> str:
        return self.fullname.replace(self.name + '-', '')

    def is_terminated(self) -> bool:
        """Checks whether the simulation was terminated"""
        return self.system.has_unbounded_particles()

    def get_iterations(self, t: float = None, in_unit=True) -> int:
        """
        Get the number of iterations needed to span the given time

        :param t: the time to span
        :param in_unit: if True, it time will be in the time unit specified when creating the simulation
        :return: the number of iterations
        """

        if in_unit:
            if t is None or t > self.time_in_unit:
                return self.N
            else:
                return int(t / self.time_in_unit * self.N)
        else:
            if t is None or t > self._time:
                return self.N
            else:
                return int(t / self._time * self.N)

    def _set_time_unit_and_conversion(self):
        """Sets self.time_unit and self.TIME_CONVERSION. Called in constructor"""
        if self.time_unit.lower() in ['year', 'y']:
            self.TIME_CONVERSION = Constants.YEAR_TO_SECONDS
            self.time_unit = 'Y'
        elif self.time_unit.lower() in ['day', 'd']:
            self.TIME_CONVERSION = Constants.DAY_TO_SECONDS
            self.time_unit = 'D'
        elif self.time_unit.lower() in ['second', 's']:
            self.time_unit = 'S'
        else:
            # when an invalid time unit has been given
            raise InvalidTimeUnitError("time_unit has to be on of: 'seconds'/'S', 'days'/'D', 'years'/'Y'")

    def _run_one_iteration(self) -> bool:
        """
        Runs one iteration of the simulation

        :return: False if at least one particle ends up in unbounded orbit (eps >= 1). True otherwise
        """
        is_bounded = True

        # run a function specified by user in Simulation.run
        if self._function is not None:
            self._function()

        # advance kepler orbits over time step dt for every particle and store particle data
        self._system.advance_orbits(self._iteration, self.dt)

        if not self._system.calculate_perturbation_velocities(self.dt, self._iteration):
            # at least one particle is not in a bounded orbit anymore
            is_bounded = False

        # update current time and iteration
        self._time += self.dt
        self._iteration += 1

        return is_bounded

    def run(self, function: callable = None, progressbar: bool = None) -> str:
        """
        Runs the simulation

        :param function: a callable (function etc.) that will be called at the start of every iteration
        :param progressbar: if True, a progressbar will be shown
        """
        if progressbar is None:
            progressbar = self._progressbar

        if self._advance_method is None or self._advance_method == AdvanceMethod.NONE:
            # for when the advance method is not specified
            raise InvalidAdvanceMethodError("advance_method not specified. Cannot run Simulation.")

        if self._central_body_movement is None:
            # for when central_body_movement is not given
            raise SimulationError("central_body_movement not specified. Cannot run simulation")

        if self._central_body_movement and self._effective_central_mass is None:
            # for when effective_central_mass is not given
            raise SimulationError("effective_central_mass not specified. Cannot run simulation")

        self._function = function

        if self._print_info:
            print(
                f"{self.get_description()}"
                f"\n\nRUNNING SIMULATION..."
            )

        # to fix bug where progressbar is on top
        time.sleep(0.1)

        finished_sim_text = "SIMULATION FINISHED"

        # tqdm for progressbar
        with tqdm.tqdm(range(self.N), disable=not progressbar) as t:

            start_time = time.time()
            for _ in t:
                # run iteration until a particle is in unbounded orbit (_run_one_iteration returns False)
                if not self._run_one_iteration():
                    finished_sim_text = self._terminate()
                    break
            end_time = time.time()

        self.runtime = end_time - start_time

        if self.is_terminated():
            print(f"\n{finished_sim_text}")

        if self._print_info:
            print(f"\n\n{self.get_description(system_details=False)}")

        return finished_sim_text

    def _terminate(self) -> str:
        """
        Called when simulation is terminated preemptively

        :return: string with text about error
        """

        unbounded_eccentricities = (
            f"\n\teccentricity {p.name + ':':<15} {p.eps:4f}" for p in self._system.unbounded_particles
        )
        error_text = f"SIMULATION ENDED PREEMPTIVELY: the particles below were not in a bounded orbit anymore:"
        for eccentricity in unbounded_eccentricities:
            error_text += eccentricity

        # simulated time has changed when simulation is terminated earlier so some variables need to be changed
        self.initial_description = error_text + "\n" + self.initial_description
        self.N = self._iteration + 1

        self._system.store_particle_data(self._iteration)
        self._system.slice_values_arrays(0, self.N)

        self.time_in_unit = int(self._time / self.TIME_CONVERSION)
        self.t_array = np.linspace(0, self.dt * self.N, self.N)
        self.t_array_in_unit = self.t_array / self.TIME_CONVERSION

        return error_text

    def save(self, name=None, directory=None, override=False):
        """
        Saves the simulation as a .pickle file in '__SAVED_SIMULATIONS__' directory. If simulation was loaded and saved
        again, it will be overridden if name or directory is not changed

        WARNING: since the simulation is saved as instance of the Simulation class in a .pickle file, changing the
        Simulation class (or the other classes for that matter) could result in errors when loading older simulations.
        A better way to store the simulation is to save all the data separately, but this requires a lot more coding

        :param name: name of the simulation. Extra properties of the simulation will be added by default
        :param directory: the directory in which the simulation will be saved (child directory of
            '__SAVED_SIMULATIONS__'). Can be multiple layers deep.
        :param override: if True, it will override the already existing simulation with the same name. Otherwise it
            will add '(i)' to the name. Previously saved simulations will be overridden unless other name or directory
            is specified
        """
        if name is not None:
            name = Path(name)
            if len(name.parents) > 1:
                # in case name also contains directories
                if directory is not None:
                    directory = str(Path(directory) / name.parent)
                else:
                    directory = str(name.parent)
                name = name.name
            name = str(name)

        if self._save_path is None:
            # if sim was not loaded before
            self._save_path = self._get_save_path(name, directory)
        elif name is None and directory is None:
            # if it was loaded from saved file, it will be overridden when saved again
            override = True

        if not override:
            # check if sim with same name already exists and change name when needed
            self._save_path = Simulation._check_for_duplicate_path(self._save_path)

        with open(str(self._save_path) + '.pickle', 'wb') as file:
            pickle.dump(self, file)

        if self._print_info:
            print(f"\nSimulation saved as {self._save_path}")

    @staticmethod
    def _load_from_path(path: Path) -> Simulation:
        """
        Loads a simulation from a pickle file

        :param path: the absolute path of the file to load
        :return: loaded simulation
        """

        # loading sim
        try:
            with open(str(path), 'rb') as file:
                sim = pickle.load(file)
        except FileNotFoundError:
            raise SimulationNotFoundError(f"the simulation or directory {path} does not exist")

        sim._save_path = path.with_suffix('')
        return sim

    @staticmethod
    def load(load_last=False, directory=None) -> Union[Simulation, List[Simulation]]:
        """
        Loads a simulation (or multiple simulations) from pickle file selected by user in GUI

        :param: if True, the lasted loaded simulation will be loaded in stead of the user selecting from GUI
        :return: the loaded simulation, or a list of loaded simulations when multiple files are selected
        """
        last_loaded_sim_paths = Simulation._get_path_main_dir().parent / Constants.LAST_LOADED_SIM_FILE

        if load_last:
            try:
                with open(last_loaded_sim_paths, 'rb') as file:
                    paths = pickle.load(file)
            except FileNotFoundError:
                raise SimulationNotFoundError(f"no previous simulation to load")
        else:
            paths = Simulation._ask_for_file(multiple=True, directory=directory)

        print()  # empty line
        loaded_sims = []
        for path in paths:
            sim = Simulation._load_from_path(path)
            loaded_sims.append(sim)

            print(f"Loaded simulation {path.parent.name}/{path.name}")

        with open(last_loaded_sim_paths, 'wb') as file:
            pickle.dump(paths, file)

        return loaded_sims

    @staticmethod
    def load_from_dir(load_last=False, directory=None) -> Tuple[List[Simulation], str]:
        """
        Loads all simulation from a directory selected by user in GUI

        :return: list of loaded simulations and the directory name
        """
        last_loaded_dir_path = Simulation._get_path_main_dir().parent / Constants.LAST_LOADED_DIR_FILE

        if load_last:
            try:
                with open(last_loaded_dir_path, 'rb') as file:
                    path = pickle.load(file)
            except FileNotFoundError:
                raise SimulationNotFoundError(f"no previous directory to load")
        else:
            path = Simulation._ask_for_file(ask_for_dir=True, directory=directory)

        print()
        simulations = Simulation._load_from_dir(path)

        # save path to file
        with open(last_loaded_dir_path, 'wb') as file:
            pickle.dump(path, file)

        print(f"Loaded simulations from {path.name}")

        return simulations, path.name

    @staticmethod
    def _load_from_dir(path: Path) -> List[Simulation]:
        """
        Load all simulations files from a directory

        :param path: the path of the directory
        :return: list of the loaded simulations
        """

        simulations = []
        for file in os.listdir(path):
            if file.endswith('.pickle'):
                # only load .pickle files
                file_path = path / file
                simulations.append(Simulation._load_from_path(file_path))

        return simulations

    @staticmethod
    def load_from_dirs_in_dir(load_last=False, path: Path = None,
                              directory=None, print_info=True) -> Tuple[Dict[str, List[Simulation]], str]:
        """
        Loads all simulation from the directories that are located in a directory selected by user in GUI

        :param directory:
        :param load_last: if True, loads simulations from last selected directory
        :return: dictionary with key the name of the directory and as value a list of the simulations in that directory
        :param path:
        :param print_info:
        """
        last_loaded_main_dir_path = Simulation._get_path_main_dir().parent / Constants.LAST_LOADED_DIRS_IN_DIR_FILE

        if load_last:
            try:
                with open(last_loaded_main_dir_path, 'rb') as file:
                    path = pickle.load(file)
            except FileNotFoundError:
                raise SimulationNotFoundError(f"no previous directory to load")
        elif path is None:
            path = Simulation._ask_for_file(ask_for_dir=True, directory=directory)

        sims_in_dirs = []
        dir_list = os.listdir(path)

        for directory in dir_list:
            dir_path = path / directory
            sims = Simulation._load_from_dir(dir_path)
            sims_in_dirs.append(sims)

        if print_info:
            print(f"Loaded simulations from dirs in {path.name}")

        # save path to file
        with open(last_loaded_main_dir_path, 'wb') as file:
            pickle.dump(path, file)

        return {directory: sims for directory, sims in zip(dir_list, sims_in_dirs)}, path.name

    @staticmethod
    def load_from_multiple_dirs(load_last=False, directory=None) -> Dict[str, List[Simulation]]:
        """
        repeatedly asked the user to selected directories to load simulations from until no directory is selected (GUI
        is closed)

        :param load_last: if True, loads simulations from last selected directories
        :param directory: directory the GUI shows initially
        :return: dictionary with key the name of the directory and as value a list of the simulations in that directory
        """

        last_loaded_dirs_path = Simulation._get_path_main_dir().parent / Constants.LAST_LOADED_MULTIPLE_DIRS_FILE

        if load_last:
            try:
                with open(last_loaded_dirs_path, 'rb') as file:
                    selected_path_list = pickle.load(file)
            except FileNotFoundError:
                raise SimulationNotFoundError(f"no previous directories to load")
        else:
            selected_path_list = []
            selected_path = Simulation._ask_for_file(ask_for_dir=True, directory=directory)

            # repeatedly ask for directory until cancelled
            while len(selected_path.parents) > 0:
                selected_path_list.append(selected_path)
                selected_path = Simulation._ask_for_file(ask_for_dir=True, directory=directory)

        # loading sims from paths and putting them in list
        print()
        simulations_from_dir_list = []
        for path in selected_path_list:
            sims = Simulation._load_from_dir(path)
            simulations_from_dir_list.append(sims)
        print()

        # save path to file
        with open(last_loaded_dirs_path, 'wb') as file:
            pickle.dump(selected_path_list, file)

        return {path.name: sims for path, sims in zip(selected_path_list, simulations_from_dir_list)}

    @staticmethod
    def _ask_for_file(ask_for_dir=False, multiple=False, directory=None) -> Union[Path, List[Path]]:
        """
        Opens window to let the user select a simulation/directory from GUI

        :param ask_for_dir: if True, it will ask for a directory. If False, it will ask for a .pickle file
        :return: the Path of the selected file
        """
        root = tk.Tk()
        root.withdraw()

        path = Simulation._get_path_main_dir()
        if directory is not None:
            path = path / directory

        # Open a file dialog for the user to select a file
        if ask_for_dir:
            selected_paths = filedialog.askdirectory(title="Select directory", initialdir=str(path))
        else:
            selected_paths = filedialog.askopenfilename(title="Select file", multiple=multiple,
                                                        filetypes=[("Saved Simulations", "*.pickle")],
                                                        initialdir=str(path))
        root.destroy()
        return [Path(path) for path in selected_paths] if not ask_for_dir else Path(selected_paths)

    def _get_save_path(self, name=None, directory=None) -> Path:
        """
        Creates the path for where to save the simulation

        :param name: the name of the simulation
        :param directory: the directory in which the simulation will be saved
        :return: the save_path
        """

        name = self._get_full_name(name)
        directory = Constants.SAVED_SIM_DIR if directory is None else directory
        path = Simulation._get_path_main_dir() / directory

        # create directory if it does not exist yet
        try:
            os.makedirs(path)
        except FileExistsError:
            pass

        path = path / name
        return path

    @staticmethod
    def _get_path_main_dir() -> Path:
        """
        Gets the path of the '__SAVED_SIMULATIONS__' directory

        :return: the path
        """
        path = Path(__file__).parent

        return path / Constants.SAVED_SIM_PARENT_DIR

    @staticmethod
    def _check_for_duplicate_path(path: Path) -> Path:
        """
        Checks if simulation already exists in the path. Adds '(1)', '(2)', ... to name if it already exists

        :param path: the path with simulation name
        :return: the path with new name
        """

        list_dir = [s.replace('.pickle', '') for s in os.listdir(path.parent)]
        name = str(path.name)
        new_name = name
        i = 1
        while new_name in list_dir:
            new_name = f"{name}-({i})"
            i += 1
        return path.with_name(new_name)

    def get_name(self, name: str = None) -> str:
        """
        Gets the name of the simulation. If name is not specified it will default to '<p1.name>-<p2.name>(N)' with
        p1 and p2 the particles that are closest and farthest from the central body, respectively, and N is the total
        number of non-particles in the system. If name is given it will return the name

        :param name: an optional name
        :return: the simulation name
        """
        if name is not None:
            self.name = name

        if self.name is not None:
            return self.name

        if name is None and self._system.n_particles > 0:
            # first particle will be included in the name if at least one particle in system
            first_particle = self._system.particle_list[0].name

            last_particle = ""
            if self._system.n_particles > 1:
                # last particle will be included in the name if at least two particles in system
                last_particle = "-" + self._system.particle_list[-1].name

            self.name = f"{first_particle}{last_particle}({self._system.n_particles})"

        else:
            self.name = f"EmptySystem(0)"

        return self.name

    def _get_full_name(self, name: str = None) -> str:
        """
        Gets the full name of the simulation. If name is not specified it will default to '<p1.name>-<p2.name>(N)' with
        p1 and p2 the particles that are closest and farthest from the central body, respectively, and N is the total
        number of non-particles in the system. Will add '<time_unit><simulated_time>-dt<dt>-<central_body_motion>-
        <advance_method> to name by default.

        :param name: name of the simulation
        :return: the full name of the simulation
        """

        name = self.get_name(name)

        # add simulation parameters to name
        name += (
            f" - {self.time_unit}{self.simulated_time_in_unit:.1f}"
            f" - dt{self.dt_in_unit:.4f}"
            f" - Adv{self._advance_method.name}"
            f" - cbm{self._central_body_movement}"
        )

        try:
            name += f" - Eff{self._effective_central_mass}"
        except AttributeError:
            pass

        return name

    def get_description(self, system_details=True, particle_details=False) -> str:
        """creates a string with description of the simulation"""

        description = (
            f"Simulation details:"
            f"\n\tname:                      {self.get_name()}"
            f"\n\tsimulated time ({self.time_unit}):        {self.simulated_time_in_unit:.3f}"
            f"\n\ttime step ({self.time_unit}):             {self.dt_in_unit:.5f}"
            f"\n\tstart time: ({self.time_unit}):           {self.t0_in_unit}"
            f"\n\titerations:                {self.N}"
            f"\n\tcentral body movement:     {self._central_body_movement}"
            f"\n\tadvance method:            {self._advance_method}"
            f"\n\teffective central mass     {self._effective_central_mass}"
            f"\n\truntime:                   {self.runtime_formatted}"
        )

        if system_details:
            # adds system details to description
            description += "\n\n" + self._system.get_description(particle_details=particle_details)
        elif particle_details:
            # for if particle_details=True but system_details=False
            warnings.warn(
                "cannot add particle details to simulation description. "
                "system_details needs to be set to true",
                DescriptionWarning
            )

        return description

    def __str__(self) -> str:
        return self.get_description()

    def get_sliced_values_simulation(self, *args) -> Simulation:
        """
        Gets a copy of the simulation with the data arrays sliced to desired size

        :param args: 2 arguments, 'start' and 'end', if only 1 arguments is given its the end index. if zero arguments
            given it will return a copy of the simulation
            - if end is float <= 1, args interpreted as fraction of total length of arrays (number of iterations)
            - if end is int > 1, args interpreted as indices
        :return: the sliced simulation copy
        """
        if len(args) == 0:
            return copy.deepcopy(self)
        if len(args) == 2:
            start = args[0]
            end = args[1]
        elif len(args) == 1:
            start = 0
            end = args[0]
        else:
            raise TypeError(f"expected 1 or 2 arguments, but {len(args)} were given")

        n0, n1 = start, end
        if end <= 1:
            n0 = int(start * self.N)
            n1 = int(end * self.N)

        sim_copy = copy.deepcopy(self)
        sim_copy.N = n1 - n0

        sim_copy.system.slice_values_arrays(n0, n1)

        sim_copy.t_array = sim_copy.t_array[n0:n1]
        sim_copy.t_array_in_unit = sim_copy.t_array_in_unit[n0:n1]

        return sim_copy
