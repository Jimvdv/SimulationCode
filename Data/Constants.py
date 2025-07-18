from pathlib import Path as _Path
import os

import numpy as np

# Constants ---------------------------------------------------
GRAVITATIONAL_CONSTANT = 6.6743e-11  # m^3 kg^-1 s^-2
ASTRONOMICAL_UNIT = 1.49597870691e11  # m

SOLAR_MASS = 1.98911e30  # kg
EARTH_MASS = 5.9742e24  # kg

DEGREES_TO_RADIANS = 2 * np.pi / 360
RADIANS_TO_DEGREES = 1 / DEGREES_TO_RADIANS

YEAR_TO_SECONDS = 365.2425 * 24 * 60 * 60  # s
DAY_TO_SECONDS = 24 * 60 * 60  # s

SECONDS_TO_YEAR = 1 / YEAR_TO_SECONDS
SECONDS_TO_DAY = 1 / DAY_TO_SECONDS

SAVED_SIM_DIR = 'Simulations'
SAVED_SIM_PARENT_DIR = _Path(__file__).parent.parent / '__SAVED_SIMULATIONS__'

LAST_LOADED_SIM_FILE = 'Data/__loaded_sim__.pickle'
LAST_LOADED_DIR_FILE = 'Data/__loaded_dir__.pickle'
LAST_LOADED_MULTIPLE_DIRS_FILE = 'Data/__loaded_multiple_dirs__.pickle'
LAST_LOADED_DIRS_IN_DIR_FILE = 'Data/__loaded_dirs_in_dir__.pickle'

SAVED_FIGURES_DIR = _Path(__file__).parent.parent / '__SAVED_FIGURES__'

# list of attributes of a Particle
ATTRIBUTE_LIST = [
    'mass',
    'semimajor axis',
    'eccentricity',
    'inclination',
    'longitude of ascending node',
    'argument of periapsis',  # longitude of periapsis can also be used instead

    # optional attributes:
    'name',  # defaults to 'Particle'
    'centre mass',  # defaults to Solar mass
    'M0',  # defaults to 0
    't0',  # defaults to 0
    'radius',  # defaults to 0
    'perturb',  # defaults to True
]

# dictionary of optional attributes and their default values
OPTIONAL_ATTRIBUTES = {
    'name': 'Particle',
    'centre mass': SOLAR_MASS,
    'M0': 0,
    'l0': 0,
    't0': 0,
    'radius': 0,
    'perturb': True

}
