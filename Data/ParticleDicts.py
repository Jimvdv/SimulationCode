from Data.Constants import *

# -------------------------------------------------------------
# This file contains dictionaries with orbital elements
# and other relevant data of celestial bodies in the
# Solar System. All values are in SI units.
# -------------------------------------------------------------

# template for dictionary with default values for optional arguments
_DICT = {
    'name':                             'Particle',         # optional, defaults to 'particle'
    'mass':                             1e24,
    'semimajor axis':                   1e9,
    'eccentricity':                     1,
    'inclination':                      1,
    'longitude of ascending node':      1,
    'longitude of periapsis':           1,

    # other optional arguments with default values:
    'centre mass':                      SOLAR_MASS,
    'M0':                               0,
    'l0':                               0,
    't0':                               0,
    'radius':                           0,
    'perturb':                          True,
}


SUN = {
    'name':                             'Sun',
    'mass':                             SOLAR_MASS,
    'radius':                           696340e3,
}

TEST_PARTICLE = {
    'name':                             'Test Particle',
    'mass':                             1e24,
    'radius':                           0,
    'semimajor axis':                   3 * ASTRONOMICAL_UNIT,
    'eccentricity':                     0.1,
    'inclination':                      0,
    'longitude of periapsis':            0,
    'longitude of ascending node':       0,
    'centre mass':                      SOLAR_MASS,
    'perturb':                          False
}

# data from https://nssdc.gsfc.nasa.gov/planetary/factsheet/plutofact.html

MERCURY = {
    'name':                             'Mercury',
    'mass':                             0.33010e24,
    'radius':                           2439.7e3,
    'semimajor axis':                   57.909e9,
    'eccentricity':                     0.2056,
    'inclination':                      7.00487,
    'longitude of ascending node':      48.33167,
    'longitude of periapsis':           77.45645,
    # mean anomaly at J2000. mean anomaly = mean longitude - argument of periapsis - longitude of ascending node
    'l0':                               252.25084
}

VENUS = {
    'name':                             'Venus',
    'mass':                             4.8673e24,
    'semimajor axis':                   108.210e9,
    'eccentricity':                     0.00677323,
    'inclination':                      3.39471,
    'longitude of ascending node':      76.68069,
    'longitude of periapsis':           131.53298,
    'l0':                               181.97973
}

EARTH = {
    'name':                             'Earth',
    'mass':                             5.9722e24,
    'semimajor axis':                   149.598e9,
    'eccentricity':                     0.01671022,
    'inclination':                      0.00005,
    'longitude of ascending node':     -11.26064,
    'longitude of periapsis':           102.94719,
    'l0':                               100.46435
}

MARS = {
    'name':                             'Mars',
    'mass':                             0.64169e24,
    'semimajor axis':                   227.956e9,
    'eccentricity':                     0.09341233,
    'inclination':                      1.85061,
    'longitude of ascending node':      49.57854,
    'longitude of periapsis':           336.04084,
    'l0':                               355.45332
}

JUPITER = {
    'name':                             'Jupiter',
    'mass':                             1.89813e27,
    'radius':                           69911e3,
    'semimajor axis':                   778.479e9,
    'eccentricity':                     0.0487,
    'inclination':                      1.304,
    'longitude of ascending node':      100.55615,
    'longitude of periapsis':           14.75385,
    'l0':                               34.40438
}
# Data from https://nssdc.gsfc.nasa.gov/planetary/factsheet/

SATURN = {
    'name':                             'Saturn',
    'mass':                             568.32e24,
    'radius':                           58232e3,
    'semimajor axis':                   1432.041e9,
    'eccentricity':                     0.0520,
    'inclination':                      2.48446,
    'longitude of ascending node':      113.71504,
    'longitude of periapsis':           92.43194,
    'l0':                               49.94432
}
# data from https://nssdc.gsfc.nasa.gov/planetary/factsheet/saturnfact.html

URANUS = {
    'name':                             'Uranus',
    'mass':                             86.811e24,
    'semimajor axis':                   2867.043e9,
    'eccentricity':                     0.04716771,
    'inclination':                      0.76986,
    'longitude of ascending node':      74.22988,
    'longitude of periapsis':           170.96424,
    'l0':                               313.23218
}

NEPTUNE = {
    'name':                             'Neptune',
    'mass':                             102.409e24,
    'radius':                           24622e3,
    'semimajor axis':                   4514.953e9,
    'eccentricity':                     0.0097,
    'inclination':                      1.76917,
    'longitude of ascending node':      131.72169,
    'longitude of periapsis':           44.97135,
    'l0':                               304.88003
}


PLUTO = {
    'name':                             'Pluto',
    'mass':                             0.01303e24,
    'radius':                           1188e3,
    'semimajor axis':                   5869.656e9,
    'eccentricity':                     0.2444,
    'inclination':                      17.16,
    'longitude of ascending node':      110.30347,
    'longitude of periapsis':           224.06676,
    'l0':                               238.92881
}


CERES = {
    'name': 'Ceres',
    'mass': 0.000159 * EARTH_MASS,
    'radius': 473e3,
    'semimajor axis': 2.7658 * ASTRONOMICAL_UNIT,
    'eccentricity': 0.078,
    'inclination': 0.1,
    'longitude of ascending node': 100.55615,
    'longitude of periapsis': 73.1
}
# Data from https://www.princeton.edu/~willman/planetary_systems/Sol/Ceres/


ENCELADUS = {
    'name': 'Enceladus',
    'mass': 1.08e20,
    'radius': 250e3,
    'semimajor axis': 238.02e6,
    'eccentricity': 0.0045,
    'inclination': 0.00,
    'longitude of ascending node': 0,
    'longitude of periapsis': 100,
    'centre mass': SATURN['mass']
}
# data from https://nssdc.gsfc.nasa.gov/planetary/factsheet/saturniansatfact.html
# argument of periapsis and ascending node are not known

DIONE = {
    'name': 'Dione',
    'mass': 11e20,
    'radius': 560e3,
    'semimajor axis': 377.40e6,
    'eccentricity': 0.0022,
    'inclination': 0.02,
    'longitude of ascending node': 0,
    'longitude of periapsis': 100,
    'centre mass': SATURN['mass']
}
# data from https://nssdc.gsfc.nasa.gov/planetary/factsheet/saturniansatfact.html

# Dione en Enceladus forced into resonance by choosing the semimajor axis
# such that exactly omega_enceladus = 2 * omega_dione
DIONE_RESONANCE = {
    'name': 'Dione',
    'mass': 11e20,
    'radius': 560e3,
    'semimajor axis': ENCELADUS['semimajor axis'] * 2 ** (2 / 3),
    'eccentricity': 0.0022,
    'inclination': 0.02,
    'longitude of periapsis': 100,
    'longitude of ascending node': 0,
    'centre mass': SATURN['mass']
}

MIMAS = {
    'name': 'Mimas',
    'mass': 0.379e20,
    'radius': 200e3,
    'semimajor axis': 185.52e6,
    'eccentricity': 0.0202,
    'inclination': 1.53,
    'longitude of periapsis': 0,
    'longitude of ascending node': 0,
    'centre mass': SATURN['mass']
}

IO = {
    'name': 'Io',
    'mass': 893.2e20,
    'radius': 1821.5e3,
    'semimajor axis': 421.8e6,
    'eccentricity': 0.004,
    'inclination': 0.04,
    'longitude of periapsis': 0,
    'longitude of ascending node': 0,
    'centre mass': JUPITER['mass']
}

EUROPA = {
    'name': 'Europa',
    'mass': 480.0e20,
    'radius': 1560.8e3,
    'semimajor axis': 671.1e6,
    'eccentricity': 0.009,
    'inclination': 0.47,
    'longitude of periapsis': 0,
    'longitude of ascending node': 0,
    'centre mass': JUPITER['mass']
}

