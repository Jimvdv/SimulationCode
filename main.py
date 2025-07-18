
from System import System
from Simulation import Simulation

from Data.ParticleDicts import *
from Data.AdvanceMethod import AdvanceMethod

time = 10
dt = 0.01
advance_method = AdvanceMethod.B

particle_dicts = [MERCURY, VENUS, EARTH, MARS, JUPITER, SATURN, URANUS, NEPTUNE, PLUTO]

system = System.system_from_dicts(particle_dicts, central_body_dict=SUN)

sim = Simulation(

    system,
    dt,
    time,
    store_data=True,
    central_body_movement=True,
    effective_central_mass=True,
    advance_method=advance_method,

)

sim.run()
sim.save()



