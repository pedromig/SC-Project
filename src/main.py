#!/usr/bin/env python3
import simcx

from hiv.models import HIVModel, GenericVirusModel
from hiv.simulators import (VirusIterator, VirusEulerSimulator,
                            VirusHeunSimulator, VirusRK4Simulator)

from hiv.visuals import VirusVisual, VirusPhaseSpace

# Phase Space
# Different Between Orbits
# Valores Proprios Sistema (estabilidade)


class VirusIntegrator:
    def __init__(self: object,
                 model: GenericVirusModel,
                 integrator=VirusRK4Simulator) -> None:
        self._model = model
        self._integrator = integrator

    def __call__(self: object, *args, **kwargs) -> None:
        sim = self._integrator(self._model, *args, **kwargs)
        vis = VirusVisual(sim)
        display = simcx.Display()
        display.add_simulator(sim)
        display.add_visual(vis)
        simcx.run()


def phase_space(model, *args):
    sim = VirusIterator(model, *args)
    vis = VirusVisual(sim)
    display = simcx.Display()
    display.add_simulator(sim)
    display.add_visual(vis)
    simcx.run()


# Integration Methods
def integration(model, *args):
    euler = VirusIntegrator(model, VirusEulerSimulator)
    # heun = VirusIntegrator(model, VirusHeunSimulator)
    # rk4 = VirusIntegrator(hiv, VirusRK4Simulator)
    euler(*args)


if __name__ == "__main__":
    x0 = [1.2320968490204192]
    v0 = [2.4298860334362407]
    z0 = 2.0218848811803576

    chaotic = HIVModel(r=0.8976409287479137,
                       p=0.3730231282375587,
                       q=0.07271925334265861,
                       c=0.966378536424239,
                       b=-0.45482582469388344,
                       u=0.8452628018285122,
                       k=0.45417337083720355)

    periodic = HIVModel(r=-0.8330282398491504,
                        p=-0.1889978596607591,
                        q=0.3227333626934952,
                        c=-0.3543348404606479,
                        b=-0.8026310615455257,
                        u=0.4755867761890036,
                        k=0.06319226271725653)

    fixed = HIVModel(k=0.7207790085804264,
                     b=0.8566381427976779,
                     u=0.07955290482250144,
                     c=-0.36286047460543713,
                     r=-0.575907636089783,
                     p=0.09287266122633797,
                     q=-0.15983849314778742)

    # integration(chaotic, x0, v0, z0)
    phase_space(chaotic, x0, v0, z0)
