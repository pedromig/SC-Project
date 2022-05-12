#!/usr/bin/env python3
import simcx
import matplotlib.pyplot as plt

from hiv.models import HIVModel, GenericVirusModel
from hiv.simulators import (VirusIterator, VirusEulerSimulator,
                            VirusHeunSimulator, VirusRK4Simulator)

from hiv.visuals import VirusVisual, VirusPhaseSpace


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


def orbit(model, *args):
    sim = VirusIterator(model, *args)
    vis = VirusPhaseSpace(sim)
    display = simcx.Display()
    display.add_simulator(sim)
    display.add_visual(vis)
    simcx.run()


# Integration Methods
def integration(model, *args, **kwargs):
    integrator = VirusIntegrator(model, VirusEulerSimulator)
    integrator = VirusIntegrator(model, VirusHeunSimulator)
    integrator = VirusIntegrator(model, VirusRK4Simulator)
    integrator(*args, **kwargs)


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

    x = [100.]
    v = [10.]
    z = 100.

    test = HIVModel(k=0.01,
                    b=0.01,
                    u=0.01,
                    c=0.01,
                    r=0.8,
                    p=0.01,
                    q=0.01)

    # integration(test, x, v, z)
    orbit(test, x, v, z)
    plt.savefig("hiv.pdf")
