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
    vis = VirusPhaseSpace(sim)
    display = simcx.Display()
    display.add_simulator(sim)
    display.add_visual(vis)
    simcx.run()


if __name__ == "__main__":

    # Models
    hiv = HIVModel(r=0.3, p=0.1, q=0.1,
                   c=0.8, b=0.4,  u=0.2,
                   k=0.8)

    # Integration Methods
    euler = VirusIntegrator(hiv, VirusEulerSimulator)
    heun = VirusIntegrator(hiv, VirusHeunSimulator)
    rk4 = VirusIntegrator(hiv, VirusRK4Simulator)

    rk4([200.0], [20.], [100.])
