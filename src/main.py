#!/usr/bin/env python3
import simcx

from hiv.models import HIVModel
from hiv.simulators import (VirusIterator, VirusEulerSimulator,
                            VirusHeunSimulator, VirusRK4Simulator)

from hiv.visuals import VirusVisual, VirusPhaseSpace

# Phase Space
# Different Between Orbits
# Valores Proprios Sistema (estabilidade)


def hiv_euler():
    hiv = HIVModel(r=0.5, p=0.5, q=0.2, b=0.2, c=1, u=0.01, k=0.8)
    sim = VirusEulerSimulator(hiv, [150.0, 150.0], [
        100.0, 200.0], 100.0, step=0.01)
    vis = VirusVisual(sim)
    display = simcx.Display()
    display.add_simulator(sim)
    display.add_visual(vis)
    simcx.run()


def hiv_heun():
    hiv = HIVModel(r=0.5, p=0.5, q=0.2, b=0.2, c=1, u=0.01, k=0.8)
    sim = VirusHeunSimulator(hiv, [150.0, 150.0], [
        100.0, 200.0], 100.0, step=0.01)
    vis = VirusVisual(sim)
    display = simcx.Display()
    display.add_simulator(sim)
    display.add_visual(vis)
    simcx.run()


def hiv_rk4():
    hiv = HIVModel(r=0.3, p=0.2, q=0.2, b=0.2, c=0.4, u=0.02, k=0.8)
    sim = VirusRK4Simulator(hiv, [1000.0, 150.0], [
        80.0, 30.0], 0.0, step=0.01)
    vis = VirusVisual(sim)
    display = simcx.Display()
    display.add_simulator(sim)
    display.add_visual(vis)
    simcx.run()


if __name__ == "__main__":
    # hiv_rk4()
    hiv = HIVModel(r=0.5, p=0.5, q=0.2, b=0.2, c=0.2, u=0.2, k=0.8)
    sim = VirusIterator(hiv, [10.0], [5.0], 10.0)
    vis = VirusPhaseSpace(sim)
    display = simcx.Display()
    display.add_simulator(sim)
    display.add_visual(vis)
    simcx.run()
