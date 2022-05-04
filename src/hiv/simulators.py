import simcx
import numpy as np

from models import HIVModel, GenericVirusModel

from simcx.simulators import Simulator
from simcx.visuals import MplVisual


class VirusSimulator(Simulator):
    def __init__(self, model: GenericVirusModel, x, v, z, step=0.1, start=0.0):
        super(VirusSimulator, self).__init__()
        self._model = model
        self._step = step
        self._x, self._v, self._z = np.array([x]), np.array([v]), np.array([z])
        self._t = np.array([start])
        self._mutants = len(self._x[0])


class VirusEulerSimulator(VirusSimulator):
    def step(self, delta=0):
        x, v = np.array([0.] * self._mutants), np.array([0.] * self._mutants)
        svi = sum(self._v[-1][i] for i in range(self._mutants))

        for i in range(self._mutants):

            x[i] = self._x[-1][i] + \
                self._model.x(self._x[-1][i],
                              self._v[-1][i],
                              self._z[-1], svi) * self._step

            v[i] = self._v[-1][i] + \
                self._model.v(self._x[-1][i],
                              self._v[-1][i],
                              self._z[-1]) * self._step

        z = self._z[-1] + self._model.z(0, 0, self._z[-1], svi) * self._step
        t = self._t[-1] + self._step

        self._x = np.concatenate((self._x, [x]), axis=0)
        self._v = np.concatenate((self._v, [v]), axis=0)
        self._z = np.concatenate((self._z, [z]), axis=0)
        self._t = np.concatenate((self._t, [t]), axis=0)


class VirusHeunSimulator(VirusSimulator):
    def step(self, delta=0):
        k1x, k1v = np.array(
            [0.] * self._mutants), np.array([0.] * self._mutants)

        svi = sum(self._v[-1][i] for i in range(self._mutants))
        k1z = self._z[-1] + self._model.z(0, 0, self._z[-1], svi) * self._step

        for i in range(self._mutants):
            k1x[i] = self._x[-1][i] + \
                self._model.x(self._x[-1][i],
                              self._v[-1][i],
                              self._z[-1], svi) * self._step

            k1v[i] = self._v[-1][i] + \
                self._model.v(self._x[-1][i],
                              self._v[-1][i],
                              self._z[-1]) * self._step

        x, v = np.array([0.] * self._mutants), np.array([0.] * self._mutants)
        sv1i = sum(k1v)

        for i in range(self._mutants):
            x[i] = self._x[-1][i] + \
                ((self._step / 2) * self._model.x(self._x[-1][i],
                                                  self._v[-1][i],
                                                  self._z[-1], svi)) + \
                ((self._step / 2) * self._model.x(k1x[i], k1v[i], k1z, sv1i))

            v[i] = self._v[-1][i] + \
                ((self._step / 2) * self._model.v(self._x[-1][i],
                                                  self._v[-1][i],
                                                  self._z[-1])) + \
                ((self._step / 2) * self._model.v(k1x[i], k1v[i], k1z))

        z = self._z[-1] + \
            ((self._step / 2) * self._model.z(0, 0, self._z[-1], svi)) +\
            ((self._step / 2) * self._model.z(0, 0, k1z, sv1i))

        t = self._t[-1] + self._step

        self._x = np.concatenate((self._x, [x]), axis=0)
        self._v = np.concatenate((self._v, [v]), axis=0)
        self._z = np.concatenate((self._z, [z]), axis=0)
        self._t = np.concatenate((self._t, [t]), axis=0)


class VirusRK4Simulator(VirusSimulator):
    def step(self, delta=0):
        # K1 Calculation
        k1x, k1v = np.array(
            [0.] * self._mutants), np.array([0.] * self._mutants)

        svi = sum(self._v[-1][i] for i in range(self._mutants))
        k1z = self._model.z(0, 0, self._z[-1], svi) * self._step

        for i in range(self._mutants):
            k1x[i] = self._model.x(self._x[-1][i],
                                   self._v[-1][i],
                                   self._z[-1], svi) * self._step

            k1v[i] = self._model.v(self._x[-1][i],
                                   self._v[-1][i],
                                   self._z[-1]) * self._step

        # K2 Calculation
        k2x, k2v = np.array(
            [0.] * self._mutants), np.array([0.] * self._mutants)

        sv2i = sum(k1v)
        k2z = self._model.z(0, 0, self._z[-1] + 0.5 * k1z, sv2i) * self._step

        for i in range(self._mutants):
            k2x[i] = self._model.x(self._x[-1][i] + 0.5 * k1x[i],
                                   self._v[-1][i] + 0.5 * k1v[i],
                                   self._z[-1] + 0.5 * k1z, sv2i)\
                * self._step

            k2v[i] = self._model.v(self._x[-1][i] + 0.5 * k1x[i],
                                   self._v[-1][i] + 0.5 * k1v[i],
                                   self._z[-1] + 0.5 * k1z) * self._step

        # K3 Calculation
        k3x, k3v = np.array(
            [0.] * self._mutants), np.array([0.] * self._mutants)

        sv3i = sum(k2v)
        k3z = self._model.z(0, 0, self._z[-1] + 0.5 * k2z, sv3i) * self._step

        for i in range(self._mutants):
            k3x[i] = self._model.x(self._x[-1][i] + 0.5 * k2x[i],
                                   self._v[-1][i] + 0.5 * k2v[i],
                                   self._z[-1] + 0.5 * k2z, sv3i)\
                * self._step

            k3v[i] = self._model.v(self._x[-1][i] + 0.5 * k2x[i],
                                   self._v[-1][i] + 0.5 * k2v[i],
                                   self._z[-1] + 0.5 * k2z) * self._step

        # K4 Calculation
        k4x, k4v = np.array(
            [0.] * self._mutants), np.array([0.] * self._mutants)

        sv4i = sum(k3v)
        k4z = self._model.z(0, 0, self._z[-1] + k3z, sv4i) * self._step

        for i in range(self._mutants):
            k4x[i] = self._model.x(self._x[-1][i] + k3x[i],
                                   self._v[-1][i] + k3v[i],
                                   self._z[-1] + k3z, sv4i) * self._step

            k4v[i] = self._model.v(self._x[-1][i] + k3x[i],
                                   self._v[-1][i] + k3v[i],
                                   self._z[-1] + k3z) * self._step

        # Final Result
        x, v = np.array([0.] * self._mutants), np.array([0.] * self._mutants)
        for i in range(self._mutants):
            x[i] = self._x[-1][i] + \
                (1/6) * (k1x[i] + 2 * k2x[i] + 2 * k3x[i] + k4x[i])
            v[i] = self._v[-1][i] + \
                (1/6) * (k1v[i] + 2 * k2v[i] + 2 * k3v[i] + k4v[i])

        z = self._z[-1] + (1/6) * (k1z + 2 * k2z + 2 * k3z + k4z)
        t = self._t[-1] + self._step

        self._x = np.concatenate((self._x, [x]), axis=0)
        self._v = np.concatenate((self._v, [v]), axis=0)
        self._z = np.concatenate((self._z, [z]), axis=0)
        self._t = np.concatenate((self._t, [t]), axis=0)


class VirusVisual(MplVisual):
    def __init__(self, sim: GenericVirusModel, title=None):
        super(VirusVisual, self).__init__(sim, width=1000, height=800)

        self.x, self.v = [], []
        self.plots = self.sim._mutants + 1
        self.ax = [
            self.figure.add_subplot(self.plots, 1, i + 1)
            for i in range(self.plots)]

        self.figure.suptitle(
            sim._model.__class__.__name__ if title is None else title,
            size=18)

        for i in range(self.sim._mutants):
            self.ax[i].set_title(f"Virus Mutant {i}")
            x, = self.ax[i].plot(self.sim._t, self.sim._v[:, i],
                                 "-",  label="Virus")
            v, = self.ax[i].plot(self.sim._t, self.sim._x[:, i],
                                 "-", label="Immune system response")
            self.x.append(x)
            self.v.append(v)
            self.ax[i].legend()

        self.ax[-1].set_title("Global immune system response")
        self.z, = self.ax[-1].plot(self.sim._t, self.sim._z,
                                   "-", label="Global immune system response")
        self.ax[-1].legend()

    def draw(self):
        for i in range(self.sim._mutants):
            self.x[i].set_data(self.sim._t, self.sim._x[:, i])
            self.v[i].set_data(self.sim._t, self.sim._v[:, i])
            self.ax[i].relim()
            self.ax[i].autoscale_view()
        self.z.set_data(self.sim._t, self.sim._z)
        self.ax[-1].relim()
        self.ax[-1].autoscale_view()


if __name__ == "__main__":

    hiv = HIVModel(r=0.5, p=0.5, q=0.2, b=0.2, c=1, u=0.01, k=0.8)
    sim = VirusRK4Simulator(hiv, [150.0, 150.0], [
        100.0, 200.0], 100.0, step=0.01)
    vis = VirusVisual(sim)
    display = simcx.Display()
    display.add_simulator(sim)
    display.add_visual(vis)

    simcx.run()
