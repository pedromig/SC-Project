import simcx
import numpy as np

from .models import GenericVirusModel


class VirusSimulator(simcx.Simulator):
    def __init__(self: object, model: GenericVirusModel,
                 x: list[float], v: list[float], z: list[float],
                 step: float = 0.1, start: float = 0.0) -> None:
        super(VirusSimulator, self).__init__()
        self._model = model
        self._step = step
        self._x, self._v, self._z = np.array([x]), np.array([v]), np.array([z])
        self._t = np.array([start])
        self._mutants = len(self._x[0])

    def reset(self):
        self._x = np.array([self._x[0]])
        self._v = np.array([self._v[0]])
        self._z = np.array([self._z[0]])
        self._t = np.array([self._t[0]])


class VirusIterator(VirusSimulator):
    def step(self, delta=0):
        x, v = np.array([0.] * self._mutants), np.array([0.] * self._mutants)
        svi = sum(self._v[-1][i] for i in range(self._mutants))
        for i in range(self._mutants):
            x[i] = self._model.x(self._x[-1][i],
                                 self._v[-1][i],
                                 self._z[-1], svi)

            v[i] = self._model.v(self._x[-1][i],
                                 self._v[-1][i],
                                 self._z[-1])

        z = self._model.z(0, 0, self._z[-1], svi)
        t = self._t[-1] + self._step

        self._x = np.concatenate((self._x, [x]), axis=0)
        self._v = np.concatenate((self._v, [v]), axis=0)
        self._z = np.concatenate((self._z, [z]), axis=0)
        self._t = np.concatenate((self._t, [t]), axis=0)


class VirusEulerSimulator(VirusSimulator):
    def step(self: object, delta: float = 0.0) -> None:
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
    def step(self: object, delta: float = 0.0) -> None:
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
    def step(self: object, delta: float = 0) -> None:
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
