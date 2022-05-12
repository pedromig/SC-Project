import simcx

from .models import GenericVirusModel
from .simulators import VirusIterator

from mpl_toolkits.mplot3d import Axes3D


class VirusVisual(simcx.MplVisual):
    def __init__(self: object,
                 sim: GenericVirusModel, title: str = None) -> None:
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
            v, = self.ax[i].plot(self.sim._t, self.sim._v[:, i],
                                 "-", label="Virus")
            x, = self.ax[i].plot(self.sim._t, self.sim._x[:, i],
                                 "-", label="Immune system response")
            self.x.append(x)
            self.v.append(v)
            self.ax[i].legend()

        self.ax[-1].set_title("Global immune system response")
        self.z, = self.ax[-1].plot(self.sim._t, self.sim._z,
                                   "-", label="Global immune system response")
        self.ax[-1].legend()

    def draw(self: object) -> None:
        for i in range(self.sim._mutants):
            self.x[i].set_data(self.sim._t, self.sim._x[:, i])
            self.v[i].set_data(self.sim._t, self.sim._v[:, i])
            self.ax[i].relim()
            self.ax[i].autoscale_view()
        self.z.set_data(self.sim._t, self.sim._z)
        self.ax[-1].relim()
        self.ax[-1].autoscale_view()


class VirusPhaseSpace(simcx.MplVisual):
    def __init__(self: object, sim: VirusIterator, title=None, **kwargs):
        super(VirusPhaseSpace, self).__init__(
            sim, width=1000, height=800, **kwargs)

        self.ax = [self.figure.add_subplot(self.sim._mutants, 1, i + 1,
                                           projection="3d")
                   for i in range(self.sim._mutants)]
        self.figure.suptitle(
            sim._model.__class__.__name__ if title is None else title,
            size=18)

        self.lines = []
        for i in range(self.sim._mutants):
            self.ax[i].set_title("Virus Phase Space")
            line, = self.ax[i].plot(
                self.sim._x[:, i], self.sim._v[:, i], self.sim._z)
            self.lines.append(line)
            self.ax[i].set_xlim(-20, 20)
            self.ax[i].set_ylim(-20, 20)
            self.ax[i].set_zlim(-20, 20)

    def draw(self: object) -> None:
        for i in range(self.sim._mutants):
            self.lines[i].set_data_3d(
                self.sim._x[:, i], self.sim._v[:, i], self.sim._z)
            self.ax[i].relim()
            self.ax[i].autoscale_view()
