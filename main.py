import simcx
import numpy as np

class EulerSimulator(simcx.simulators.Simulator):
    def __init__(self, func, init_state_v,init_state_z,init_state_x, Dt):
        super(EulerSimulator, self).__init__()
        self._func = func
        self.x_axis = [0]
        self.v = np.array([init_state_v])
        self.z = np.array([init_state_z])
        self.x = np.array([init_state_x])
        self._number_of_disease = len(init_state_v)
        self.Dt = Dt


    def step(self, delta=0):
        new_v = []
        new_x = []
        for disease in range(self._number_of_disease):
            new_v.append(self.v[-1][disease] + self._func[0](self.x[-1][disease],self.v[-1][disease],self.z[-1])*self.Dt)
            new_x.append(self.x[-1][disease] + self._func[1](self.x[-1][disease],self.v[-1][disease],self.z[-1])*self.Dt)
        self.z =np.vstack((self.z,[self.z[-1] + self._func[2](0,np.sum(new_v),self.z[-1])*self.Dt]))
        
        self.v = np.vstack([self.v,np.array(new_v).flatten()])
        self.x = np.vstack([self.x,np.array(new_x).flatten()])
        self.x_axis += [self.x_axis[-1] + self.Dt]


class EulerVisual(simcx.MplVisual):
    def __init__(self, sim : EulerSimulator):
        super(EulerVisual, self).__init__(sim)
        self.ax = [self.figure.add_subplot(self.sim._number_of_disease,1,i+1) for i in range(self.sim._number_of_disease)]
        self.l = []
        self.l2 = []
        self.l3 = []
        for i in range(self.sim._number_of_disease):
            l, = self.ax[i].plot(self.sim.x_axis, self.sim.v[:,i], '-.',color='b',label="V value")
            l2, = self.ax[i].plot(self.sim.x_axis, self.sim.x[:,i], '-.',color='r',label="X value")
            l3, = self.ax[i].plot(self.sim.x_axis, self.sim.z, '-.',color='y',label="Z value")
            self.l += [l]
            self.l2 += [l2]
            self.l3 += [l3]

    def draw(self):
        for i in range(self.sim._number_of_disease):
            self.l[i].set_data(self.sim.x_axis, self.sim.v[:,i])
            self.l2[i].set_data(self.sim.x_axis, self.sim.x[:,i])
            self.l3[i].set_data(self.sim.x_axis, self.sim.z)
            self.ax[i].relim()
            self.ax[i].autoscale_view()


def make_functs(r,p,q,b,c,u,k):
    return (lambda x, v, z: -v*(r-p*x-q*z),
            lambda x,v,z: c*v-b*x-u*v*x,
            lambda x,v,z:k*v-b*z-u*v*z)


y0= [2.0,4.0]
x0= [20.0,22.]
z0 = [1.0]
    
if __name__ == "__main__":
    Dt = 0.01
    functs = make_functs(0.5,0.5,0.2,0.2,1,0.01,0.8)
    sim = EulerSimulator(functs, y0,z0,x0, Dt)
    vis = EulerVisual(sim)
    display = simcx.Display()
    display.add_simulator(sim)
    display.add_visual(vis)
    simcx.run()
