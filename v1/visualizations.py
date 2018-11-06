import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import numpy as np

from qubit import * 
from gates import *

class AnimatedHistogram(object):
    def __init__(self,qubit,basis=None,skip=100):
        self.qubit   = qubit
        self.basis   = basis 
        self.results = []
        self.single  = [[] for i in range(self.qubit.N)]
        self.skip    = skip
        self.states  = np.arange(qubit.size)

    def init_plot(self):
        fig     = plt.figure()
        self.ax = fig.add_subplot(111)
        return fig
    
    def init_single_plot(self):
        fig = plt.figure()
        self.axes = []
        for n in range(self.qubit.N):
            self.axes.append(fig.add_subplot(self.qubit.N,1,n+1))
        return fig

    def update(self,frame):
        self.measure()
        self.replot(frame)
   
    def update_single(self,frame):
        self.single_measurements()
        self.replot_single(frame)

    def replot(self,frame):
        self.ax.cla()
        self.ax.hist(self.results,bins=np.arange(self.qubit.size+1)-.5,normed=True)
        self.ax.set_title('N = '+str(frame*self.skip))
        self.ax.set_xlabel('State')
        self.ax.set_xticks(self.states)
        self.ax.set_xticklabels(self.labels)
        self.ax.set_ylabel('P')
        self.ax.set_ylim((0,1))
    
    def replot_single(self,frame):
        for i,ax in enumerate(self.axes):
            ax.cla()
            ax.hist(self.single[i],bins=np.arange(2+1)-.5,normed=True)
            ax.set_title('Qubit ' + chr(ord('A')+i) + ' N = '+str(frame*self.skip))
            ax.set_xlabel('State')
            ax.set_xticks((0,1))
            ax.set_xticklabels(('$|0>_'+chr(ord('A')+i)+'$','$|1>_'+chr(ord('A')+i)+'$'))
            ax.set_ylim((0,1))

    def basis_measure(self):
        for i in range(self.skip):
            self.results.append(self.qubit.measure_by_basis(self.basis))
    
    def computational_measure(self):
        for i in range(self.skip):
            self.results.append(self.qubit.measure())

    def single_measurements(self):
        for n in range(self.qubit.N):
            for i in range(self.skip):
                self.single[n].append(self.qubit.measure_partial((n,))[0])
    
    def animate(self):
        fig = self.init_plot()
        if self.basis is None:
            self.measure = self.computational_measure
            self.labels  = [self.qubit.ket(i) for i in np.arange(self.qubit.size)]
        else:
            self.measure = self.basis_measure
            self.labels  = [i.latex_label() for i in self.basis]
        ani = animation.FuncAnimation(fig,self.update,interval=1)
        plt.show()
    
    def animate_single(self):
        fig = self.init_single_plot()
        ani = animation.FuncAnimation(fig,self.update_single,interval=1)
        plt.show()
        
    def plot(self,N=1000):
        self.labels  = [self.qubit.ket(i) for i in np.arange(self.qubit.size)]
        fig = self.init_plot()
        for n in range(N):
            self.computational_measure()
        self.ax.cla()
        self.ax.hist(self.results,bins=np.arange(self.qubit.size+1)-.5,normed=True)
        self.ax.set_title('N = '+str(N))
        self.ax.set_xlabel('State')
        self.ax.set_xticks(self.states)
        self.ax.set_xticklabels(self.labels)
        self.ax.set_ylabel('P')
        self.ax.set_ylim((0,1))
        plt.show()
        

# |psi> = a|0> + b|1>
# |psi> = e^(ig)(cos(t/2)|0> + e^(ip)*sin(t/2)|1>)
# a = cos(t/2)
# t = acos(a)/2
# b = e^(ip)*sin(t/2)
# c = b/sin(t/2)
# cos(p) + i*sin(p) = c
# cos(p) = real(c)
# p = acos(real(c))
# sin(p) = imag(c)
# p = asin(imag(c))
# x = sin(p)*sin(t)
# y = cos(p)*sin(t)
# z = cos(p)

class BlochSphere(object):
    def __init__(self,qubit):
        if qubit.N == 1:
            self.qubit = qubit
        else:
            raise BlochError('Bloch Sphere too high order for N > 1')

    def angles(self):
        return self._angles(self.qubit)
    
    def _angles(self,qubit):
        #global_phase = 
        theta = 2*np.arccos(qubit.state[0])
        if np.allclose(0,theta) or np.allclose(np.pi,theta):
            phi = 0 
        else:    
            c = qubit.state[1]/np.arcsin(theta/2)
            phi = np.arcsin(np.imag(c))
            phi = np.arccos(np.real(c))
        return theta,phi

    def xyz(self):
        theta,phi = self.angles()
        return self._xyz(theta,phi)

    def _xyz(self,theta,phi):
        if phi is None: 
            x = 0 
            y = 0 
        else:
            x = np.cos(phi)*np.sin(theta) 
            y = np.sin(phi)*np.sin(theta) 
        z = np.cos(theta)
        return x,y,z

    def plot(self):
        self._plot(*self.xyz())
        plt.show()

    def _plot(self,x,y,z):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.quiver(0, 0, 0, x, y, z)
        ax.plot((0,),(0,),(0,),'ro')
        ax.set_xlim([-1.1, 1.1])
        ax.set_ylim([-1.1, 1.1])
        ax.set_zlim([-1.1, 1.1])
        return fig

    def animate(self,final,N):
        theta0,phi0 = self.angles()
        theta1,phi1 = self._angles(final)
        self.N      = N
        self.theta  = np.linspace(theta0,theta1,N)
        if phi0 is None or phi1 is None: 
            self.phi = np.zeros(N)
        else:
            self.phi = np.linspace(phi0,phi1,N)
        fig = self.init_plot()
        ani = animation.FuncAnimation(fig,self.update,frames=N,interval=1)
        plt.show()

    def init_plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x,y,z = self.xyz()
        ax.quiver(0, 0, 0, x, y, z)
        ax.plot((0,),(0,),(0,),'ro')
        ax.set_xlim([-1.1, 1.1])
        ax.set_ylim([-1.1, 1.1])
        ax.set_zlim([-1.1, 1.1])
        self.ax = ax
        return fig

    def update(self,frame):
        if frame < self.N:
            x,y,z = self._xyz(self.theta[frame],self.phi[frame])
            self.ax.cla()
            self.ax.quiver(0, 0, 0, x, y, z)
            self.ax.plot((0,),(0,),(0,),'ro')
            self.ax.set_xlim([-1.1, 1.1])
            self.ax.set_ylim([-1.1, 1.1])
            self.ax.set_zlim([-1.1, 1.1])

class BellInequality(object):
    def __init__(self,steps,trials=100):
        self.steps    = steps + 1
        self.trials   = trials 
        self.measured = [[[],[]],[[],[]]]
        a1,a2 = X(),Y()
        b1,b2 = X(),Y()
        self.op    = [a.tensor(a),
                      a.tensor(b),
                      b.tensor(a),
                      b.tensor(b)]
        self.theta = np.linspace(0,2*np.pi,steps)-np.pi/4
        self.asoln = np.vectorize(self.analytic)(self.theta) 
    
    def analytic(self,theta):
        psi = QubitArray(np.array([1,0,0,np.exp(1j*theta)])/np.sqrt(2))
        C   = np.ones(4)
        C[-1] = -1
        return sum(C*np.array([psi.inner(op*psi) for op in self.op]))

    def measure(self,theta):
        psi = QubitArray(np.array([1,0,0,np.exp(1j*theta)])/np.sqrt(2))
        print('theta =',theta,psi)
        S = []
        for op in self.op:
            q = op*psi
            Es = sum(q.in_state(psi) for n in range(self.trials))
            for n in range(self.trials):
                m = q.in_state(psi)
                Es += m
            S.append((2*Es/self.trials)-1)
        C = np.ones(4)
        C[3] = -1
        S = np.sum(C*np.array(S))
        if -2 < S < 2:
            self.measured[0][0].append(theta)
            self.measured[0][1].append(S)
        else:
            self.measured[1][0].append(theta)
            self.measured[1][1].append(S)

    def init_plot(self):
        fig = plt.figure()
        ax  = fig.add_subplot(111)
        ax.plot(self.theta,2*(np.sin(self.theta)+np.cos(self.theta)),'g-',label='Analytic')
        ax.axhline(+2,0,1,c='k',linestyle='--',label='Local Hidden Variable')
        ax.axhline(-2,0,1,c='k',linestyle='--')
        self.lines = []
        line, = ax.plot(self.measured[0][0],self.measured[0][1],'bo')
        self.lines.append(line)
        line, = ax.plot(self.measured[1][0],self.measured[1][1],'ro')
        self.lines.append(line)
        ax.set_xlim([self.theta.min(),self.theta.max()])
        ax.set_ylim([-3, 3])
        ax.legend(loc='best')
        return fig

    def replot(self):
        self.lines[0].set_xdata(self.measured[0][0])
        self.lines[0].set_ydata(self.measured[0][1])
        self.lines[1].set_xdata(self.measured[1][0])
        self.lines[1].set_ydata(self.measured[1][1])
    
    def update(self,frame):
        if not self.done:
            self.measure(self.theta[frame])
            self.replot()
        if frame == self.steps-2:
            self.done = True

    def animate(self):
        self.done = False
        fig = self.init_plot()
        ani = animation.FuncAnimation(fig,self.update,interval=10)
        plt.show()

class Entanglement(object):
    pass

class Teleportation(object):
    def __init__(self,q):
        self.A = {
            'q'      : q,
            'dist'   : [q.measure() for i in range(10000)],
            'labels' : ['$'+q.ket(i)+'_A$' for i in np.arange(q.size)],
            'bins'   : np.arange(q.size+1)-.5
        }
        self.B = {
            'dist'   : [], 
            'labels' : ['$'+q.ket(i)+'_B$' for i in np.arange(q.size)],
            'bins'   : np.arange(q.size+1)-.5
        }
        self.skip  = 1
        self.qubit = q*PhiMinus()
        self.basis = extend_basis(bell_basis(),1)
        self.gates = []
        g = I(2).tensor(Z()*X())
        self.gates.append(g)
        g = I(2).tensor(X())
        self.gates.append(g)
        g = I(2).tensor(Z())
        self.gates.append(g)
        g = I(3) 
        self.gates.append(g)

    def init_plot(self):
        fig = plt.figure()
        self.axes = []
        ax  = fig.add_subplot(121)
        ax.hist(self.A['dist'],bins=self.A['bins'],normed=True)
        ax.set_xticks([0,1])
        ax.set_xticklabels(self.A['labels'])
        ax.set_xlabel('State')
        ax.set_ylabel('P')
        ax.set_ylim(0,1.1)
        ax.set_title('Input Qubit (Alice) N = ' + str(len(self.A['dist'])))
        self.axes.append(ax)
        self.axes.append(fig.add_subplot(122))
        return fig
    
    def update(self,frame):
        for s in range(self.skip):
            m = self.qubit.real_measure_partial_by_basis((0,1),self.basis)
            m = 2*m[0] + m[1]
            self.qubit = self.gates[m]*self.qubit
            m = self.qubit.measure()%2
            self.B['dist'].append(m)
            self.qubit = q*PhiMinus()
            self.replot()

    def replot(self):
        ax = self.axes[1]
        ax.cla()
        ax.hist(self.B['dist'],bins=self.B['bins'],normed=True)
        ax.set_xticks([0,1])
        ax.set_xticklabels(self.B['labels'])
        ax.set_xlabel('State')
        ax.set_ylabel('P')
        ax.set_ylim(0,1.1)
        ax.set_title('Output Qubit (Bob) N = ' + str(len(self.B['dist'])))

    def animate(self):
        fig = self.init_plot()
        ani = animation.FuncAnimation(fig,self.update,interval=10)
        plt.show()

class Decoherence(object):
    def __init__(self,tau,noise,N):
        self.N     = N 
        self.t     = np.linspace(0,2*tau,2*N)
        self.p     = np.zeros(2*N)
        self.qubit = Zero()
        self.tau   = tau
        self.noise = noise
    
    def angles(self):
        return self._angles(self.qubit)
    
    def _angles(self,qubit):
        #global_phase = 
        theta = 2*np.arccos(qubit.state[0])
        if np.allclose(0,theta) or np.allclose(np.pi,theta):
            phi = 0 
        else:    
            c = qubit.state[1]/np.arcsin(theta/2)
            phi = np.arcsin(np.imag(c))
            phi = np.arccos(np.real(c))
        return theta,phi

    def xyz(self):
        theta,phi = self.angles()
        return self._xyz(theta,phi)

    def _xyz(self,theta,phi):
        if phi is None: 
            x = 0 
            y = 0 
        else:
            x = np.cos(phi)*np.sin(theta) 
            y = np.sin(phi)*np.sin(theta) 
        z = np.cos(theta)
        return x,y,z

    def init_plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(121, projection='3d')
        x,y,z = self.xyz()
        ax.quiver(0, 0, 0, x, y, z)
        ax.plot((0,),(0,),(0,),'ro')
        ax.set_xlim([-1.1, 1.1])
        ax.set_ylim([-1.1, 1.1])
        ax.set_zlim([-1.1, 1.1])
        self.ax = ax
        ax2 = fig.add_subplot(122)
        self.line, = ax2.plot([],[],'ro')
        ax2.set_ylim((0,1))
        ax2.set_xlim((self.t[0],self.t[-1]))
        self.ax2 = ax2
        return fig

    def replot(self,frame):
        self.line.set_xdata(self.t[:frame])
        self.line.set_ydata(self.p[:frame])
        x,y,z = self.xyz()
        self.ax2.cla()
        self.ax2.quiver(0, 0, 0, x, y, z)
        self.ax2.plot((0,),(0,),(0,),'ro')
        self.ax2.set_xlim([-1.1, 1.1])
        self.ax2.set_ylim([-1.1, 1.1])
        self.ax2.set_zlim([-1.1, 1.1])

    def update(self,frame):
        g = Gate(np.array([[1,0],[0,np.exp(1j*self.t[frame]/self.tau)]]))
        r = random.random()
        if r > self.noise:
            self.qubit = g*self.qubit
        for n in range(self.N):
            self.p[frame] += (1+self.qubit.measure())%2
        self.p[frame] = self.p[frame]/self.N
        self.replot(frame)

    def animate(self):
        fig = self.init_plot()
        ani = animation.FuncAnimation(fig,self.update,frames=2*self.N,interval=10)
        plt.show()

class GroverSearch(object):
    def __init__(self,search_term):
        print(search_term)
        self.search_term = search_term
        self.N    = search_term.N
        self.size = search_term.size
        II = I(self.N)
        Uf = II - search_term.outer(search_term)*2
        Uh = H()
        for n in range(N-1):
            Uh = Uh.tensor(H())
        z = Zero(N=self.N)
        U0 = II - z.outer(z)*2
        self.psi = Uh*z
        self.Q = -Uh*U0*Uh*Uf
        self.labels = [self.search_term.ket(i) for i in np.arange(self.size)]
        self.states = np.arange(self.size)
        self.px,self.py = [],[]
        self.count = 0
    
    def iterate(self):
        self.psi = self.Q*self.psi
        self.count += 1

    def init_plot(self):
        fig = plt.figure(figsize=(10,10))
        self.axl = fig.add_subplot(211)
        self.axl.set_ylim((0,1))
        self.axl.set_xlim((0,1))
        self.axl.set_ylabel('P('+str(self.search_term)+')')
        self.axl.axvline(np.pi/4*np.sqrt(self.size),0,1)
        self.line, = self.axl.plot(-1,-1,'go--')
        self.axh = fig.add_subplot(212)
        self.axh.hist([self.psi.measure() for n in range(1000)],bins=np.arange(self.size+1)-.5,normed=True)
        self.axh.set_title('Iterations = 0 (measurments =1000)')
        self.axh.set_xlabel('State')
        self.axh.set_xticks(self.states)
        self.axh.set_xticklabels(self.labels,rotation=-90)
        self.axh.set_ylabel('P')
        self.axh.set_ylim((0,1))
        return fig

    def replot(self):
        self.px.append(self.count)
        p = self.search_term.inner(self.psi)
        self.py.append(np.conjugate(p)*p)
        self.line.set_xdata(self.px)
        self.axl.set_xlim((0,max(1,max(self.px))))
        self.line.set_ydata(self.py)
        self.axh.cla()
        bins = np.arange(self.size+1)-.5
        self.axh.hist([self.psi.measure() for n in range(1000)],bins=bins,normed=True)
        self.axh.set_title('Iterations = '+str(self.count) + ' (measurments =1000)')
        self.axh.set_xlabel('State')
        self.axh.set_xticks(self.states)
        self.axh.set_xticklabels(self.labels,rotation=-90)
        self.axh.set_ylabel('P')
        self.axh.set_xlim((bins.min(),bins.max()))
        self.axh.set_ylim((0,1))
 
    def update(self,frame):
        self.iterate()
        if frame < 5*self.size:
            self.replot()

    def animate(self):
        fig = self.init_plot()
        ani = animation.FuncAnimation(fig,self.update,interval=100)
        plt.show()

class SimonPeriod(object):
    def __init__(self):
        pass

    

N = 4
s = EvenlyDistributed(N)
s.real_measure()
GroverSearch(s).animate()
