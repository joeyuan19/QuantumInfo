import numpy as np
import qubit

class NotYetImplemented(Exception):
    pass

def pad(s,spaces):
    l = len(s)
    return s + ' '*max(0,spaces-l)

class Gate(object):
    def __init__(self,gate_matrix):
        self.gate = gate_matrix
        self.size = gate_matrix.shape

    def __mul__(self,other):
        if isinstance(other,qubit.QubitArray):
            return qubit.QubitArray(np.dot(self.gate,other.state))
        elif isinstance(other,Gate):
            return Gate(np.dot(self.gate,other.gate))
        return Gate(other*self.gate)

    def tensor(self,other):
        N,M = self.size[0],other.size[0]
        new = 1j*np.zeros((N*M,N*M))
        for n1 in range(N):
            for n2 in range(N):
                for m1 in range(M):
                    for m2 in range(M):
                        new[n1*M+m1,n2*M+m2] = self.gate[n1,n2]*other.gate[m1,m2]
        return Gate(new)

    def __str__(self):
        l = tuple()
        for row in self.gate:
            l += tuple(len(str(i)) for i in row)
        m = max(l)
        return '\n'.join('('+' '.join(pad(str(i),m) for i in row)+')' for row in self.gate)

    def __truediv__(self,other):
        return Gate(self.gate/other)

    def __add__(self,other):
        if isinstance(other,Gate):
            return Gate(self.gate+other.gate)
    
    def __sub__(self,other):
        if isinstance(other,Gate):
            return Gate(self.gate-other.gate)

    def __neg__(self):
        return Gate(-self.gate)

class H(Gate):
    def __init__(self):
        super().__init__(np.array([[1,1],[1,-1]])/np.sqrt(2))

class X(Gate):
    def __init__(self):
        super().__init__(np.array([[0,1],[1,0]]))

class Y(Gate):
    def __init__(self):
        super().__init__(np.array([[0,-1j],[1j,0]]))

class Z(Gate):
    def __init__(self):
        super().__init__(np.array([[1,0],[0,-1]]))

class I(Gate):
    def __init__(self,N=1):
        N = 2**N
        gate = np.zeros((N,N))
        for n in range(N):
            gate[n,n] = 1
        super().__init__(gate)

class CNOT(Gate):
    def __init__(self,N=2,control=0,target=1):
        N = 2**N
        if N == 4:
            gate = np.zeros((N,N))
            if control == 0:
                for n in range(2):
                    gate[n,n] = 1
                gate[2,3] = 1
                gate[3,2] = 1
            elif control == 1:
                gate = np.zeros((N,N))
                gate[1,1] = 1
                gate[2,2] = 1
                gate[1,3] = 1
                gate[3,1] = 1
        else:
            raise NotYetImplemented() 
        super().__init__(gate)

class PHASE(Gate):
    def __init__(self,phase_angle):
        super().__init__(np.array([[1,0],[0,np.exp(1j*phase_angle)]]))
