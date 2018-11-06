import numpy as np
import random
import gates

MP = 4/3 - 1/3 - 1

def norm(x):
    return x*np.conjugate(x)

# sum(vv*) = A
def normalize(v):
    return v/np.sqrt(np.sum(norm(v)))

def bra(n,N=1):
    s = str(bin(n))[2:]
    pad = "0"*max(0,N-len(s))
    return '<'+pad+s+'|'

def ket(n,N=1):
    s = str(bin(n))[2:]
    pad = "0"*max(0,N-len(s))
    return '|'+pad+s+'>'

class QubitArray(object):
    sym = ''
    def __init__(self,state):
        if len(state.shape) != 2:
            state = state[...,None]
        self.size  = max(state.shape)
        self.N     = int(np.log2(self.size))
        state = normalize(state)
        self.state = state
        self.basis = None

    def initialize(self,state):
        self.state = state

    def bra(self,i):
        return bra(i,N=self.N)

    def ket(self,i):
        return ket(i,N=self.N)

    def p_dist(self):
        return norm(self.state)

    def _p_dist(self,state):
        return norm(state)

    def change_basis(self,basis):
        self.basis = basis
        self.state = self._change_basis(self.state,basis)
        self.label = [base.sym for base in basis]

    def _change_basis(self,state,basis):
        return np.array([np.sum(np.transpose(base.state)*self.state) for base in basis])

    def computational_basis(self):
        if self.basis is not None:
            comp_basis = computational_basis(self.N)
            [base.change_basis(self.basis) for base in comp_basis]
            self.change_basis(comp_basis)
            self.basis = None
    
    def measure(self):
        return self._measure(self.p_dist())

    def measure_by_basis(self,basis):
        P = []
        for base in basis:
            p = base.inner(self)
            P.append(np.sum(p*np.conjugate(p)))
        return self._measure(P)

    def in_state(self,state):
        p = state.inner(self)
        p = np.sum(p*np.conjugate(p))
        return self._measure([p,1-p])

    def _measure(self,P):
        r = random.random()
        s = 0
        for i,p in enumerate(P):
            s += p
            if r <= s:
                return i
        return i

    def real_measure(self):
        i,self.state = self._real_measure()
        return i
    
    def _real_measure(self):
        i = self.measure()
        s = np.zeros((self.size,1))
        s[i] = 1
        return i,s
    
    def measure_single(self,qubit):
        i,s = self._measure_single(self.state,qubit)
        return i
   
    def real_measure_single(self,qubit):
        i,self.state = self._measure_single(self.state,qubit)
        return i

    def _measure_single(self,state,qubit):
        c = np.zeros(self.size)
        period = self.size//(2**(qubit+1))
        for n in range(0,self.size,2*period):
            c[n:n+period] += 1
        P = self._p_dist(state)
        P = [np.sum(c*P),np.sum((c+1)%2*P)]
        i = self._measure(P)
        if i == 0:
            s = normalize(state*c)
        else:
            s = normalize(state*((c+1)%2))
        return i,s

    def measure_partial(self,qubits):
        results,s = self._measure_partial(self.state,qubits) 
        return results 
    
    def measure_partial_by_basis(self,qubits,basis):
        state = self._change_basis(self.state,basis) 
        results,s = self._measure_partial(state,qubits)
        return results
    
    def real_measure_partial(self,qubits):
        results,self.state = self._measure_partial(self.state,qubits)
        return results

    def real_measure_partial_by_basis(self,qubits,basis):
        state = self._change_basis(self.state,basis)
        results,self.state = self._measure_partial(state,qubits)
        self.computational_basis()
        return results

    def _measure_partial(self,state,qubits):
        results = []
        for qubit in qubits:
            r,state = self._measure_single(state,qubit)
            results.append(r)
        return results,state

    def copy(self):
        return QubitArray(self.state)

    def __str__(self):
        if self.basis is None:
            s = ''
            for i in range(self.size):
                if not np.allclose(self.state[i],0):
                    if np.allclose(self.state[i],1):
                        s += self.ket(i)+' + '
                    else:
                        s += str(round(self.state[i][0],2))+self.ket(i)+' + '
        else:
            s = ''
            for i in range(self.size):
                if not np.allclose(self.state[i],0):
                    if np.allclose(self.state[i],1):
                        s += self.label[i]+' + '
                    else:
                        s += str(round(self.state[i],2))+self.label[i]+' + '
        return s[:-3]

    def __mul__(self,other):
        if isinstance(other,QubitArray):
            new = np.zeros(self.size*other.size)
            for i,q1 in enumerate(self.state):
                for j,q2 in enumerate(other.state):
                    new[i*other.size+j] = q1*q2
            return QubitArray(new)
        elif isinstance(other,gates.Gate):
            return np.dot(other.gate,self.state)

    def __add__(self,other):
        s = self.state + other.state
        s = normalize(s)
        return QubitArray(s)
    
    def __sub__(self,other):
        s = self.state - other.state
        s = normalize(s)
        return QubitArray(s)

    def __neg__(self):
        return QubitArray(-self.state)

    def inner(self,other):
        return np.sum(np.conjugate(self.state)*other.state)

    def outer(self,other):
        return gates.Gate(np.dot(self.state,np.conjugate(other.state.T)))

    def conjugate(self):
        return np.conjugate(self.state)

class One(QubitArray):
    def __init__(self,N=1):
        s = np.zeros((2**N,1))
        s[-1] = 1
        super().__init__(s)

class Zero(QubitArray):
    def __init__(self,N=1):
        s = np.zeros((2**N,1))
        s[0] = 1
        super().__init__(s)

def computational_basis(N):
    basis = []
    N = 2**N
    for n in range(N):
        s = np.zeros((N,1))
        s[n] = 1
        basis.append(QubitArray(s))
    return basis

class EvenlyDistributed(QubitArray):
    def __init__(self,N=1):
        N = 2**N
        s = np.ones((N,1))/np.sqrt(N)
        super().__init__(s)

class XPlus(QubitArray):
    sym = '|+x>'
    def __init__(self):
        super().__init__(np.array((1,1))/np.sqrt(2))

    def latex_label(self):
        return '$\\frac{(|0> + |1>)}{\\sqrt{2}}$'

class XMinus(QubitArray):
    sym = '|-x>'
    def __init__(self):
        super().__init__(np.array((1,-1))/np.sqrt(2))
    
    def latex_label(self):
        return '$\\frac{(|0> - |1>)}{\\sqrt{2}}$'

def x_basis():
    return XPlus(),XMinus()

def y_basis():
    return XPlus(),XMinus()

def z_basis():
    return Zero(),-One()

class PsiPlus(QubitArray):
    sym = '|+Psi>'
    def __init__(self):
        super().__init__(np.array((1,0,0,1))/np.sqrt(2))

    def latex_label(self):
        return '$\\frac{(|00> + |11>)}{\\sqrt{2}}$'

class PsiMinus(QubitArray):
    sym = '|-Psi>'
    def __init__(self):
        super().__init__(np.array((1,0,0,-1))/np.sqrt(2))

    def latex_label(self):
        return '$\\frac{(|00> - |11>)}{\\sqrt{2}}$'

class PhiPlus(QubitArray):
    sym = '|+Phi>'
    def __init__(self):
        super().__init__(np.array((0,1,1,0))/np.sqrt(2))

    def latex_label(self):
        return '$\\frac{(|10> + |01>)}{\\sqrt{2}}$'

class PhiMinus(QubitArray):
    sym = '|-Phi>'
    def __init__(self):
        super().__init__(np.array((0,1,-1,0))/np.sqrt(2))
    
    def latex_label(self):
        return '$\\frac{(|10> - |01>)}{\\sqrt{2}}$'

def bell_basis():
    return PsiPlus(),PsiMinus(),PhiPlus(),PhiMinus()

def extend_basis(basis,N_ancillary_qubits,add_to_front=False):
    if add_to_front:
        for n in range(N_ancillary_qubits):
            new_basis = []
            for base in basis:
                new_basis.append(Zero()*base)
                if n == 0:
                    new_basis[-1].sym = '0>'+base.sym
                else:
                    new_basis[-1].sym = '0'+base.sym
                new_basis.append(One()*base)
                if n == 0:
                    new_basis[-1].sym = '0'+base.sym
                else:
                    new_basis[-1].sym = '1'+base.sym
            basis = new_basis
        for base in new_basis:
            base.sym = '|'+base.sym
        return new_basis
    else:
        for n in range(N_ancillary_qubits):
            new_basis = []
            for base in basis:
                new_basis.append(base*Zero())
                if n == 0:
                    new_basis[-1].sym = base.sym+'|0'
                else:
                    new_basis[-1].sym = base.sym+'0'
                new_basis.append(base*One())
                if n == 0:
                    new_basis[-1].sym = base.sym+'|0'
                else:
                    new_basis[-1].sym = base.sym+'1'
            basis = new_basis
        for base in new_basis:
            base.sym += '>'
        return new_basis
