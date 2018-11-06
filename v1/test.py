import matplotlib.pyplot as plt
import numpy as np
from gates import * 
from qubit import * 
from visualizations import *

N = 2
s = EvenlyDistributed(N)
s.real_measure()
print(s)
S = s.outer(s)
q = Zero(N=N)
Z = q.outer(q)
Uf = I(N) - S*2
Uh = H()
for n in range(N-1):
    Uh = Uh.tensor(H())
U0 = I(N) - Z*2
psi = Uh*q
Q = -Uh*U0*Uh*Uf

AnimatedHistogram(psi).plot()
for n in range(10*N):
    psi = Q*psi
    if n%3 == 0:
        AnimatedHistogram(psi).plot()

"""
def combine_basis(a,b):
    new = []
    for s in a:
        for _s in b:
            new.append(s*_s)
    return new

def create_basis(theta):
    arrs = [
        np.array([1,0,0,np.exp(1j*theta)])/np.sqrt(2),
        np.array([0,1,0,0]),
        np.array([0,0,1,0]),
        np.array([1,0,0,-np.exp(1j*theta)])/np.sqrt(2),
    ]
    return [QubitArray(arr) for arr in arrs]

W  = (Z()+X())/np.sqrt(2)
#B1 = [W*Zero(),W*One()]
B1 = x_basis() 
V  = (Z()-X())/np.sqrt(2)
B2 = [V*Zero(),V*One()]

A1 = x_basis()
A2 = y_basis()
B1 = x_basis()
B2 = y_basis()

B = [combine_basis(A1,B1),combine_basis(A1,B2),combine_basis(A2,B1),combine_basis(A2,B2)]
A = [extend_basis(A1,1),extend_basis(A2,1)]
B = [extend_basis(B1,1,add_to_front=True),extend_basis(B2,1,add_to_front=True)]
print(A)
print(B)
O = [X().tensor(X()),
     X().tensor(Y()),
     Y().tensor(X()),
     Y().tensor(Y())]

theta = np.pi/4
psi = QubitArray(np.array([1,0,0,np.exp(1j*theta)])/np.sqrt(2))
p = psi.inner(O[0]*psi)
print(p)
print(np.conjugate(p)*p)

R = np.array([1,-1,-1,1])
C = np.array([1,1,1,1])
T = np.linspace(0,2*np.pi,20)-np.pi/4
N = 100
for theta in T:
    basis = create_basis(theta) 
    psi = QubitArray(np.array([1,0,0,np.exp(1j*theta)])/np.sqrt(2))
    S = []
    i = 0
    for a in A:
        for b in B:
            E = 0
            _psi = O[i]*psi
            i += 1
            for n in range(N):
                m1 = _psi.measure_partial_by_basis((0,),(_a for _a in a))
                m2 = _psi.measure_partial_by_basis((1,),(_b for _b in b))
                E += 1 if m1 == m2 else -1
            S.append(E/N)
    S = sum(C*S)
    if np.abs(S) < 2:
        plt.plot(theta,S,'bo')
    else:
        plt.plot(theta,S,'ro')
plt.plot(T,2*(np.sin(T)+np.cos(T)),'g--')
plt.axhline(2,0,1)
plt.axhline(-2,0,1)
plt.show()
"""
