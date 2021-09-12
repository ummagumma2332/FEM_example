import numpy as np
import sympy as sym
from tqdm import tqdm


def getEigValEigFunc(KL_terms, l, b):

    a = l/2.
    x = np.linspace(-a, a, 40) # define the stochastic field for Karhunen Loeve method


    eigVal = np.zeros(KL_terms) 
    eigFunc = np.zeros((KL_terms, len(x)))

    Wsol = np.zeros(KL_terms)
    w = sym.symbols('w')
    f1 = 1/b - w*sym.tan(a*w) # function to be solved for n = odd
    f2 = sym.tan(a*w)/b + w   # function to be solved for n = even

    for i in range(KL_terms):
        for j in range(len(x)):
            if ((i+1) % 2) != 0:   # odd
                Wsol[i] = sym.nsolve(f1, w, (((i+1)-1)*np.pi/a, ((i+1)-1/2)*np.pi/a), solver='bisect', verify=False)
                eigFunc[i,j] = (a + np.sin(2*Wsol[i]*a)/2*Wsol[i])**(-0.5)*np.cos(Wsol[i]*x[j])
            else:                  # even
                Wsol[i] = sym.nsolve(f2, w, (((i+1)-1/2)*np.pi/a, (i+1)*np.pi/a), solver='bisect', verify=False)
                eigFunc[i,j] = (a - np.sin(2*Wsol[i]*a)/2*Wsol[i])**(-0.5)*np.sin(Wsol[i]*x[j])
            eigVal[i] = 2*b/(1+Wsol[i]**2*b**2)

    return eigVal, eigFunc


def run_KL(M):

    # M: number of realization

    KL_terms = 6   # terms to retain in the Karhunen-Loeve series expansion
    l = 4          # length of the stochastic field
    b = 3          # autocorrelation length

    # calculate eigenvalues and eigenfunctions
    eigVal, eigFunc = getEigValEigFunc(KL_terms, l, b)

    E_kl = np.zeros((M ,l*10))
    E0 = 1e5
    print("Generating realizations ...")
    for i in tqdm(range(M)):
        ksi = 2*np.random.rand(KL_terms) - 1 # random variable
        for j in range(l*10):
            for n in range(KL_terms):
                E_kl[i,j] += sym.sqrt(eigVal[n])*eigFunc[n,j]*0.1*ksi[n]
            E_kl[i,j] = E0*(E_kl[i,j] + 1)
    
    return E_kl

