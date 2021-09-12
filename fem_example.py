import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import itertools
from statsmodels.distributions.empirical_distribution import ECDF
from tqdm import tqdm

import karhunen_loeve_solver

class Analysis:

    def __init__(self, lenX, lenY, n_elemX, n_elemY, load, poisson, thickness, n_simulations):

        self.lenX = lenX                     # length of the beam
        self.lenY = lenY                     # hight of the beam
        self.n_elemX = n_elemX               # number of elements along x-axis
        self.n_elemY = n_elemY               # number of elements along y-axis
        self.load = load                     # applied force
        self.poisson = poisson               # poisson ratio
        self.thickness = thickness           # thickness of beam
        self.n_simulations = n_simulations   # number of simulations for the monte carlo 
        self.element_nodes = []
        self.elem_dofs = []
        self.g_dofs = 2 * (n_elemX + 1) * (n_elemY + 1)  # global degrees of freedom

    def rectangular_mesh(self):

        j = 1
        i = 1
        i1 = 0
        counter = 0

        for j in range(self.n_elemY):
            for i in range(self.n_elemX):
                counter += 1
                if i == 0 and j == 0:
                    i1 = 1
                else:
                    i1 += 1
                i2 = i1 + 1
                i4 = i1 + self.n_elemX + 1
                i3 = i2 + self.n_elemX + 1
                self.element_nodes.append([i1, i2, i3, i4])
            i1 += 1
            i2 += 1

        self.element_nodes = np.array(self.element_nodes)

        return self.element_nodes


    def get_dofs(self):
        
        for j in range(len(self.element_nodes)):
            dofs = []
            for i in self.element_nodes[j]:
                dofs.append(i*2-1)
                dofs.append(i*2)
            self.elem_dofs.append(dofs)  
        self.elem_dofs = np.array(self.elem_dofs) - 1

        return self.elem_dofs

        
    def local_stiffness_matrix(self, young_mod):
    
        a = self.lenX / (2 * self.n_elemX)
        b = self.lenY / (2 * self.n_elemY)


        r = a/b
        rho = (1 - self.poisson)/2
        mu = (1 + self.poisson) * 3/2
        lamda = (1 - 3*self.poisson)/2
        k = np.zeros((8, 8))
        
        k[0, 0] = k[2, 2] = k[4, 4] = k[6, 6] = 4/r + 4*rho*r
        k[1, 1] = k[3, 3] = k[5, 5] = k[7, 7] = 4*r + 4*rho/r
        
        k[1, 0] = k[0, 1] = k[7, 2] = k[2, 7] = k[6, 3] = k[3, 6] = k[5, 4] = k[4, 5] = mu
        k[3, 0] = k[0, 3] = k[6, 1] = k[1, 6] = k[5, 2] = k[2, 5] = k[7, 4] = k[4, 7] = -lamda
        k[5, 0] = k[0, 5] = k[4, 1] = k[1, 4] = k[3, 2] = k[2, 3] = k[7, 6] = k[6, 7] = -mu
        k[7, 0] = k[0, 7] = k[1, 2] = k[2, 1] = k[4, 3] = k[3, 4] = k[6, 5] = k[5, 6] = lamda
        
        k[2, 0] = k[0, 2] = k[6, 4] = k[4, 6] = -4/r + 2*rho*r
        k[4, 0] = k[0, 4] = k[6, 2] = k[2, 6] = -2/r - 2*rho*r
        k[6, 0] = k[0, 6] = k[4, 2] = k[2, 4] = 2/r - 4*rho*r
        k[3, 1] = k[1, 3] = k[7, 5] = k[5, 7] = 2*r - 4*rho/r
        k[5, 1] = k[1, 5] = k[7, 3] = k[3, 7] = -2*r - 2*rho/r
        k[7, 1] = k[1, 7] = k[5, 3] = k[3, 5] = -4*r + 2*rho/r

        self.k_element_local = young_mod * self.thickness / (12 * (1 - self.poisson**2)) * k

        return self.k_element_local
         

    def get_element_global_stiffness(self, element_dof):

        self.k_element_glob = np.zeros((self.g_dofs, self.g_dofs))
        c1 = 0
        for i in element_dof:
            c2 = 0
            for j in element_dof:
                self.k_element_glob[i, j] = self.k_element_local[c1, c2]
                c2 += 1
            c1 += 1

        return self.k_element_glob

    def get_global_stiffness(self):

        self.globalStiffness = np.zeros((self.g_dofs, self.g_dofs))
        for el in range(len(self.elem_dofs)):
            elementGlobalStiffness = self.get_element_global_stiffness(self.elem_dofs[el])
            self.globalStiffness += elementGlobalStiffness

        return self.globalStiffness

    def set_boundary_conditions(self):
        self.constrained_dofs = np.unique(self.elem_dofs[::self.n_elemX][:, [0, 1, -2, -1]])
        return self.constrained_dofs

    def set_force_vector(self):
        # Define the external force vector
        self.forces = np.zeros(self.g_dofs)
        self.forces[-1] = -self.load      # the load is applied to the last dof - direction to negative y
        return self.forces

    def get_displacements(self): 
        # For each constrained dof i, set the elements of i-row & i-column of the global stiffness matrix to zero
        # Assign a very large positive number to the i-diagonal element 
        for i in self.constrained_dofs:  
            self.globalStiffness[:, i] = 0
            self.globalStiffness[i, :] = 0
            self.globalStiffness[i, i] = 1e10

        # Compute the displacement vector
        self.displacements = np.dot(np.linalg.inv(self.globalStiffness), self.forces)
        return self.displacements


run_FEM = Analysis(n_elemX=40, n_elemY=10, lenX=4, lenY=1, load=10, poisson=0.3, thickness=0.2, n_simulations=10)

run_FEM.rectangular_mesh()
run_FEM.get_dofs()
run_FEM.local_stiffness_matrix(young_mod=1e5)
run_FEM.get_global_stiffness()
run_FEM.set_boundary_conditions()
run_FEM.set_force_vector()

print(run_FEM.get_displacements()[81]) # the 81-st dof corresponds to the vertical displacement of right bottom corner of the structure

