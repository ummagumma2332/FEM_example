This is an application of the Finite Elements Method.
In this example I make use of quadrilateral finite elements in order to find the displacement field of a rectangular structural beam.
The size of the structure is reconfigurable as well as the density of the meshing, the loading conditions (magnitude and dofs to receive loads) 
and the boundary conditions.

This code solves the deterministic system given a constant Young's modulus (10^5 kN/m^2) and constant load (10 kN). The stochastic problem it is also addressed,
assuming that the Young's modulus can be described by a stochastic field along the length of the beam, and the load follows a normal distribution N~(10, 2).
To generate realizations of the stochastic field we use the Karhunen-Loeve series expansion.

Feel free to run and experiment with the code or even better suggest improvements.![image](https://user-images.githubusercontent.com/63021871/132966811-fd5f27b5-abd1-41df-9c39-f13c8858ba32.png)

