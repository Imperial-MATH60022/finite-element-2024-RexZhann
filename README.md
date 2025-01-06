## MATH60022 Finite Element: Analysis and Implementation

This repository is the implementation project for the course, where the main tasks included:

### Numerical Quadrature
Explore and extend Legendre-Gauss quadrature to 1D and 2D;\
Implement quadrature rules in Python;

### Constructing Finite Elements
Implement Lagrange element nodes and basis functions solvers;\
Implement finite elements;\
Tabulate basis functions and interpolate to finite element nodes;

### Meshes
Implement a mesh structure with correct global numbering and Jacobian;

### Function Spaces
Associate data with meshes using local/global numbering;\
Study cell-node maps and implement function spaces;

### Functions in Finite Element Spaces
Implement and interpolate functions in finite element spaces;\
Implement  integration in finite element spaces;

### Assembling and Solving Problems
Assemble the right-hand side and left-hand side matrices;\
Solve the Poisson problem;\
Solve the Helmholtz problem.

Link to the repository: https://github.com/Imperial-MATH60022/finite-element-2024-RexZhann.git\

All implementations done are stored in the fe_utils folder, where the two problems solved in the last part are stored in the fe_utils/solvers which can be run directly after installing the package.


A quick guide for installing the package: https://finite-element.github.io/implementation.html#obtaining-the-skeleton-code
