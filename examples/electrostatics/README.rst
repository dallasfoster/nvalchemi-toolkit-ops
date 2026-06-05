Electrostatics
==============

Examples demonstrating GPU-accelerated computation of long-range electrostatic
interactions in periodic systems using Coulomb, Ewald summation, and Particle
Mesh Ewald (PME).

These examples show how to:

* Compute direct Coulomb interactions (damped and undamped)
* Use Ewald summation for periodic systems with automatic parameter estimation
* Apply Particle Mesh Ewald (PME) for O(N log N) scaling
* Work with neighbor list and neighbor matrix formats
* Perform batch evaluation for multiple systems
* Leverage autograd for computing forces and gradients
* Compute multipole Ewald and PME totals with charges, dipoles, and
  quadrupoles (l_max = 0 / 1 / 2), including stress tensors via the
  cell gradient and force-loss-style training via the second-order
  backward Warp kernel
* Extract atom-centered multipole features by projecting the periodic
  potential onto receiver GTOs
* Amortize the position-independent reciprocal-space state with the
  multipole SCF cache and reuse it across many step evaluations
