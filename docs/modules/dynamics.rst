:mod:`nvalchemiops.dynamics`: Molecular Dynamics Integrators
=============================================================

.. automodule:: nvalchemiops.dynamics
    :no-members:
    :no-inherited-members:

High-Level Interface
--------------------

This module provides GPU-accelerated integrators and thermostats for molecular dynamics
simulations. All functions support automatic differentiation through PyTorch's autograd
system and offer three execution modes: single system, batched with ``batch_idx``, and
batched with ``atom_ptr``.

.. tip::
    Check out the :ref:`dynamics_userguide` page for usage examples and
    a conceptual overview of the available integrators and thermostats.

Integrators
-----------

Velocity Verlet
^^^^^^^^^^^^^^^

Time-reversible, symplectic integrator for NVE (microcanonical) ensemble.

.. autofunction:: nvalchemiops.dynamics.integrators.velocity_verlet.velocity_verlet_position_update
.. autofunction:: nvalchemiops.dynamics.integrators.velocity_verlet.velocity_verlet_velocity_finalize
.. autofunction:: nvalchemiops.dynamics.integrators.velocity_verlet.velocity_verlet_position_update_out
.. autofunction:: nvalchemiops.dynamics.integrators.velocity_verlet.velocity_verlet_velocity_finalize_out

Langevin Dynamics
^^^^^^^^^^^^^^^^^

BAOAB splitting scheme for NVT (canonical) ensemble with stochastic thermostat.

.. autofunction:: nvalchemiops.dynamics.integrators.langevin.langevin_baoab_half_step
.. autofunction:: nvalchemiops.dynamics.integrators.langevin.langevin_baoab_finalize
.. autofunction:: nvalchemiops.dynamics.integrators.langevin.langevin_baoab_half_step_out
.. autofunction:: nvalchemiops.dynamics.integrators.langevin.langevin_baoab_finalize_out

Nosé-Hoover Chain
^^^^^^^^^^^^^^^^^

Deterministic thermostat for NVT ensemble using Nosé-Hoover chains with Yoshida-Suzuki integration.

.. autofunction:: nvalchemiops.dynamics.integrators.nose_hoover.nhc_velocity_half_step
.. autofunction:: nvalchemiops.dynamics.integrators.nose_hoover.nhc_position_update
.. autofunction:: nvalchemiops.dynamics.integrators.nose_hoover.nhc_thermostat_chain_update
.. autofunction:: nvalchemiops.dynamics.integrators.nose_hoover.nhc_velocity_half_step_out
.. autofunction:: nvalchemiops.dynamics.integrators.nose_hoover.nhc_position_update_out
.. autofunction:: nvalchemiops.dynamics.integrators.nose_hoover.nhc_thermostat_chain_update_out
.. autofunction:: nvalchemiops.dynamics.integrators.nose_hoover.nhc_compute_masses
.. autofunction:: nvalchemiops.dynamics.integrators.nose_hoover.nhc_compute_chain_energy

Velocity Rescaling
^^^^^^^^^^^^^^^^^^

Simple velocity rescaling thermostat for quick equilibration (non-canonical).

.. autofunction:: nvalchemiops.dynamics.integrators.velocity_rescaling.velocity_rescale
.. autofunction:: nvalchemiops.dynamics.integrators.velocity_rescaling.velocity_rescale_out
.. autofunction:: nvalchemiops.dynamics.integrators.velocity_rescaling.compute_rescale_factor

Optimizers
----------

FIRE
^^^^

Fast Inertial Relaxation Engine for geometry optimization.

.. autofunction:: nvalchemiops.dynamics.optimizers.fire.fire_step

Thermostat Utilities
--------------------

Functions for temperature control and velocity initialization.

.. autofunction:: nvalchemiops.dynamics.utils.thermostat_utils.compute_kinetic_energy
.. autofunction:: nvalchemiops.dynamics.utils.thermostat_utils.compute_temperature
.. autofunction:: nvalchemiops.dynamics.utils.thermostat_utils.initialize_velocities
.. autofunction:: nvalchemiops.dynamics.utils.thermostat_utils.initialize_velocities_out
.. autofunction:: nvalchemiops.dynamics.utils.thermostat_utils.remove_com_motion
.. autofunction:: nvalchemiops.dynamics.utils.thermostat_utils.remove_com_motion_out
