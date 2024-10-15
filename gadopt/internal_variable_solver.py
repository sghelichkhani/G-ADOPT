r"""This module provides a fine-tuned solver class for the energy conservation equation.
Users instantiate the `EnergySolver` class by providing relevant parameters and call
the `solve` method to request a solver update.

"""

from typing import Any, Optional

from firedrake import Constant, Function

from .approximations import BaseApproximation
from .time_stepper import RungeKuttaTimeIntegrator
from .viscoelastic_equation import InternalVariableEquation
from .utility import log_level, INFO, DEBUG

__all__ = [
    "iterative_solver_parameters",
    "direct_solver_parameters",
    "InternalVariableSolver",
]

iterative_solver_parameters: dict[str, Any] = {
    "mat_type": "aij",
    "snes_type": "ksponly",
    "ksp_type": "gmres",
    "ksp_rtol": 1e-5,
    "pc_type": "sor",
}
"""Default iterative solver parameters for solution of energy equation. Configured to use the GMRES Krylov scheme
   with Successive Over Relaxation (SOR) preconditioning. Note that default energy solver parameters
   can be augmented or adjusted by accessing the solver_parameter dictionary, for example:
   energy_solver.solver_parameters['ksp_converged_reason'] = None
   energy_solver.solver_parameters['ksp_rtol'] = 1e-4
   G-ADOPT defaults to iterative solvers in 3-D.
"""

direct_solver_parameters: dict[str, Any] = {
    "mat_type": "aij",
    "snes_type": "ksponly",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}
"""Default direct solver parameters for solution of energy equation. Configured to use LU factorisation,
   using the MUMPS library. G-ADOPT defaults to direct solvers in 2-D.
"""


class InternalVariableSolver:
    """Timestepper and solver for the energy equation. The temperature, T, is updated in place.

    Arguments:
      m:                 Firedrake function for internal variable
      u:                 Firedrake function for velocity
      approximation:     G-ADOPT base approximation describing the system of equations
      delta_t:           Simulation time step
      timestepper:       Runge-Kutta time integrator implementing an explicit or implicit numerical scheme
      bcs:               Dictionary of identifier-value pairs specifying boundary conditions
      solver_parameters: Solver parameters provided to PETSc

    """

    def __init__(
        self,
        m: Function,
        u: Function,
        approximation: BaseApproximation,
        delta_t: Constant,
        timestepper: RungeKuttaTimeIntegrator,
        solver_parameters: Optional[dict[str, Any]] = None,
    ):
        self.m = m
        self.Q = m.function_space()
        self.mesh = self.Q.mesh()
        self.delta_t = delta_t
        self.eq = InternalVariableEquation(self.Q, self.Q)

        d = approximation.deviatoric_strain(u)
        tau = approximation.maxwell_time
        self.fields = {
            'source': d/tau,
            'absorption_coefficient': 1/tau
        }

        if solver_parameters is None:
            if self.mesh.topological_dimension() == 2:
                self.solver_parameters = direct_solver_parameters.copy()
                if INFO >= log_level:
                    # not really "informative", but at least we get a 1-line message saying we've passed the energy solve
                    self.solver_parameters['ksp_converged_reason'] = None
                    self.solver_parameters['ksp_monitor'] = None
            else:
                self.solver_parameters = iterative_solver_parameters.copy()
                if DEBUG >= log_level:
                    self.solver_parameters['ksp_monitor'] = None
                elif INFO >= log_level:
                    self.solver_parameters['ksp_converged_reason'] = None
        else:
            self.solver_parameters = solver_parameters
        self.timestepper = timestepper
        self.m_old = Function(self.Q)
        # solver is setup only at the end, so users
        # can overwrite or augment default parameters specified above
        self._solver_setup = False

    def setup_solver(self):
        """Sets up timestepper and associated solver, using specified solver parameters"""
        self.ts = self.timestepper(self.eq, self.m, self.fields, self.delta_t,
                                   solution_old=self.m_old,
                                   solver_parameters=self.solver_parameters)
        self._solver_setup = True

    def solve(self, t=0, update_forcings=None):
        """Advances solver in time."""
        if not self._solver_setup:
            self.setup_solver()
        self.ts.advance(t, update_forcings)
