from typing import Optional

from firedrake import (
    Constant,
    DirichletBC,
    Function,
    Jacobian,
    TensorFunctionSpace,
    dot,
)

from .approximations import BaseApproximation
from .scalar_equation import EnergyEquation
from .time_stepper import RungeKuttaTimeIntegrator
from .utility import (
    DEBUG,
    INFO,
    absv,
    ensure_constant,
    is_continuous,
    log,
    log_level,
    su_nubar,
)

__all__ = [
    "iterative_energy_solver_parameters",
    "direct_energy_solver_parameters",
    "EnergySolver",
]

iterative_energy_solver_parameters = {
    "mat_type": "aij",
    "snes_type": "ksponly",
    "ksp_type": "gmres",
    "ksp_rtol": 1e-5,
    "pc_type": "sor",
}

direct_energy_solver_parameters = {
    "mat_type": "aij",
    "snes_type": "ksponly",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}


class EnergySolver:
    """Time-stepper solver for the energy equation."""

    def __init__(
        self,
        T: Function,
        u: Function,
        approximation: BaseApproximation,
        delta_t: Constant,
        timestepper: RungeKuttaTimeIntegrator,
        bcs: Optional[dict[int, dict[str, int | float]]] = None,
        solver_parameters: Optional[dict[str, str | float]] = None,
        su_advection: bool = False,
    ):
        """Initialises the solver instance from the simulation's state.

        Args:
          T:
            Firedrake function for the temperature.
          u:
            Firedrake function for the velocity.
          approximation:
            G-ADOPT base approximation describing the system of equations.
          delta_t:
            Firedrake constant representing the simulation's time step.
          bcs:
            Dictionary of identifier-value pairs specifying boundary conditions.
          solver_parameters:
            Dictionary of solver parameters provided to PETSc.
          su_advection:
            Boolean signaling the use of the streamline-upwind scheme.
        """
        self.T = T
        self.Q = T.function_space()
        self.mesh = self.Q.mesh()
        self.delta_t = delta_t
        rhocp = approximation.rhocp()
        self.eq = EnergyEquation(self.Q, self.Q, rhocp=rhocp)
        self.fields = {
            'diffusivity': ensure_constant(approximation.kappa()),
            'reference_for_diffusion': approximation.Tbar,
            'source': approximation.energy_source(u),
            'velocity': u,
            'advective_velocity_scaling': rhocp
        }
        sink = approximation.linearized_energy_sink(u)
        if sink:
            self.fields['absorption_coefficient'] = sink

        if su_advection:
            log("Using SU advection")
            # SU(PG) ala Donea & Huerta:
            # Columns of Jacobian J are the vectors that span the quad/hex
            # which can be seen as unit-vectors scaled with the dx/dy/dz in that direction (assuming physical coordinates x,y,z aligned with local coordinates)
            # thus u^T J is (dx * u , dy * v)
            # and following (2.44c) Pe = u^T J / 2 kappa
            # beta(Pe) is the xibar vector in (2.44a)
            # then we get artifical viscosity nubar from (2.49)

            J = Function(TensorFunctionSpace(self.mesh, 'DQ', 1), name='Jacobian').interpolate(Jacobian(self.mesh))
            kappa = self.fields['diffusivity'] + 1e-12  # Set lower bound for diffusivity in case zero diffusivity specified for pure advection.
            vel = self.fields['velocity']
            Pe = absv(dot(vel, J)) / (2*kappa)  # Calculate grid peclet number
            nubar = su_nubar(vel, J, Pe)  # Calculate SU artifical diffusion

            self.fields['su_nubar'] = nubar

        if solver_parameters is None:
            if self.mesh.topological_dimension() == 2:
                self.solver_parameters = direct_energy_solver_parameters.copy()
                if INFO >= log_level:
                    # not really "informative", but at least we get a 1-line message saying we've passed the energy solve
                    self.solver_parameters['ksp_converged_reason'] = None
            else:
                self.solver_parameters = iterative_energy_solver_parameters.copy()
                if DEBUG >= log_level:
                    self.solver_parameters['ksp_monitor'] = None
                elif INFO >= log_level:
                    self.solver_parameters['ksp_converged_reason'] = None
        else:
            self.solver_parameters = solver_parameters
        apply_strongly = is_continuous(T)
        self.strong_bcs = []
        self.weak_bcs = {}
        bcs = bcs or {}
        for id, bc in bcs.items():
            weak_bc = {}
            for type, value in bc.items():
                if type == 'T':
                    if apply_strongly:
                        self.strong_bcs.append(DirichletBC(self.Q, value, id))
                    else:
                        weak_bc['q'] = value
                else:
                    weak_bc[type] = value
            self.weak_bcs[id] = weak_bc

        self.timestepper = timestepper
        self.T_old = Function(self.Q)
        # solver is setup only last minute
        # so people can overwrite parameters we've setup here
        self._solver_setup = False

    def setup_solver(self):
        """Setup timestepper and associated solver."""
        self.ts = self.timestepper(self.eq, self.T, self.fields, self.delta_t,
                                   bnd_conditions=self.weak_bcs, solution_old=self.T_old,
                                   strong_bcs=self.strong_bcs,
                                   solver_parameters=self.solver_parameters)
        self._solver_setup = True

    def solve(self):
        """Advance solver."""
        if not self._solver_setup:
            self.setup_solver()
        t = 0  # not used atm
        self.ts.advance(t)
