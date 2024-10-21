r"""This module provides a minimal solver for generic transport equations, which may
include advection, diffusion, sink, and source terms, and a fine-tuned solver class for
the energy conservation equation. Users instantiate the `GenericTransportSolver` or
`EnergySolver` classes by providing the appropriate documented parameters and call the
`solve` method to request a solver update.

"""

import abc
from numbers import Number
from typing import Any, Callable

from firedrake import *

from . import scalar_equation as scalar_eq
from .approximations import BaseApproximation
from .equations import Equation
from .time_stepper import RungeKuttaTimeIntegrator
from .utility import DEBUG, INFO, absv, is_continuous, log, log_level

__all__ = [
    "GenericTransportSolver",
    "EnergySolver",
    "direct_energy_solver_parameters",
    "iterative_energy_solver_parameters",
]

iterative_energy_solver_parameters: dict[str, Any] = {
    "mat_type": "aij",
    "snes_type": "ksponly",
    "ksp_type": "gmres",
    "ksp_rtol": 1e-5,
    "pc_type": "sor",
}
"""Default iterative solver parameters for solution of energy equation.

Configured to use the GMRES Krylov scheme with Successive Over Relaxation (SOR)
preconditioning. Note that default energy solver parameters can be augmented or adjusted
by accessing the solver_parameter dictionary, for example:
energy_solver.solver_parameters['ksp_converged_reason'] = None
energy_solver.solver_parameters['ksp_rtol'] = 1e-4

Note:
  G-ADOPT defaults to iterative solvers in 3-D.
"""

direct_energy_solver_parameters: dict[str, Any] = {
    "mat_type": "aij",
    "snes_type": "ksponly",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}
"""Default direct solver parameters for solution of energy equation.

Configured to use LU factorisation performed via the MUMPS library.

Note:
  G-ADOPT defaults to direct solvers in 2-D.
"""


class MetaPostInit(abc.ABCMeta):
    """Calls the user-defined __post_init__ method after __init__ returns."""

    def __call__(cls, *args, **kwargs):
        class_instance = super().__call__(*args, **kwargs)
        class_instance.__post_init__()

        return class_instance


class GenericTransportBase(abc.ABC, metaclass=MetaPostInit):
    """Base class for advancing a generic transport equation in time.

    All combinations of advection, diffusion, sink, and source terms are handled.

    Note: The solution field is updated in place.

    Arguments:
      solution:
        Firedrake function for the field of interest
      timestepper:
        Runge-Kutta time integrator employing an explicit or implicit numerical scheme
      delta_t:
        Simulation time step
      solution_old:
        Firedrake function holding the solution field at the previous time step
      bcs:
        Dictionary specifying boundary conditions (identifier, type, and value)
      solver_parameters:
        Dictionary of solver parameters or a string specifying a default configuration
        provided to PETSc

    """

    terms_mapping = {
        "advection": scalar_eq.advection_term,
        "diffusion": scalar_eq.diffusion_term,
        "sink": scalar_eq.sink_term,
        "source": scalar_eq.source_term,
    }

    def __init__(
        self,
        solution: Function,
        /,
        timestepper: RungeKuttaTimeIntegrator,
        delta_t: Constant,
        *,
        solution_old: Function | None = None,
        bcs: dict[int, dict[str, Number]] = {},
        solver_parameters: dict[str, str | Number] | str | None = None,
    ) -> None:
        self.solution = solution
        self.timestepper = timestepper
        self.delta_t = delta_t
        self.solution_old = solution_old or Function(solution)
        self.bcs = bcs
        self.solver_parameters = solver_parameters

        self.solution_space = solution.function_space()
        self.mesh = self.solution_space.mesh()
        self.test = TestFunction(self.solution_space)

        self.continuous_solution = is_continuous(self.solution)

        # Solver object is set up later to permit editing default solver options.
        self._solver_ready = False

    def __post_init__(self) -> None:
        self.set_boundary_conditions()
        self.set_equation()
        self.set_solver_options()

    def set_boundary_conditions(self) -> None:
        """Sets up boundary conditions."""
        self.strong_bcs = []
        self.weak_bcs = {}

        for bc_id, bc in self.bcs.items():
            weak_bc = {}

            for bc_type, value in bc.items():
                if bc_type == "T":
                    if self.continuous_solution:
                        strong_bc = DirichletBC(self.solution_space, value, bc_id)
                        self.strong_bcs.append(strong_bc)
                    else:
                        weak_bc["q"] = value
                else:
                    weak_bc[bc_type] = value

            self.weak_bcs[bc_id] = weak_bc

    def set_su_nubar(self, u: Function, su_diffusivity: float) -> ufl.algebra.Product:
        """Sets up the advection streamline-upwind scheme (Donea & Huerta, 2003).

        Columns of Jacobian J are the vectors that span the quad/hex and can be seen as
        unit vectors scaled with the dx/dy/dz in that direction (assuming physical
        coordinates x, y, z aligned with local coordinates).
        Thus u^T J is (dx * u , dy * v). Following (2.44c), Pe = u^T J / 2 kappa, and
        beta(Pe) is the xibar vector in (2.44a). Finally, we get the artificial
        diffusion nubar from (2.49).

        Donea, J., & Huerta, A. (2003).
        Finite element methods for flow problems.
        John Wiley & Sons.
        """
        if not self.continuous_solution:
            raise TypeError("SU advection requires a continuous function space.")

        log("Using SU advection")

        J = Function(TensorFunctionSpace(self.mesh, "DQ", 1), name="Jacobian")
        J.interpolate(Jacobian(self.mesh))
        # Calculate grid Peclet number. Note the use of a lower bound for diffusivity if
        # a pure advection scenario is considered.
        Pe = absv(dot(u, J)) / 2 / (su_diffusivity + 1e-12)
        beta_Pe = as_vector([1 / tanh(Pe_i + 1e-6) - 1 / (Pe_i + 1e-6) for Pe_i in Pe])
        nubar = dot(absv(dot(u, J)), beta_Pe) / 2  # Calculate SU artificial diffusion

        return nubar

    @abc.abstractmethod
    def set_equation(self):
        """Sets up the term contributions in the equation."""
        raise NotImplementedError

    def set_solver_options(self) -> None:
        """Sets PETSc solver parameters."""
        if isinstance(solver_preset := self.solver_parameters, dict):
            return

        if solver_preset is not None:
            match solver_preset:
                case "direct":
                    self.solver_parameters = direct_energy_solver_parameters.copy()
                case "iterative":
                    self.solver_parameters = iterative_energy_solver_parameters.copy()
                case _:
                    raise ValueError(f"Solver type '{solver_preset}' not implemented.")
        elif self.mesh.topological_dimension() == 2:
            self.solver_parameters = direct_energy_solver_parameters.copy()
        else:
            self.solver_parameters = iterative_energy_solver_parameters.copy()

        if DEBUG >= log_level:
            self.solver_parameters["ksp_monitor"] = None
        elif INFO >= log_level:
            self.solver_parameters["ksp_converged_reason"] = None

    def setup_solver(self) -> None:
        """Sets up the timestepper using specified parameters."""
        self.ts = self.timestepper(
            self.equation,
            self.solution,
            self.delta_t,
            solution_old=self.solution_old,
            solver_parameters=self.solver_parameters,
            strong_bcs=self.strong_bcs,
        )

        self._solver_ready = True

    def solver_callback(self) -> None:
        """Optional instructions to execute right after a solve."""
        pass

    def solve(self, t: Number = 0, update_forcings: Callable | None = None) -> None:
        """Advances solver in time."""
        if not self._solver_ready:
            self.setup_solver()

        self.ts.advance(t, update_forcings)

        self.solver_callback()


class GenericTransportSolver(GenericTransportBase):
    """Advances in time a generic transport equation.

    All combinations of advection, diffusion, sink, and source terms are handled.

    Note: The solution field is updated in place.

    Arguments:
      terms:
        List of equation terms (refer to terms_mapping)
      solution:
        Firedrake function for the field of interest
      timestepper:
        Runge-Kutta time integrator employing an explicit or implicit numerical scheme
      delta_t:
        Simulation time step
      solution_old:
        Firedrake function holding the solution field at the previous time step
      eq_attrs:
        Dictionary of terms arguments and their values
      bcs:
        Dictionary specifying boundary conditions (identifier, type, and value)
      solver_parameters:
        Dictionary of solver parameters or a string specifying a default configuration
        provided to PETSc
      su_diffusivity:
        Float activating the streamline-upwind stabilisation scheme and specifying the
        corresponding diffusivity

    """

    def __init__(
        self,
        terms: str | list[str],
        solution: Function,
        /,
        timestepper: RungeKuttaTimeIntegrator,
        delta_t: Constant,
        *,
        eq_attrs: dict[str, float] | None = None,
        su_diffusivity: float | None = None,
        **kwargs,
    ) -> None:
        super().__init__(solution, timestepper, delta_t, **kwargs)

        if isinstance(terms, str):
            terms = [terms]

        self.terms = terms
        self.eq_attrs = eq_attrs or {}
        self.su_diffusivity = su_diffusivity

    def set_equation(self) -> None:
        if self.su_diffusivity is not None:
            if (u := self.eq_attrs.get("u")) is None:
                raise ValueError(
                    "'u' must be included into `eq_attrs` if `su_diffusivity` is given."
                )

            self.eq_attrs["su_nubar"] = self.set_su_nubar(u, self.su_diffusivity)

        eq_terms = [self.terms_mapping[term] for term in self.terms]

        self.equation = Equation(
            self.test,
            self.solution_space,
            eq_terms,
            mass_term=scalar_eq.mass_term,
            eq_attrs=self.eq_attrs,
            bcs=self.weak_bcs,
        )


class EnergySolver(GenericTransportBase):
    """Advances in time the energy conservation equation.

    Note: The solution field is updated in place.

    Arguments:
      approximation:
        G-ADOPT approximation defining terms in the system of equations
      solution:
        Firedrake function for temperature
      u:
        Firedrake function for velocity
      timestepper:
        Runge-Kutta time integrator employing an explicit or implicit numerical scheme
      delta_t:
        Simulation time step
      solution_old:
        Firedrake function holding the solution field at the previous time step
      bcs:
        Dictionary specifying boundary conditions (identifier, type, and value)
      solver_parameters:
        Dictionary of solver parameters or a string specifying a default configuration
        provided to PETSc
      su_diffusivity:
        Float activating the streamline-upwind stabilisation scheme and specifying the
        corresponding diffusivity

    """

    def __init__(
        self,
        approximation: BaseApproximation,
        solution: Function,
        u: Function,
        /,
        timestepper: RungeKuttaTimeIntegrator,
        delta_t: Constant,
        *,
        su_diffusivity: float | None = None,
        **kwargs,
    ) -> None:
        super().__init__(solution, timestepper, delta_t, **kwargs)

        self.approximation = approximation
        self.u = u
        self.su_diffusivity = su_diffusivity

    def set_equation(self) -> None:
        rho_cp = self.approximation.rhocp()

        eq_attrs = {
            "advective_velocity_scaling": rho_cp,
            "diffusivity": self.approximation.kappa(),
            "reference_for_diffusion": self.approximation.Tbar,
            "sink_coeff": self.approximation.linearized_energy_sink(self.u),
            "source": self.approximation.energy_source(self.u),
            "u": self.u,
        }

        if self.su_diffusivity is not None:
            eq_attrs["su_nubar"] = self.set_su_nubar(self.u, self.su_diffusivity)

        eq_terms = self.terms_mapping.values()

        self.equation = Equation(
            self.test,
            self.solution_space,
            eq_terms,
            mass_term=lambda eq, trial: scalar_eq.mass_term(eq, rho_cp * trial),
            eq_attrs=eq_attrs,
            approximation=self.approximation,
            bcs=self.weak_bcs,
        )
