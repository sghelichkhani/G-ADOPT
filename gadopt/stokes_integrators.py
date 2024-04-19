from typing import Optional

import firedrake as fd

from .approximations import BaseApproximation
from .momentum_equation import StokesEquations
from .utility import DEBUG, INFO, depends_on, ensure_constant, log_level, upward_normal, InteriorBC

iterative_stokes_solver_parameters = {
    "mat_type": "matfree",
    "ksp_type": "preonly",
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_type": "full",
    "fieldsplit_0": {
        "ksp_type": "cg",
        "ksp_rtol": 1e-5,
        "ksp_max_it": 1000,
        "pc_type": "python",
        "pc_python_type": "gadopt.SPDAssembledPC",
        "assembled_pc_type": "gamg",
        "assembled_mg_levels_pc_type": "sor",
        "assembled_pc_gamg_threshold": 0.01,
        "assembled_pc_gamg_square_graph": 100,
        "assembled_pc_gamg_coarse_eq_limit": 1000,
        "assembled_pc_gamg_mis_k_minimum_degree_ordering": True,
    },
    "fieldsplit_1": {
        "ksp_type": "fgmres",
        "ksp_rtol": 1e-4,
        "ksp_max_it": 200,
        "pc_type": "python",
        "pc_python_type": "firedrake.MassInvPC",
        "Mp_pc_type": "ksp",
        "Mp_ksp_ksp_rtol": 1e-5,
        "Mp_ksp_ksp_type": "cg",
        "Mp_ksp_pc_type": "sor",
    }
}
"""Default solver parameters for iterative solvers"""

direct_stokes_solver_parameters = {
    "mat_type": "aij",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}
"""Default solver parameters for direct solvers"""

newton_stokes_solver_parameters = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "l2",
    "snes_max_it": 100,
    "snes_atol": 1e-10,
    "snes_rtol": 1e-5,
}
"""Default solver parameters for non-linear systems"""


def create_stokes_nullspace(
    Z: fd.functionspaceimpl.WithGeometry,
    closed: bool = True,
    rotational: bool = False,
    translations: Optional[list[int]] = None
) -> fd.nullspace.MixedVectorSpaceBasis:
    """Create a null space for the mixed Stokes system.

    Arguments:
      Z: Firedrake mixed function space associated with the Stokes system
      closed: Whether to include a constant pressure null space
      rotational: Whether to include all rotational modes
      translations: List of translations to include

    Returns:
      A Firedrake mixed vector space basis incorporating the null space components

    """
    X = fd.SpatialCoordinate(Z.mesh())
    dim = len(X)
    V, W = Z.subfunctions

    if rotational:
        if dim == 2:
            rotV = fd.Function(V).interpolate(fd.as_vector((-X[1], X[0])))
            basis = [rotV]
        elif dim == 3:
            x_rotV = fd.Function(V).interpolate(fd.as_vector((0, -X[2], X[1])))
            y_rotV = fd.Function(V).interpolate(fd.as_vector((X[2], 0, -X[0])))
            z_rotV = fd.Function(V).interpolate(fd.as_vector((-X[1], X[0], 0)))
            basis = [x_rotV, y_rotV, z_rotV]
        else:
            raise ValueError("Unknown dimension")
    else:
        basis = []

    if translations:
        for tdim in translations:
            vec = [0] * dim
            vec[tdim] = 1
            basis.append(fd.Function(V).interpolate(fd.as_vector(vec)))

    if basis:
        V_nullspace = fd.VectorSpaceBasis(basis, comm=Z.mesh().comm)
        V_nullspace.orthonormalize()
    else:
        V_nullspace = V

    if closed:
        p_nullspace = fd.VectorSpaceBasis(constant=True, comm=Z.mesh().comm)
    else:
        p_nullspace = W

    return fd.MixedVectorSpaceBasis(Z, [V_nullspace, p_nullspace])


class StokesSolver:
    """Solves the Stokes system.

    Arguments:
      z: Firedrake function representing the mixed Stokes system
      T: Firedrake function representing the temperature
      approximation: Approximation describing the system of equations
      bcs: Dictionary of identifier-value pairs specifying boundary conditions
      mu: Firedrake function representing dynamic viscosity
      quad_degree: Quadrature degree. Default value is `2p + 1`, where
                     p is the polynomial degree of the trial space
      cartesian: Whether to use Cartesian coordinates
      solver_parameters: Dictionary of solver parameters provided to PETSc
      J: Firedrake function representing the Jacobian of the system
      constant_jacobian: Whether the Jacobian of the system is constant

    """
    name = 'Stokes'

    def __init__(
        self,
        z: fd.Function,
        T: fd.Function,
        approximation: BaseApproximation,
        bcs: Optional[dict[int, dict[str, int | float]]] = None,
        mu: fd.Function | int | float = 1,
        quad_degree: int = 6,
        cartesian: bool = True,
        solver_parameters: Optional[dict[str, str | float]] = None,
        J: Optional[fd.Function] = None,
        constant_jacobian: bool = False,
        **kwargs
    ):
        self.Z = z.function_space()
        self.mesh = self.Z.mesh()
        self.test = fd.TestFunctions(self.Z)
        self.equations = StokesEquations(self.Z, self.Z, quad_degree=quad_degree,
                                         compressible=approximation.compressible)
        self.solution = z
        self.approximation = approximation
        self.mu = ensure_constant(mu)
        self.solver_parameters = solver_parameters
        self.J = J
        self.constant_jacobian = constant_jacobian
        self.linear = not depends_on(self.mu, self.solution)

        self.solver_kwargs = kwargs
        u, p = fd.split(self.solution)
        self.k = upward_normal(self.Z.mesh(), cartesian)
        self.fields = {
            'velocity': u,
            'pressure': p,
            'viscosity': self.mu,
            'interior_penalty': fd.Constant(2.0),  # allows for some wiggle room in imposition of weak BCs
                                                   # 6.25 matches C_ip=100. in "old" code for Q2Q1 in 2d.
            'source': self.approximation.buoyancy(p, T) * self.k,
            'rho_continuity': self.approximation.rho_continuity(),
        }

        self.weak_bcs = {}
        self.strong_bcs = []
        for id, bc in bcs.items():
            weak_bc = {}
            for bc_type, value in bc.items():
                if bc_type == 'u':
                    self.strong_bcs.append(fd.DirichletBC(self.Z.sub(0), value, id))
                elif bc_type == 'ux':
                    self.strong_bcs.append(fd.DirichletBC(self.Z.sub(0).sub(0), value, id))
                elif bc_type == 'uy':
                    self.strong_bcs.append(fd.DirichletBC(self.Z.sub(0).sub(1), value, id))
                elif bc_type == 'uz':
                    self.strong_bcs.append(fd.DirichletBC(self.Z.sub(0).sub(2), value, id))
                else:
                    weak_bc[bc_type] = value
            self.weak_bcs[id] = weak_bc

        self.F = 0
        for test, eq, u in zip(self.test, self.equations, fd.split(self.solution)):
            self.F -= eq.residual(test, u, u, self.fields, bcs=self.weak_bcs)

        if self.solver_parameters is None:
            if self.linear:
                self.solver_parameters = {"snes_type": "ksponly"}
            else:
                self.solver_parameters = newton_stokes_solver_parameters.copy()
            if INFO >= log_level:
                self.solver_parameters['snes_monitor'] = None

            if self.mesh.topological_dimension() == 2 and cartesian:
                self.solver_parameters.update(direct_stokes_solver_parameters)
            else:
                self.solver_parameters.update(iterative_stokes_solver_parameters)
                if DEBUG >= log_level:
                    self.solver_parameters['fieldsplit_0']['ksp_converged_reason'] = None
                    self.solver_parameters['fieldsplit_1']['ksp_monitor'] = None
                elif INFO >= log_level:
                    self.solver_parameters['fieldsplit_1']['ksp_converged_reason'] = None
        # solver is setup only last minute
        # so people can overwrite parameters we've setup here
        self._solver_setup = False

    def setup_solver(self):
        """Sets up the solver."""
        if self.constant_jacobian:
            z_tri = fd.TrialFunction(self.Z)
            F_stokes_lin = fd.replace(self.F, {self.solution: z_tri})
            a, L = fd.lhs(F_stokes_lin), fd.rhs(F_stokes_lin)
            self.problem = fd.LinearVariationalProblem(a, L, self.solution,
                                                       bcs=self.strong_bcs,
                                                       constant_jacobian=True)
            self.solver = fd.LinearVariationalSolver(self.problem,
                                                     solver_parameters=self.solver_parameters,
                                                     options_prefix=self.name,
                                                     appctx={"mu": self.mu},
                                                     **self.solver_kwargs)
        else:
            self.problem = fd.NonlinearVariationalProblem(self.F, self.solution,
                                                          bcs=self.strong_bcs, J=self.J)
            self.solver = fd.NonlinearVariationalSolver(self.problem,
                                                        solver_parameters=self.solver_parameters,
                                                        options_prefix=self.name,
                                                        appctx={"mu": self.mu},
                                                        **self.solver_kwargs)
        self._solver_setup = True

    def solve(self):
        """Solves the system."""
        if not self._solver_setup:
            self.setup_solver()
        self.solver.solve()

    def deviatoric_normal_stress(
            self,
            force: fd.Function,
            subdomain_id: int | str):
        """Compute deviatoric normal stress on surface(s) identified by subdomain_ids

        Args:
            force: Firedrake Function to which the deviatoric
                   normal stress field should beinterpolate into
            subdomain_id: list of markers for the surfaces this field should apply to

        Returns:
            force
        """

        # pressure and velocity together with viscosity are needed
        u, p, *epsilon = self.solution.subfunctions
        mu = self.mu

        mesh = force.function_space().mesh()
        n = fd.FacetNormal(mesh)

        # function space of the solution
        Q = force.function_space()
        R = fd.FunctionSpace(mesh, "R", 0)
        W = Q * R

        # Test and Trial Functions
        (phi, psi) = fd.TestFunction(W)
        (v, lambda_) = fd.TrialFunction(W)

        # Stress, compressible formulation has an additional term
        stress = -p * fd.Identity(2) + mu * 2 * fd.sym(fd.grad(u))
        compressible = self.approximation.compressible
        if compressible:
            stress -= 2/3 * mu * self.Identity(self.dim) * fd.div(u)

        # Surface integral for extruded mesh is different
        # Are we dealing with extruded mesh
        extruded_mesh = mesh.extruded

        # Building matrix
        if extruded_mesh and subdomain_id in ["top", "bottom"]:
            ds = {"top": fd.ds_t, "bottom": fd.ds_b}.get(id)
        else:
            ds = fd.ds(id)
        a = phi * v * ds + lambda_ * phi * ds + psi * v * ds
        L = - phi * fd.dot(fd.dot(stress, n), n) * ds

        w = fd.Function(W)
        # Solve a linear system
        fd.solve(
            a == L,
            w,
            bcs=[InteriorBC(W.sub(0), 0., subdomain_id)],
        )
        v, lambda_ = w.subfunctions
        force.assign(v)
        return force
