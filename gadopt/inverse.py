from firedrake import CheckpointFile
import firedrake.utils
from firedrake_adjoint import *
from mpi4py import MPI
from pyadjoint.optimization.optimization_solver import OptimizationSolver
import pyadjoint.optimization.rol_solver as pyadjoint_rol
import ROL


_vector_registry = []


class ROLSolver(pyadjoint_rol.ROLSolver):
    def __init__(self, problem, parameters, inner_product="L2", vector_class=pyadjoint_rol.ROLVector):
        OptimizationSolver.__init__(self, problem, parameters)
        self.rolobjective = ROLObjective(problem.reduced_functional)
        x = [p.tape_value() for p in self.problem.reduced_functional.controls]
        self.rolvector = vector_class(x, inner_product=inner_product)
        self.params_dict = parameters

        self.bounds = self.__get_bounds()
        self.constraints = self.__get_constraints()


class CheckpointedROLVector(pyadjoint_rol.ROL_Vector):
    def __init__(self, dat, checkpoint_dir, inner_product="L2"):
        super().__init__(dat, inner_product)

        self._checkpoint_dir = checkpoint_dir

    def clone(self):
        dat = []
        for x in self.dat:
            dat.append(x._ad_copy())
        res = CheckpointedROLVector(dat, inner_product=self.inner_product)
        res.scale(0.0)
        return res

    def save(self, checkpoint_path):
        with CheckpointFile(checkpoint_path, "w") as f:
            for i, func in enumerate(self.dat):
                f.save_function(func, name=f"dat_{i}")

    def load(self, mesh):
        with CheckpointFile(self.checkpoint_path, "r") as f:
            for i in range(len(self.dat)):
                self.dat[i] = f.load_function(mesh, name=f"dat_{i}")

    def __setstate__(self, state):
        """Set the state from the result of unpickling.

        This happens during the restoration of a checkpoint. self.dat needs to be
        separately set, then self.load() can be called.
        """

        # initialise C++ state
        super().__init__(state)
        self.checkpoint_path, self.inner_product = state

        _vector_registry.append(self)

    def __getstate__(self):
        """Return a state tuple suitable for pickling"""

        checkpoint_path = self._checkpoint_dir / f"vector_checkpoint_{firedrake.utils._new_uid()}.h5"
        self.save(checkpoint_path)

        return (checkpoint_path, self.inner_product)


class LinMoreOptimiser:
    def __init__(self, problem, parameters, checkpoint_dir=None):
        solver_kwargs = {}
        if checkpoint_dir is not None:
            self._checkpoint_dir = Path(checkpoint_dir)
            self._ensure_checkpoint_dir()

            self._mesh = problem.reduced_functional.controls[0].control.function_space().mesh()
            solver_kwargs["vector_class"] = CheckpointedROLVector

        self.rol_solver = ROLSolver(problem, parameters, inner_product="L2", **solver_kwargs)
        self.rol_parameters = ROL.ParameterList(parameters, "Parameters")

        try:
            self.rol_secant = ROL.InitBFGS(parameters["General"]["Secant"]["Maximum Storage"])
        except KeyError:
            # Use the default storage value
            self.rol_secant = ROL.InitBFGS()

        self.rol_algorithm = ROL.LinMoreAlgorithm(self.rol_parameters, self.rol_secant)

    def _ensure_checkpoint_dir(self):
        if MPI.COMM_WORLD.rank == 0:
            self._checkpoint_dir.mkdir(exist_ok=True)

        MPI.COMM_WORLD.Barrier()

    def checkpoint(self):
        """Checkpoint the current ROL state to disk."""

        ROL.serialise_secant(self.rol_secant, MPI.COMM_WORLD.rank, self._checkpoint_dir)
        ROL.serialise_algorithm(self.rol_algorithm, MPI.COMM_WORLD.rank, self._checkpoint_dir)

        with CheckpointFile(self._checkpoint_dir / "solution_checkpoint.h5", "w") as f:
            for i, func in enumerate(self.rol_solver.rolvector.dat):
                f.save_function(func, name=f"dat_{i}")

    def reload(self, iteration):
        ROL.load_secant(self.rol_secant, MPI.COMM_WORLD.rank, self._checkpoint_dir)
        ROL.load_algorithm(self.rol_algorithm, MPI.COMM_WORLD.rank, self._checkpoint_dir)

        self.rol_solver.rolvector.checkpoint_path = self._checkpoint_dir / "solution_checkpoint.h5"
        self.rol_solver.rolvector.load(self._mesh)

        # The various ROLVector objects can load all their metadata, but can't actually
        # restore from the Firedrake checkpoint. They register themselves, so we can access
        # them through a flat list.
        vec = self.rol_solver.rolvector.dat
        for v in _vector_registry:
            x = [p.copy(deepcopy=True) for p in vec]
            v.dat = x
            v.load(self._mesh)

        _vector_registry.clear()

    def run(self):
        self.rol_algorithm.run(
            self.rol_solver.rolvector,
            self.rol_solver.rolobjective,
            self.rol_solver.bounds,
        )

    def add_callback(self, callback):
        # XXX: this doesn't really handle chained callbacks
        class StatusTest(ROL.StatusTest):
            def check(self, status):
                callback()

                return super().check(status)

        # Don't chain with the default status test
        self.rol_algorithm.setStatusTest(StatusTest(self.rol_parameters), False)


minimisation_parameters = {
    "General": {
        "Print Verbosity": 1 if MPI.COMM_WORLD.rank == 0 else 0,
        "Output Level": 1 if MPI.COMM_WORLD.rank == 0 else 0,
        "Krylov": {
            "Iteration Limit": 10,
            "Absolute Tolerance": 1e-4,
            "Relative Tolerance": 1e-2,
        },
        "Secant": {
            "Type": "Limited-Memory BFGS",
            "Maximum Storage": 10,
            "Use as Hessian": True,
            "Barzilai-Borwein": 1,
        },
    },
    "Step": {
        "Type": "Trust Region",
        "Trust Region": {
            "Lin-More": {
                "Maximum Number of Minor Iterations": 10,
                "Sufficient Decrease Parameter": 1e-2,
                "Relative Tolerance Exponent": 1.0,
                "Cauchy Point": {
                    "Maximum Number of Reduction Steps": 10,
                    "Maximum Number of Expansion Steps": 10,
                    "Initial Step Size": 1.0,
                    "Normalize Initial Step Size": True,
                    "Reduction Rate": 0.1,
                    "Expansion Rate": 10.0,
                    "Decrease Tolerance": 1e-8,
                },
                "Projected Search": {
                    "Backtracking Rate": 0.5,
                    "Maximum Number of Steps": 20,
                },
            },
            "Subproblem Model": "Lin-More",
            "Initial Radius": 1.0,
            "Maximum Radius": 1e20,
            "Step Acceptance Threshold": 0.05,
            "Radius Shrinking Threshold": 0.05,
            "Radius Growing Threshold": 0.9,
            "Radius Shrinking Rate (Negative rho)": 0.0625,
            "Radius Shrinking Rate (Positive rho)": 0.25,
            "Radius Growing Rate": 10.0,
            "Sufficient Decrease Parameter": 1e-2,
            "Safeguard Size": 100,
        },
    },
    "Status Test": {
        "Gradient Tolerance": 0,
        "Iteration Limit": 100,
    },
}
