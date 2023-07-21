from firedrake_adjoint import *
from mpi4py import MPI
import ROL


class LinMoreOptimiser:
    def __init__(self, problem, parameters, callback=None):
        self.rol_solver = ROLSolver(problem, parameters, inner_product="L2")
        self.rol_parameters = ROL.ParameterList(parameters, "Parameters")

        try:
            self.rol_secant = ROL.InitBFGS(parameters["General"]["Secant"]["Maximum Storage"])
        except KeyError:
            # Use the default storage value
            self.rol_secant = ROL.InitBFGS()

        self.rol_algorithm = ROL.LinMoreAlgorithm(self.rol_parameters, self.rol_secant)

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
