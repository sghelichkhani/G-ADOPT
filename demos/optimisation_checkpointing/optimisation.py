from firedrake import Function, assemble, inner, dx, conditional
import warnings
warnings.filterwarnings("ignore")


class OptimisationFunction(object):
    def __init__(self, cntrl):
        self.f = Function(cntrl.control.function_space(), name="control")
        self.f.assign(cntrl.control)
        self.updated = True

    def sum(self, v):
        self.f.interpolate(self.f + v)

    def inner(self, v):
        return assemble(inner(self.f, v) * dx(self.f.ufl_domain()))

    def clone(self):
        return Function(self.f.function_space(), name=self.f.name()).assign(self.f)


class L_BFGS_BOptimizer:
    def __init__(self, rf, x0, bounds, m=10, tol=1e-6, max_iter=1000):
        """
        Initialize L-BFGS-B Optimizer.

        Parameters:
        rf: reduced functional
        x0: initial point
        bounds: list of tuple bounds for each variable
        m: memory size for storing past states
        tol: tolerance for stopping criterion
        max_iter: maximum number of iterations
        """
        # TODO: make sure rf is a reduced functional and x0 is control

        self.f = rf.__call__
        self.derivative = rf.derivative
        self.x0 = x0.clone()
        self.bounds = bounds
        self.m = m
        self.tol = tol
        self.max_iter = max_iter
        self.x = None  # Will hold the optimized value
        self.history_s = []  # History of s = x_{k+1} - x_k
        self.history_y = []  # History of y = derivative(x_{k+1}) - derivative(x_k)
        self.iteration = 0

    def bounded_line_search(self, x, p):
        """
        Line search with bounds.
        """
        alpha = 1.0
        c1 = 1e-4
        x_new = Function(x.function_space())
        x_new.interpolate(x + alpha * p)
        # Apply bounds
        for i, (l, u) in enumerate(self.bounds):
            x_new.interpolate(conditional(x_new > u, u, conditional(x_new < l, l, x_new)))

        fval = self.f(x)
        derivative_x = self.derivative()
        fval_new = self.f(x_new)
        while fval_new > fval + c1 * alpha * assemble(inner(derivative_x, p) * dx):
            alpha *= 0.75
            x_new.interpolate(x + alpha * p)
            # Apply bounds
            for i, (l, u) in enumerate(self.bounds):
                x_new.interpolate(conditional(x_new > u, u, conditional(x_new < l, l, x_new)))
            fval_new = self.f(x_new)
            print(f"alpha = {alpha}, fval_new= {fval_new}, fval={fval}")
        return alpha

    def two_loop_recursion(self, q, m):
        """
        Two-loop recursion to compute the L-BFGS approximation to the inverse Hessian-vector product.
        """
        alpha = []
        for i in range(m - 1, -1, -1):
            alpha_i = (
                assemble(inner(self.history_s[i], q) * dx) /
                assemble(inner(self.history_y[i], self.history_s[i]) * dx)
            )
            q.interpolate(q - alpha_i * self.history_y[i])
            alpha.append(alpha_i)

        # Scaling factor for initial Hessian approximation
        if self.history_s and self.history_y:
            scaling = (
                assemble(inner(self.history_y[-1], self.history_s[-1]) * dx) /
                assemble(inner(self.history_y[-1], self.history_y[-1]) * dx))
        else:
            scaling = 1.0
        z = Function(q.function_space())
        z.interpolate(scaling * q)

        for i in range(m):
            beta = (
                assemble(inner(self.history_y[i], z) * dx) /
                assemble(inner(self.history_y[i], self.history_s[i]) * dx)
            )
            z.interpolate((z + self.history_s[i] * (alpha[m - i - 1] - beta)))
        z.interpolate(-1 * z)
        return z

    def optimize(self):
        """
        Perform L-BFGS-B optimization.
        """
        x = self.x0

        while self.iteration < self.max_iter:
            f_val = self.f(x)
            g = self.derivative()
            print(f"Iteration: {self.iteration}, Val = {f_val}")

            # Check stopping criterion
            if assemble(g**2 * dx) < self.tol:
                print(f"Convergence reached after {self.iteration} iterations.")
                self.x = x
                return x
            # Compute direction p using two-loop recursion
            p = self.two_loop_recursion(g, min(self.m, len(self.history_s)))

            # Line search to find step size
            alpha = self.bounded_line_search(x, p)

            # Update point
            s = Function(p.function_space())
            s.interpolate(alpha * p)

            x_new = Function(x.function_space())
            x_new.interpolate(x + s)

            for i, (l, u) in enumerate(self.bounds):
                x_new.interpolate(conditional(x_new > u, u, conditional(x_new < l, l, x_new)))

            self.f(x_new)
            # Compute change in gradient
            y = Function(x_new.function_space())
            y.interpolate(self.derivative() - g)

            # Update history
            if len(self.history_s) == self.m:
                self.history_s.pop(0)
                self.history_y.pop(0)
            self.history_s.append(s)
            self.history_y.append(y)

            x.interpolate(x_new)  # Move to the new point
            self.iteration += 1  # Increment iteration counter

        print("Maximum number of iterations reached.")
        self.x = x
        return x
