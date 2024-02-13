from abc import ABC, abstractmethod
from typing import Optional

import firedrake

from .utility import CombinedSurfaceMeasure


class BaseEquation:
    """Produces the UFL for the registered terms constituting an equation.

    Attributes:
      terms:
        List of equation terms defined through inheritance from BaseTerm.
    """
    terms = []

    def __init__(
        self,
        test_space: firedrake.functionspaceimpl.WithGeometry,
        trial_space: firedrake.functionspaceimpl.WithGeometry,
        quad_degree: Optional[int] = None,
        **kwargs
    ):
        """Initialises the equation instance given function spaces.

        Test and trial spaces are only used to determine the employed discretisation
        (i.e. UFL elements); test and trial functions are provided separately in
        residual.

        Keyword arguments are passed on to each term of the equation.

        Args:
          test_space:
            Firedrake function space of the test function.
          trial_space:
            Firedrake function space of the rial function.
          quad_degree:
            Integer representing the quadrature degree. Default value is `2p + 1`, with
            p the polynomial degree of the trial space.
        """
        self.test_space = test_space
        self.trial_space = trial_space
        self.mesh = trial_space.mesh()

        p = trial_space.ufl_element().degree()
        if isinstance(p, int):  # isotropic element
            if quad_degree is None:
                quad_degree = 2*p + 1
        else:  # tensorproduct element
            p_h, p_v = p
            if quad_degree is None:
                quad_degree = 2*max(p_h, p_v) + 1

        if trial_space.extruded:
            # Create surface measures that treat the bottom and top boundaries similarly
            # to lateral boundaries. This way, integration using the ds and dS measures
            # occurs over both horizontal and vertical boundaries, and we can also use
            # "bottom" and "top" as surface identifiers, for example, ds("top").
            self.ds = CombinedSurfaceMeasure(self.mesh, quad_degree)
            self.dS = (
                firedrake.dS_v(domain=self.mesh, degree=quad_degree)
                + firedrake.dS_h(domain=self.mesh, degree=quad_degree)
            )
        else:
            self.ds = firedrake.ds(domain=self.mesh, degree=quad_degree)
            self.dS = firedrake.dS(domain=self.mesh, degree=quad_degree)

        self.dx = firedrake.dx(domain=self.mesh, degree=quad_degree)

        # self._terms stores the actual instances of the BaseTerm-classes in self.terms
        self._terms = []
        for TermClass in self.terms:
            self._terms.append(TermClass(test_space, trial_space, self.dx, self.ds, self.dS, **kwargs))

    def mass_term(
        self,
        test: firedrake.ufl_expr.Argument,
        trial: firedrake.ufl_expr.Argument | firedrake.Function,
    ) -> firedrake.ufl.core.expr.Expr:
        """UFL expression for the typical mass term used in the time discretisation.

        Args:
          test:
            Firedrake test function.
          trial:
            Firedrake trial function.

        Returns:
          The UFL expression associated with the mass term of the equation.
        """
        return firedrake.inner(test, trial) * self.dx

    def residual(
        self,
        test: firedrake.ufl_expr.Argument,
        trial: firedrake.ufl_expr.Argument | firedrake.Function,
        trial_lagged: Optional[firedrake.ufl_expr.Argument | firedrake.Function] = None,
        fields: Optional[dict[str, firedrake.Constant | firedrake.Function]] = None,
        bcs: Optional[dict[int, dict[str, int | float]]] = None,
    ) -> firedrake.ufl.core.expr.Expr:
        """UFL expression for the residual term expressed as a sum of all terms.

        Args:
          test:
            Firedrake test function.
          trial:
            Firedrake trial function.
          trial_lagged:
            Firedrake trial function from the previous time step.
          fields:
            Dictionary of physical fields from the simulation's state.
          bcs:
            Dictionary of identifier-value pairs specifying boundary conditions.

        Returns:
          The UFL expression associated with all equation terms except the mass term.
        """
        if trial_lagged is None:
            trial_lagged = trial
        if fields is None:
            fields = {}
        if bcs is None:
            bcs = {}

        F = 0
        for term in self._terms:
            F += term.residual(test, trial, trial_lagged, fields, bcs)

        return F


class BaseTerm(ABC):
    """Defines an equation's term using an UFL expression.

    The implemented expression describes the term's contribution to the residual in the
    finite element discretisation.
    """
    def __init__(
        self,
        test_space: firedrake.functionspaceimpl.WithGeometry,
        trial_space: firedrake.functionspaceimpl.WithGeometry,
        dx: firedrake.Measure,
        ds: firedrake.Measure,
        dS: firedrake.Measure,
        **kwargs,
    ):
        """Initialises the equation instance given function spaces.

        Args:
          test_space:
            Firedrake function space of the test function.
          trial_space:
            Firedrake function space of the rial function.
          dx:
            UFL measure for the domain, boundaries excluded.
          ds:
            UFL measure for the domain's outer boundaries.
          dS:
            UFL measure for the domain's inner boundaries when using a discontinuous
            function space.
        """
        self.test_space = test_space
        self.trial_space = trial_space

        self.dx = dx
        self.ds = ds
        self.dS = dS

        self.mesh = test_space.mesh()
        self.dim = self.mesh.geometric_dimension()
        self.n = firedrake.FacetNormal(self.mesh)

        self.term_kwargs = kwargs

    @abstractmethod
    def residual(
        self,
        test: firedrake.ufl_expr.Argument,
        trial: firedrake.ufl_expr.Argument | firedrake.Function,
        trial_lagged: Optional[firedrake.ufl_expr.Argument | firedrake.Function] = None,
        fields: Optional[dict[str, firedrake.Constant | firedrake.Function]] = None,
        bcs: Optional[dict[int, dict[str, int | float]]] = None,
    ) -> firedrake.ufl.core.expr.Expr:
        """UFL expression for the residual associated with the equation's term.

        Args:
          test:
            Firedrake test function.
          trial:
            Firedrake trial function.
          trial_lagged:
            Firedrake trial function from the previous time step.
          fields:
            Dictionary of physical fields from the simulation's state.
          bcs:
            Dictionary of identifier-value pairs specifying boundary conditions.

        Returns:
          A UFL expression for the term's contribution to the finite element residual.
        """
        pass
