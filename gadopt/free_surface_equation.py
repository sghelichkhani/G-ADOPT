from .equations import BaseTerm, BaseEquation
from firedrake import dot, inner, div, grad, avg, jump, sign
from firedrake import min_value, Identity, assemble
from firedrake import FacetArea, CellVolume
from .utility import is_continuous, normal_is_continuous, cell_edge_integral_ratio
r"""
This module contains the scalar terms and equations (e.g. for temperature and salinity transport)

NOTE: for all terms, the residual() method returns the residual as it would be on the RHS of the equation, i.e.:

  dq/dt = \sum term.residual()

This sign-convention is for compatibility with Thetis' timeintegrators. In general, however we like to think about
the terms as they are on the LHS. Therefore in the residual methods below we assemble in F as it would be on the LHS:

  dq/dt + F(q) = 0

and at the very end "return -F".
"""


class FreeSurfaceTerm(BaseTerm):
    r"""
    Free Surface term: u \dot n
    """
    def residual(self, test, trial, trial_lagged, fields, bcs):
        u = fields['velocity']
        psi = test
        n = self.n
        continuous_u_normal = normal_is_continuous(u)
        if 'advective_velocity_scaling' in fields:
            u = fields['advective_velocity_scaling'] * u

        print("eta u.. inside: ", u.dat.data[:])
        F = psi * dot(u,n) * self.ds(4)  # Note this term is already on the RHS
        print("hello free surface")
        print("assemble RHS...: ", assemble(F))
        print("type assemble RHS...: ", type(assemble(F)))
        print("assemble RHS...: ", assemble(F).dat.data[:])

        return F




class FreeSurfaceEquation(BaseEquation):
    """
    Free Surface Equation.
    """

    terms = [FreeSurfaceTerm]

