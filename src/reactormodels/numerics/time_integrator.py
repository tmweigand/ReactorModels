"""Time integration via the SUNDIALS IDA DAE solver."""

import numpy as np
from scikits.odes import dae


class TimeIntegrator:
    """Time integrator class"""

    def __init__(self, rtol, atol, max_steps):
        self.rtol = rtol
        self.atol = atol
        self.max_steps = max_steps

    def integrate_ida(
        self,
        residual,
        y0: np.ndarray,
        yp0: np.ndarray,
        t_span: tuple,
        t_eval: np.ndarray,
        algebraic_vars_idx: list,
        jacobian=None,
    ):
        """Integrate a DAE system using the SUNDIALS IDA solver.

        See scikit_odes_sundioals/idas.pyx for more details.
        """
        if jacobian is None:
            solver = dae(
                "ida",
                residual,
                # jacfn=jacobian,
                algebraic_vars_idx=algebraic_vars_idx,
                rtol=self.rtol,
                atol=self.atol,
                max_steps=self.max_steps,
                # old_api=False,
                compute_initcond="yp0",
                linsolver="dense",
            )

        else:
            solver = dae(
                "ida",
                residual,
                jacfn=jacobian,
                algebraic_vars_idx=algebraic_vars_idx,
                rtol=self.rtol,
                atol=self.atol,
                max_steps=self.max_steps,
                # old_api=False,
                compute_initcond="yp0",
                linsolver="dense",
            )

        t_out = np.concatenate([[t_span[0]], t_eval])
        return solver.solve(t_out, y0, yp0)
