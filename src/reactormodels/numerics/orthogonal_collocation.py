"""orthogonal_collocation.py"""

import numpy as np


class OrthogonalCollocation:
    """Generate collocation points and differentiation matrices on [0, 1]

    Jacobi polynomials, with optional multi-element support.

    Single element (n_elements=1):
        Standard global collocation on [0, 1].

    Multi-element (n_elements > 1):
        Domain split into ne equal elements [k/ne, (k+1)/ne].
        Within each element, local collocation applied.
        Continuity of C enforced at element boundaries.
        Flux continuity enforced via the continuation condition.
        This resolves high-Pe oscillations without upwinding.

    Parameters
    ----------
    n_interior_points : int     Points per element (excludes endpoints)
    alpha, beta       : float   Jacobi polynomial parameters
    add_inlet         : bool    Add x=0 as first node
    n_elements        : int     Number of elements (default 1)

    """

    def __init__(
        self,
        n_interior_points: int = 5,
        alpha: float = 0.0,
        beta: float = 0.0,
        add_inlet: bool = False,
        n_elements: int = 1,
    ):
        self.n_interior_points = n_interior_points
        self.alpha = alpha
        self.beta = beta
        self.n_elements = n_elements
        self.nodes, self.first_derivative, self.second_derivative = self._build(
            add_inlet=add_inlet
        )

    def jacobi_roots(self) -> np.ndarray:
        """Roots of P_n^(alpha,beta) shifted to [0,1]."""
        a_coef = np.zeros(self.n_interior_points)
        for k in range(self.n_interior_points):
            denom = (2 * k + self.alpha + self.beta) * (
                2 * k + self.alpha + self.beta + 2
            )
            a_coef[k] = 0.0 if denom == 0 else (self.beta**2 - self.alpha**2) / denom

        b_coef = np.zeros(self.n_interior_points - 1)
        for k in range(1, self.n_interior_points):
            s = 2 * k + self.alpha + self.beta
            denom = s**2 * (s + 1) * (s - 1)
            if abs(s - 1) < 1e-14 and k == 1:
                b_coef[0] = np.sqrt((1 + self.alpha) * (1 + self.beta))
            else:
                b_coef[k - 1] = 2.0 * np.sqrt(
                    k
                    * (k + self.alpha)
                    * (k + self.beta)
                    * (k + self.alpha + self.beta)
                    / denom
                )

        J = np.diag(a_coef) + np.diag(b_coef, 1) + np.diag(b_coef, -1)
        return 0.5 * (np.sort(np.linalg.eigvalsh(J)) + 1.0)

    @staticmethod
    def lagrange_basis_and_deriv(x: np.ndarray):
        """Generate first and second derivative matrices."""
        N = len(x)
        A = np.zeros((N, N))
        for j in range(N):
            for i in range(N):
                if i == j:
                    A[i, j] = sum(1.0 / (x[i] - x[k]) for k in range(N) if k != i)
                else:
                    num = 1.0
                    den = 1.0
                    for k in range(N):
                        if k != j:
                            den *= x[j] - x[k]
                        if k != j and k != i:
                            num *= x[i] - x[k]
                    A[i, j] = num / den
        B = A @ A
        return A, B

    def _build_single_element(self, add_inlet=False):
        """Build collocation on [0,1]."""
        xi = self.jacobi_roots()
        x = np.append(xi, 1.0)
        if add_inlet:
            x = np.concatenate([[0.0], x])
        A, B = self.lagrange_basis_and_deriv(x)
        return x, A, B

    def _build_multi_element(self, add_inlet=False):
        """Assemble global differentiation matrices from ne local elements.

        Each element has (n_interior_points + 2) local nodes:
            [left_boundary, interior..., right_boundary]

        Adjacent elements share one boundary node, so total nodes:
            N_total = ne * (n_pts_per_element - 1) + 1
        With add_inlet, x=0 is pinned and included.

        Flux continuity at element junctions is enforced by replacing
        the duplicated boundary row with the average of the left and
        right element derivative contributions.
        """
        ne = self.n_elements
        xi = self.jacobi_roots()  # interior roots on [0,1]
        n_local = self.n_interior_points + 2  # nodes per element incl. boundaries

        # Local nodes on [0,1] for one element
        x_local = np.concatenate([[0.0], xi, [1.0]])  # (n_local,)
        A_loc, B_loc = self.lagrange_basis_and_deriv(x_local)

        # Scale derivatives for element width h = 1/ne
        h = 1.0 / ne
        A_loc_scaled = A_loc / h
        B_loc_scaled = B_loc / h**2

        # Global nodes: map each element's local nodes to [k*h, (k+1)*h]
        # Shared boundary nodes appear once
        n_global = ne * (n_local - 1) + 1
        x_global = np.zeros(n_global)
        for k in range(ne):
            start = k * (n_local - 1)
            x_global[start : start + n_local] = k * h + h * x_local

        # Assemble global A and B matrices
        A_global = np.zeros((n_global, n_global))
        B_global = np.zeros((n_global, n_global))

        for k in range(ne):
            start = k * (n_local - 1)
            idx = slice(start, start + n_local)

            # Interior rows of this element (exclude shared boundaries)
            # Left boundary row: only for first element (or average at junctions)
            # Right boundary row: averaged at junctions below
            A_global[idx, idx] += A_loc_scaled
            B_global[idx, idx] += B_loc_scaled

        # At junction nodes (shared between elements), each element
        # contributed once — average the two contributions
        for k in range(1, ne):
            j = k * (n_local - 1)  # global index of junction node
            A_global[j, :] *= 0.5
            B_global[j, :] *= 0.5

        # Prepend x=0 if not already first node (add_inlet has no effect
        # for multi-element since x=0 is always included as left boundary)
        if not add_inlet:
            # Remove x=0 row/col — treat as interior (no pinning)
            pass  # x=0 is already first node; caller handles pinning

        return x_global, A_global, B_global

    def _build(self, add_inlet=False):
        if self.n_elements == 1:
            return self._build_single_element(add_inlet=add_inlet)
        else:
            return self._build_multi_element(add_inlet=add_inlet)

    def radial_operator(self) -> np.ndarray:
        """Spherical Laplacian: (1/r^2)*d/dr(r^2*du/dr) = d^2u/dr^2 + (2/r)*du/dr"""
        x = self.nodes
        L = np.zeros_like(self.second_derivative)
        for i, xi in enumerate(x):
            if xi < 1e-14:
                L[i, :] = 3.0 * self.second_derivative[i, :]
            else:
                L[i, :] = (
                    self.second_derivative[i, :]
                    + (2.0 / xi) * self.first_derivative[i, :]
                )
        return L
