"""orthogonal_collocation.py"""

import numpy as np
from scipy.special import beta as beta_fn


class OrthogonalCollocation:
    """Generate collocation points and differentiation matrices on [0, L].

    Jacobi polynomials, with optional multi-element support.

    Single element (n_elements=1):
        Standard global collocation on [0, L].

    Multi-element (n_elements > 1):
        Domain split into ne equal elements [k*L/ne, (k+1)*L/ne].
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
        domain_length: float = 1.0,
        n_interior_points: int = 5,
        alpha: float = 0.0,
        beta: float = 0.0,
        add_inlet: bool = False,
        n_elements: int = 1,
    ):
        self.domain_length = domain_length
        self.n_interior_points = n_interior_points
        self.alpha = alpha
        self.beta = beta
        self.add_inlet = add_inlet
        self.n_elements = n_elements
        self.nodes, self.first_derivative, self.second_derivative = self._build()
        self.radial_operator_matrix = self._build_radial_operator()

    def jacobi_roots_and_weights(self) -> tuple[np.ndarray, np.ndarray]:
        """Roots and quadrature weights of P_n^(alpha,beta), shifted to [0,1]."""
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
        eigenvalues, eigenvectors = np.linalg.eigh(J)
        idx = np.argsort(eigenvalues)
        nodes = 0.5 * (eigenvalues[idx] + 1.0)

        # Golub-Welsch: w_j = (v[0,j])^2 * mu_0
        # mu_0 = integral of weight function over [-1,1] shifted to [0,1]
        mu0 = (
            0.5
            * (2 ** (self.alpha + self.beta + 1))
            * beta_fn(self.alpha + 1, self.beta + 1)
        )
        weights = (eigenvectors[0, idx] ** 2) * mu0

        return nodes, weights

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

    def _build_single_element(self):
        """Build collocation on [0,1]."""
        xi, wi = self.jacobi_roots_and_weights()
        x = np.append(xi, 1.0)
        w = np.append(wi, 0.0)  # outlet node gets zero weight (not a quadrature point)
        if self.add_inlet:
            x = np.concatenate([[0.0], x])
            w = np.concatenate([[0.0], w])
        A, B = self.lagrange_basis_and_deriv(x)
        self.weights = w
        return x, A, B

    def _build_multi_element(self):
        """Build collocation for multiple elements on [0,1]."""
        ne = self.n_elements
        xi, wi = self.jacobi_roots_and_weights()  # interior roots + weights on [0,1]
        n_local = self.n_interior_points + 2

        x_local = np.concatenate([[0.0], xi, [1.0]])
        # boundary nodes are not quadrature points — zero weight
        w_local = np.concatenate([[0.0], wi, [0.0]])

        A_loc, B_loc = self.lagrange_basis_and_deriv(x_local)

        h = 1.0 / ne
        A_loc_scaled = A_loc / h
        B_loc_scaled = B_loc / h**2
        w_local_scaled = w_local * h  # integral scales with element width

        n_global = ne * (n_local - 1) + 1
        x_global = np.zeros(n_global)
        A_global = np.zeros((n_global, n_global))
        B_global = np.zeros((n_global, n_global))
        w_global = np.zeros(n_global)

        for k in range(ne):
            start = k * (n_local - 1)
            idx = slice(start, start + n_local)
            x_global[start : start + n_local] = k * h + h * x_local
            A_global[idx, idx] += A_loc_scaled
            B_global[idx, idx] += B_loc_scaled
            w_global[start : start + n_local] += w_local_scaled

        # Junction nodes were accumulated from two elements — average derivatives,
        # but SUM weights (each element contributes a distinct piece of the integral)
        for k in range(1, ne):
            j = k * (n_local - 1)
            A_global[j, :] *= 0.5
            B_global[j, :] *= 0.5
            # w_global[j] is already the correct sum — no averaging needed

        self.weights = w_global
        return x_global, A_global, B_global

    def _build(self):
        """Build elements"""
        if self.domain_length <= 0:
            raise ValueError("domain_length must be > 0")

        if self.n_elements == 1:
            nodes, first_derivative, second_derivative = self._build_single_element()
        else:
            nodes, first_derivative, second_derivative = self._build_multi_element()

        self.weights = self.weights * self.domain_length
        return (
            nodes * self.domain_length,
            first_derivative / self.domain_length,
            second_derivative / (self.domain_length**2),
        )

    def _build_radial_operator(self):
        """Spherical Laplacian: (1/r^2)*d/dr(r^2*du/dr) = d^2u/dr^2 + (2/r)*du/dr"""
        x = self.nodes
        L = np.zeros_like(self.second_derivative)

        for i, xi in enumerate(x):
            if xi < 1e-14:
                L[i] = 3.0 * self.second_derivative[i]
            else:
                L[i] = self.second_derivative[i] + (2.0 / xi) * self.first_derivative[i]

        return L

    def evaluate_radial_operator(
        self, f: np.ndarray, node: int | None = None
    ) -> np.ndarray | float:
        """Evaluate ∇²f = d²f/dr² + (2/r) df/dr."""
        if node is None:
            return self.radial_operator_matrix @ f
        return self.radial_operator_matrix[node, :] @ f

    def evaluate_gradient(self, f: np.ndarray, node: None | int = None) -> float:
        """Return df/dx at a specific collocation node.

        If node is provided, only return the value at specified node.
        """
        if node is None:
            return self.first_derivative @ f
        else:
            return self.first_derivative[node, :] @ f

    def evaluate_second_derivative(
        self, f: np.ndarray, node: None | int = None
    ) -> float:
        """Return d²f/dx² at a specific collocation node.

        If node is provided, only return the value at specified node.
        """
        if node is None:
            return self.second_derivative @ f
        else:
            return self.second_derivative[node, :] @ f

    def integrate(self, f: np.ndarray) -> float:
        """Return the weighted integral of f over [0, L] via quadrature weights."""
        return self.weights @ f
