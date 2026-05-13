# from model.orthogonal_collocation import OrthogonalCollocation
# from model.isotherm import Isotherm, FreundlichIsotherm, LinearIsotherm
# from model.advection_diffusion import PSDMParticle
# import numpy as np
# import pytest


# def _make_particle(n_col=4, Ds=0.0):
#     iso = LinearIsotherm(K=100.0)
#     return PSDMParticle(
#         isotherm=iso,
#         R=5e-4,  # 0.5 mm radius
#         Dp=1e-10,  # m^2/s
#         Ds=Ds,
#         kf=1e-4,  # m/s
#         eps_p=0.4,
#         rho_p=800.0,  # kg/m^3
#         n_col=n_col,
#     )


# def test_initial_state_zero():
#     p = _make_particle()
#     y0 = p.initial_state()
#     np.testing.assert_array_equal(y0, np.zeros(2 * p.N))


# def test_initial_state_nonzero():
#     p = _make_particle()
#     y0 = p.initial_state(Cp0=1.0, q0=50.0)
#     Cp, q = p.split_state(y0)
#     np.testing.assert_array_equal(Cp, np.ones(p.N))
#     np.testing.assert_array_equal(q, 50.0 * np.ones(p.N))


# def test_rhs_shape():
#     p = _make_particle()
#     y0 = p.initial_state()
#     dydt = p.rhs(0.0, y0, Cb=5.0)
#     assert dydt.shape == y0.shape


# def test_rhs_equilibrium_no_surface_diffusion():
#     """
#     At equilibrium (Cp = Cb, q = q*(Cb)) and Ds=0,
#     the RHS should be near zero (no driving force).
#     """
#     p = _make_particle(Ds=0.0)
#     Cb = 5.0
#     q_eq = p.isotherm.q(np.array([Cb]))[0]
#     # Set ALL collocation points to equilibrium
#     y_eq = p.pack_state(
#         np.full(p.N, Cb),
#         np.full(p.N, q_eq),
#     )
#     dydt = p.rhs(0.0, y_eq, Cb=Cb)
#     np.testing.assert_allclose(dydt, np.zeros_like(dydt), atol=1e-6)


# def test_solve_monotonic_surface_concentration():
#     """
#     Starting from zero, Cp at the surface should monotonically increase
#     toward the bulk concentration.
#     """
#     p = _make_particle(n_col=4)
#     sol = p.solve(
#         t_span=(0, 3600),
#         Cb_func=lambda t: 10.0,
#         t_eval=np.linspace(0, 3600, 50),
#     )
#     assert sol.success
#     Cp_surface = np.array(
#         [p.surface_concentration(sol.y[:, k]) for k in range(sol.y.shape[1])]
#     )
#     # Monotonically increasing
#     diffs = np.diff(Cp_surface)
#     assert np.all(diffs >= -1e-8), "Surface Cp should be non-decreasing"
#     # Should approach (but not exceed) bulk
#     assert Cp_surface[-1] <= 10.0 + 1e-6


# def test_solve_reaches_equilibrium():
#     """
#     After a long time, average loading should approach q*(Cb).
#     """
#     iso = LinearIsotherm(K=50.0)
#     p = PSDMParticle(
#         isotherm=iso,
#         R=5e-4,
#         Dp=5e-10,
#         Ds=1e-13,
#         kf=1e-3,
#         eps_p=0.4,
#         rho_p=800.0,
#         n_col=4,
#     )
#     Cb = 5.0
#     t_end = 1e5  # very long time
#     sol = p.solve(
#         t_span=(0, t_end),
#         Cb_func=lambda t: Cb,
#     )
#     assert sol.success
#     q_avg = p.average_loading(sol.y[:, -1])
#     q_target = iso.q(np.array([Cb]))[0]
#     rel_err = abs(q_avg - q_target) / q_target
#     assert rel_err < 0.05, f"q_avg={q_avg:.4f}, q_target={q_target:.4f}"


# def test_surface_diffusion_accelerates_uptake():
#     """
#     Adding surface diffusion should increase the average loading at intermediate time.
#     """
#     t_mid = 1800.0
#     iso = FreundlichIsotherm(K=20.0, n=2.0)

#     def make(Ds):
#         return PSDMParticle(
#             isotherm=iso,
#             R=5e-4,
#             Dp=1e-10,
#             Ds=Ds,
#             kf=1e-4,
#             eps_p=0.4,
#             rho_p=800.0,
#             n_col=4,
#         )

#     for Ds_val in [0.0, 1e-13]:
#         p = make(Ds_val)
#         sol = p.solve(
#             t_span=(0, t_mid),
#             Cb_func=lambda t: 10.0,
#         )
#         assert sol.success

#     p0 = make(0.0)
#     ps = make(1e-13)
#     sol0 = p0.solve((0, t_mid), lambda t: 10.0)
#     sols = ps.solve((0, t_mid), lambda t: 10.0)
#     q0 = p0.average_loading(sol0.y[:, -1])
#     qs = ps.average_loading(sols.y[:, -1])
#     assert (
#         qs >= q0 - 1e-8
#     ), "Surface diffusion should not reduce uptake at intermediate time"


# def test_step_change_bulk():
#     """Solve with a step change in bulk concentration at t=1800s."""
#     p = _make_particle(n_col=4)

#     def Cb_step(t):
#         return 5.0 if t < 1800 else 10.0

#     sol = p.solve(
#         t_span=(0, 5400),
#         Cb_func=Cb_step,
#         t_eval=np.linspace(0, 5400, 100),
#     )
#     assert sol.success
#     # Cp at surface must be bounded
#     Cp_surface = np.array(
#         [p.surface_concentration(sol.y[:, k]) for k in range(sol.y.shape[1])]
#     )
#     assert np.all(Cp_surface >= -1e-8)
#     assert np.all(Cp_surface <= 10.0 + 1e-4)


# def test_average_loading_nonnegative():
#     p = _make_particle()
#     y0 = p.initial_state()
#     sol = p.solve((0, 3600), lambda t: 5.0)
#     for k in range(sol.y.shape[1]):
#         q_avg = p.average_loading(sol.y[:, k])
#         assert q_avg >= -1e-10
