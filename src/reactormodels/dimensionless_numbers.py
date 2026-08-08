"""dimensionless_numbers.py

Reference book:
    MWH's Water Treatment: Principles and Design, 3rd ed.

Additional Sherwood correlations:
    Xu et al. (2013), Mathematically Modeling Fixed-Bed
    Adsorption in Aqueous Systems, Table 3.
"""

from collections.abc import Callable


def reynolds_number(
    density: float,
    interstitial_velocity: float,
    diameter: float,
    viscosity: float,
    sphericity: float = 1.0,
) -> float:
    """Calculate the standard or packed-bed Reynolds number."""
    assert 0 < sphericity <= 1, f"sphericity must be in (0, 1], got {sphericity}"

    return density * sphericity * interstitial_velocity * diameter / viscosity


def schmidt_number(
    viscosity: float,
    density: float,
    diffusion_coefficient: float,
) -> float:
    """Calculate the Schmidt number."""
    return viscosity / (density * diffusion_coefficient)


def peclet_number(
    interstitial_velocity: float,
    length: float,
    diffusion: float,
) -> float:
    """Calculate the Peclet number."""
    return length * interstitial_velocity / diffusion


_PorosityRequiredMethods = frozenset(
    {
        "tan",
        "wilson_geankoplis",
        "williamson",
        "ko",
        "kataoka",
        "chern_chien",
        "gnielinski",
    }
)


def sherwood_number(
    method: str,
    reynolds: float,
    schmidt: float,
    bed_porosity: float | None = None,
) -> float:
    """Calculate Sherwood number using the selected correlation.

    Available methods:
        tan, wilson_geankoplis, ohashi, williamson, ko,
        wakao_funazkri, kataoka, chern_chien, gnielinski
    """
    method = method.lower()

    try:
        correlation = _CORRELATIONS[method]
    except KeyError:
        raise ValueError(f"Unknown Sherwood method: {method}") from None

    if method in _PorosityRequiredMethods and bed_porosity is None:
        raise ValueError(f"{method} requires bed_porosity.")

    return correlation(reynolds, schmidt, bed_porosity)


def _tan(reynolds: float, schmidt: float, bed_porosity: float | None) -> float:
    """Calculate Sherwood number using the Tan correlation."""
    assert bed_porosity is not None
    peclet = reynolds * schmidt
    return 1.1 * peclet ** (1 / 3) / bed_porosity


def _wilson_geankoplis(
    reynolds: float, schmidt: float, bed_porosity: float | None
) -> float:
    """Calculate Sherwood number using the Wilson-Geankoplis correlation."""
    assert bed_porosity is not None
    porosity_reynolds = bed_porosity * reynolds

    if not 0.0016 < porosity_reynolds < 55:
        raise ValueError(
            "Wilson-Geankoplis requires 0.0016 < bed_porosity * Reynolds < 55."
        )
    if not 950 < schmidt < 70000:
        raise ValueError("Wilson-Geankoplis requires 950 < Schmidt < 70000.")

    peclet = reynolds * schmidt
    return 1.09 * peclet ** (1 / 3) / bed_porosity


def _ohashi(reynolds: float, schmidt: float, _bed_porosity: float | None) -> float:
    """Calculate Sherwood number using the Ohashi correlation."""
    if reynolds <= 0.001:
        raise ValueError("Ohashi requires Reynolds > 0.001.")

    if reynolds < 5.8:
        return 2 + 1.58 * reynolds**0.4 * schmidt ** (1 / 3)
    if reynolds < 500:
        return 2 + 1.21 * reynolds**0.5 * schmidt ** (1 / 3)
    return 2 + 0.59 * reynolds**0.6 * schmidt ** (1 / 3)


def _williamson(reynolds: float, schmidt: float, bed_porosity: float | None) -> float:
    """Calculate Sherwood number using the Williamson correlation."""
    assert bed_porosity is not None
    if not 0.08 < reynolds < 125:
        raise ValueError("Williamson requires 0.08 < Reynolds < 125.")
    if not 150 < schmidt < 1300:
        raise ValueError("Williamson requires 150 < Schmidt < 1300.")

    return 2.4 * bed_porosity * reynolds**0.3 * schmidt**0.42


def _ko(reynolds: float, schmidt: float, bed_porosity: float | None) -> float:
    """Calculate Sherwood number using the Ko correlation."""
    assert bed_porosity is not None
    return 0.325 / (bed_porosity * reynolds**0.36 * schmidt ** (1 / 3))


def _wakao_funazkri(
    reynolds: float, schmidt: float, _bed_porosity: float | None
) -> float:
    """Calculate Sherwood number using the Wakao-Funazkri correlation."""
    if not 3 < reynolds < 10000:
        raise ValueError("Wakao-Funazkri requires 3 < Reynolds < 10000.")

    return 2 + 1.1 * reynolds**0.6 * schmidt ** (1 / 3)


def _kataoka(reynolds: float, schmidt: float, bed_porosity: float | None) -> float:
    """Calculate Sherwood number using the Kataoka correlation."""
    assert bed_porosity is not None
    adjusted_reynolds = reynolds * bed_porosity / (1 - bed_porosity)

    if adjusted_reynolds >= 100:
        raise ValueError(
            "Kataoka requires Reynolds * bed_porosity / (1 - bed_porosity) < 100."
        )

    return (
        1.85
        * ((1 - bed_porosity) / bed_porosity) ** (1 / 3)
        * reynolds ** (1 / 3)
        * schmidt ** (1 / 3)
    )


def _chern_chien(reynolds: float, schmidt: float, bed_porosity: float | None) -> float:
    """Calculate Sherwood number using the Chern-Chien correlation."""
    assert bed_porosity is not None
    return (2 + 0.644 * reynolds**0.5 * schmidt ** (1 / 3)) * (
        1 + 1.5 * (1 - bed_porosity)
    )


def _gnielinski(reynolds: float, schmidt: float, bed_porosity: float | None) -> float:
    """Calculate Sherwood number using the Gnielinski correlation."""
    assert bed_porosity is not None
    peclet = reynolds * schmidt

    if peclet <= 500:
        raise ValueError("Gnielinski requires Reynolds * Schmidt > 500.")
    if schmidt >= 12000:
        raise ValueError("Gnielinski requires Schmidt < 12000.")

    sherwood_laminar = 0.644 * reynolds**0.5 * schmidt ** (1 / 3)
    sherwood_turbulent = (
        0.037
        * reynolds**0.8
        * schmidt
        / (1 + 2.443 * reynolds ** (-0.1) * (schmidt ** (2 / 3) - 1))
    )

    return (2 + (sherwood_laminar**2 + sherwood_turbulent**2) ** 0.5) * (
        1 + 1.5 * (1 - bed_porosity)
    )


_CORRELATIONS: dict[str, Callable[[float, float, float | None], float]] = {
    "tan": _tan,
    "wilson_geankoplis": _wilson_geankoplis,
    "ohashi": _ohashi,
    "williamson": _williamson,
    "ko": _ko,
    "wakao_funazkri": _wakao_funazkri,
    "kataoka": _kataoka,
    "chern_chien": _chern_chien,
    "gnielinski": _gnielinski,
}
