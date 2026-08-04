"""dimensionless_numbers.py

Reference book:
    MWH's Water Treatment: Principles and Design, 3rd ed.

Additional Sherwood correlations:
    Xu et al. (2013), Mathematically Modeling Fixed-Bed
    Adsorption in Aqueous Systems, Table 3.
"""

from .water_class import Water
from .media_class import Media
from .breakthrough_data import Breakthrough
from .column_data import Column
from .chemical_class import Chemical


def _positive_parameter(
    parameter_name: str,
    value: float | None,
) -> float:
    """Return a required positive parameter."""
    if value is None:
        raise ValueError(f"{parameter_name} is required.")

    assert value > 0, f"{parameter_name} must be positive, got {value}"

    return value


def _porosity_parameter(
    parameter_name: str,
    value: float | None,
) -> float:
    """Return a required porosity."""
    if value is None:
        raise ValueError(f"{parameter_name} is required.")

    assert 0 < value < 1, f"{parameter_name} must be between 0 and 1, got {value}"

    return value


def reynolds_number(
    density: float | None = None,
    interstitial_velocity: float | None = None,
    diameter: float | None = None,
    viscosity: float | None = None,
    *,
    water: Water | None = None,
    media: Media | None = None,
    column: Column | None = None,
    breakthrough: Breakthrough | None = None,
    superficial_velocity: float | None = None,
    bed_porosity: float | None = None,
    sphericity: float | None = None,
) -> float:
    """Calculate the standard or packed-bed Reynolds number."""
    if water is not None:
        if density is None:
            density = water.density

        if viscosity is None:
            viscosity = water.viscosity

    if media is not None:
        if diameter is None:
            diameter = media.mean_diameter

        if sphericity is None:
            sphericity = media.sphericity

    if interstitial_velocity is not None and superficial_velocity is not None:
        raise ValueError(
            "Provide either interstitial_velocity or " "superficial_velocity, not both."
        )

    if (
        interstitial_velocity is None
        and column is not None
        and breakthrough is not None
    ):
        superficial_velocity = breakthrough.calculate_superficial_velocity(
            breakthrough.flow_rate,
            column.cross_section_area(),
        )

        interstitial_velocity = breakthrough.calculate_interstitial_velocity(
            superficial_velocity,
            column.porosity,
        )

    if interstitial_velocity is None and superficial_velocity is not None:
        if bed_porosity is None and column is not None:
            bed_porosity = column.porosity

        bed_porosity = _porosity_parameter(
            "bed_porosity",
            bed_porosity,
        )

        superficial_velocity = _positive_parameter(
            "superficial_velocity",
            superficial_velocity,
        )

        interstitial_velocity = superficial_velocity / bed_porosity

    density = _positive_parameter(
        "density",
        density,
    )

    interstitial_velocity = _positive_parameter(
        "interstitial_velocity",
        interstitial_velocity,
    )

    diameter = _positive_parameter(
        "diameter",
        diameter,
    )

    viscosity = _positive_parameter(
        "viscosity",
        viscosity,
    )

    if sphericity is None:
        sphericity = 1.0

    assert 0 < sphericity <= 1, f"sphericity must be in (0, 1], got {sphericity}"

    return density * sphericity * interstitial_velocity * diameter / viscosity


def schmidt_number(
    viscosity: float | None = None,
    density: float | None = None,
    diffusion_coefficient: float | None = None,
    *,
    water: Water | None = None,
    chemical: Chemical | None = None,
) -> float:
    """Calculate the Schmidt number."""
    if water is not None:
        if viscosity is None:
            viscosity = water.viscosity

        if density is None:
            density = water.density

    viscosity = _positive_parameter(
        "viscosity",
        viscosity,
    )

    density = _positive_parameter(
        "density",
        density,
    )

    if diffusion_coefficient is None:
        if chemical is None:
            raise ValueError(
                "chemical is required when diffusion_coefficient " "is not supplied."
            )

        diffusion_coefficient = chemical.liquid_diffusion_coefficient(viscosity)

    diffusion_coefficient = _positive_parameter(
        "diffusion_coefficient",
        diffusion_coefficient,
    )

    return viscosity / (density * diffusion_coefficient)


def _resolve_reynolds_and_schmidt(
    reynolds: float | None,
    schmidt: float | None,
    *,
    water: Water | None,
    chemical: Chemical | None,
    media: Media | None,
    column: Column | None,
    breakthrough: Breakthrough | None,
) -> tuple[float, float]:
    """Return supplied or calculated Reynolds and Schmidt numbers."""
    if reynolds is None:
        reynolds = reynolds_number(
            water=water,
            media=media,
            column=column,
            breakthrough=breakthrough,
        )

    if schmidt is None:
        schmidt = schmidt_number(
            water=water,
            chemical=chemical,
        )

    reynolds = _positive_parameter(
        "reynolds",
        reynolds,
    )
    schmidt = _positive_parameter(
        "schmidt",
        schmidt,
    )

    return reynolds, schmidt


def peclet_number(
    interstitial_velocity: float | None = None,
    length: float | None = None,
    axial_dispersion_coefficient: float | None = None,
    *,
    reynolds: float | None = None,
    schmidt: float | None = None,
    water: Water | None = None,
    chemical: Chemical | None = None,
    media: Media | None = None,
    column: Column | None = None,
    breakthrough: Breakthrough | None = None,
) -> float:
    """Calculate an axial or mass-transfer Peclet number."""
    axial_parameters_provided = any(
        value is not None
        for value in (
            interstitial_velocity,
            length,
            axial_dispersion_coefficient,
        )
    )

    mass_transfer_parameters_provided = reynolds is not None or schmidt is not None

    if axial_parameters_provided and mass_transfer_parameters_provided:
        raise ValueError(
            "Provide axial-dispersion parameters or Reynolds and "
            "Schmidt numbers, not both."
        )

    if axial_parameters_provided:
        if (
            interstitial_velocity is None
            and column is not None
            and breakthrough is not None
        ):
            superficial_velocity = breakthrough.calculate_superficial_velocity(
                breakthrough.flow_rate,
                column.cross_section_area(),
            )

            interstitial_velocity = breakthrough.calculate_interstitial_velocity(
                superficial_velocity,
                column.porosity,
            )

        if length is None and column is not None:
            length = column.length

        interstitial_velocity = _positive_parameter(
            "interstitial_velocity",
            interstitial_velocity,
        )

        length = _positive_parameter(
            "length",
            length,
        )

        axial_dispersion_coefficient = _positive_parameter(
            "axial_dispersion_coefficient",
            axial_dispersion_coefficient,
        )

        return length * interstitial_velocity / axial_dispersion_coefficient

    if reynolds is None:
        reynolds = reynolds_number(
            water=water,
            media=media,
            column=column,
            breakthrough=breakthrough,
        )

    if schmidt is None:
        schmidt = schmidt_number(
            water=water,
            chemical=chemical,
        )

    reynolds = _positive_parameter(
        "reynolds",
        reynolds,
    )

    schmidt = _positive_parameter(
        "schmidt",
        schmidt,
    )

    return reynolds * schmidt


def sherwood_number(
    method: str,
    reynolds: float | None = None,
    schmidt: float | None = None,
    bed_porosity: float | None = None,
    *,
    water: Water | None = None,
    chemical: Chemical | None = None,
    media: Media | None = None,
    column: Column | None = None,
    breakthrough: Breakthrough | None = None,
) -> float:
    """Calculate Sherwood number using the selected correlation.

    Available methods:

        tan
        wilson_geankoplis
        ohashi
        williamson
        ko
        wakao_funazkri
        kataoka
        chern_chien
        gnielinski
    """
    method = method.lower()

    reynolds, schmidt = _resolve_reynolds_and_schmidt(
        reynolds,
        schmidt,
        water=water,
        chemical=chemical,
        media=media,
        column=column,
        breakthrough=breakthrough,
    )

    if bed_porosity is None and column is not None:
        bed_porosity = column.porosity

    peclet = reynolds * schmidt

    if method == "tan":
        bed_porosity = _porosity_parameter(
            "bed_porosity",
            bed_porosity,
        )

        return 1.1 * peclet ** (1 / 3) / bed_porosity

    if method == "wilson_geankoplis":
        bed_porosity = _porosity_parameter(
            "bed_porosity",
            bed_porosity,
        )

        porosity_reynolds = bed_porosity * reynolds

        if not 0.0016 < porosity_reynolds < 55:
            raise ValueError(
                "Wilson-Geankoplis requires " "0.0016 < bed_porosity * Reynolds < 55."
            )

        if not 950 < schmidt < 70000:
            raise ValueError("Wilson-Geankoplis requires " "950 < Schmidt < 70000.")

        return 1.09 * peclet ** (1 / 3) / bed_porosity

    if method == "ohashi":
        if reynolds <= 0.001:
            raise ValueError("Ohashi requires Reynolds > 0.001.")

        if reynolds < 5.8:
            return 2 + 1.58 * reynolds**0.4 * schmidt ** (1 / 3)

        if reynolds < 500:
            return 2 + 1.21 * reynolds**0.5 * schmidt ** (1 / 3)

        return 2 + 0.59 * reynolds**0.6 * schmidt ** (1 / 3)

    if method == "williamson":
        bed_porosity = _porosity_parameter(
            "bed_porosity",
            bed_porosity,
        )

        if not 0.08 < reynolds < 125:
            raise ValueError("Williamson requires " "0.08 < Reynolds < 125.")

        if not 150 < schmidt < 1300:
            raise ValueError("Williamson requires " "150 < Schmidt < 1300.")

        return 2.4 * bed_porosity * reynolds**0.3 * schmidt**0.42

    if method == "ko":
        bed_porosity = _porosity_parameter(
            "bed_porosity",
            bed_porosity,
        )

        return 0.325 / (bed_porosity * reynolds**0.36 * schmidt ** (1 / 3))

    if method == "wakao_funazkri":
        if not 3 < reynolds < 10000:
            raise ValueError("Wakao-Funazkri requires " "3 < Reynolds < 10000.")

        return 2 + 1.1 * reynolds**0.6 * schmidt ** (1 / 3)

    if method == "kataoka":
        bed_porosity = _porosity_parameter(
            "bed_porosity",
            bed_porosity,
        )

        adjusted_reynolds = reynolds * bed_porosity / (1 - bed_porosity)

        if adjusted_reynolds >= 100:
            raise ValueError(
                "Kataoka requires "
                "Reynolds * bed_porosity "
                "/ (1 - bed_porosity) < 100."
            )

        return (
            1.85
            * ((1 - bed_porosity) / bed_porosity) ** (1 / 3)
            * reynolds ** (1 / 3)
            * schmidt ** (1 / 3)
        )

    if method == "chern_chien":
        bed_porosity = _porosity_parameter(
            "bed_porosity",
            bed_porosity,
        )

        return (2 + 0.644 * reynolds**0.5 * schmidt ** (1 / 3)) * (
            1 + 1.5 * (1 - bed_porosity)
        )

    if method == "gnielinski":
        bed_porosity = _porosity_parameter(
            "bed_porosity",
            bed_porosity,
        )

        if peclet <= 500:
            raise ValueError("Gnielinski requires " "Reynolds * Schmidt > 500.")

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

    raise ValueError(f"Unknown Sherwood method: {method}")


# note: missing some parameters that are not calculated yet,
# will add it when we complete numerical methods

# def helfferich_number(
#     total_resin_capacity: float,
#     intraparticle_diffusion_coefficient: float,
#     film_thickness: float,
#     liquid_concentration: float,
#     liquid_diffusion_coefficient: float,
#     particle_radius: float,
#     separation_factor: float,
# ) -> float:
#     """Helfferich number for ion-exchange rate-control evaluation.
#
#     He = (qT * Dp * film_thickness)
#          / (C * Dl * particle_radius)
#          * (5 + 2 * separation_factor)
#
#     Interpretation:
#         He << 1: intraparticle diffusion controls.
#         He >> 1: liquid-film diffusion controls.
#         He near 1: both contribute.
#
#     Reference:
#         MWH, Chapter 16, p. 1298, Eq. 16-38.
#
#     This function is reserved for future implementation after the
#     required numerical-model parameters are added to ReactorModels.
#     """
#     assert total_resin_capacity > 0, (
#         "total_resin_capacity must be positive, "
#         f"got {total_resin_capacity}"
#     )
#
#     assert intraparticle_diffusion_coefficient > 0, (
#         "intraparticle_diffusion_coefficient must be positive, "
#         f"got {intraparticle_diffusion_coefficient}"
#     )
#
#     assert film_thickness > 0, (
#         "film_thickness must be positive, "
#         f"got {film_thickness}"
#     )
#
#     assert liquid_concentration > 0, (
#         "liquid_concentration must be positive, "
#         f"got {liquid_concentration}"
#     )
#
#     assert liquid_diffusion_coefficient > 0, (
#         "liquid_diffusion_coefficient must be positive, "
#         f"got {liquid_diffusion_coefficient}"
#     )
#
#     assert particle_radius > 0, (
#         "particle_radius must be positive, "
#         f"got {particle_radius}"
#     )
#
#     assert separation_factor > 0, (
#         "separation_factor must be positive, "
#         f"got {separation_factor}"
#     )
#
#     return (
#         total_resin_capacity
#         * intraparticle_diffusion_coefficient
#         * film_thickness
#         / (
#             liquid_concentration
#             * liquid_diffusion_coefficient
#             * particle_radius
#         )
#         * (
#             5
#             + 2 * separation_factor
#         )
#     )
