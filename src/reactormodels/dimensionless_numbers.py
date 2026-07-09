"""dimensionless_numbers.py

Reference book is MWH's Water Treatment: Principles and Design, 3rd ed.
"""


def reynolds_number(
    density: float,
    interstitial_velocity: float,
    diameter: float,
    viscosity: float,
) -> float:
    """Reynolds number.

    Re = density * interstitial_velocity * diameter / viscosity

    For GAC or ion-exchange particles, use interstitial_velocity,
    not superficial_velocity.

    Reference:
        MWH, Chapter 7, p. 421, Eq. 7-56.
        MWH, Chapter 15, p. 1241, Eq. 15-219 for GAC RSSCT scaling.
    """
    assert density > 0, f"density must be positive, got {density}"
    assert (
        interstitial_velocity > 0
    ), f"interstitial_velocity must be positive, got {interstitial_velocity}"
    assert diameter > 0, f"diameter must be positive, got {diameter}"
    assert viscosity > 0, f"viscosity must be positive, got {viscosity}"

    return density * interstitial_velocity * diameter / viscosity


def packed_bed_reynolds_number(
    density: float,
    superficial_velocity: float,
    particle_diameter: float,
    viscosity: float,
    bed_porosity: float,
    sphericity: float = 1.0,
) -> float:
    """Packed-bed Reynolds number for GAC or ion-exchange resin beds.

    interstitial_velocity = superficial_velocity / bed_porosity

    Re = density * sphericity * particle_diameter * interstitial_velocity
         / viscosity

    This is equivalent to:

    Re = density * sphericity * particle_diameter * superficial_velocity
         / (bed_porosity * viscosity)

    Reference:
        MWH, Chapter 7, p. 423, Table 7-5.
    """
    assert density > 0, f"density must be positive, got {density}"
    assert (
        superficial_velocity > 0
    ), f"superficial_velocity must be positive, got {superficial_velocity}"
    assert (
        particle_diameter > 0
    ), f"particle_diameter must be positive, got {particle_diameter}"
    assert viscosity > 0, f"viscosity must be positive, got {viscosity}"
    assert (
        0 < bed_porosity < 1
    ), f"bed_porosity must be between 0 and 1, got {bed_porosity}"
    assert sphericity > 0, f"sphericity must be positive, got {sphericity}"

    interstitial_velocity = superficial_velocity / bed_porosity

    return density * sphericity * particle_diameter * interstitial_velocity / viscosity


def schmidt_number(
    viscosity: float,
    density: float,
    diffusion_coefficient: float,
) -> float:
    """Schmidt number.

    Sc = viscosity / (density * diffusion_coefficient)

    Reference:
        MWH, Chapter 7, p. 421, Eq. 7-55.
        MWH, Chapter 15, p. 1241, Eq. 15-217.
    """
    assert viscosity > 0, f"viscosity must be positive, got {viscosity}"
    assert density > 0, f"density must be positive, got {density}"
    assert (
        diffusion_coefficient > 0
    ), f"diffusion_coefficient must be positive, got {diffusion_coefficient}"

    return viscosity / (density * diffusion_coefficient)


def sherwood_number(
    mass_transfer_coefficient: float,
    characteristic_length: float,
    diffusion_coefficient: float,
) -> float:
    """Sherwood number.

    Sh = mass_transfer_coefficient * characteristic_length
         / diffusion_coefficient

    For GAC or ion-exchange particles, characteristic_length is normally
    the particle diameter.

    Reference:
        MWH, Chapter 7, p. 420, Eq. 7-52.
        MWH, Chapter 15, p. 1241, Eq. 15-214.
    """
    assert mass_transfer_coefficient > 0, (
        f"mass_transfer_coefficient must be positive, "
        f"got {mass_transfer_coefficient}"
    )
    assert (
        characteristic_length > 0
    ), f"characteristic_length must be positive, got {characteristic_length}"
    assert (
        diffusion_coefficient > 0
    ), f"diffusion_coefficient must be positive, got {diffusion_coefficient}"

    return mass_transfer_coefficient * characteristic_length / diffusion_coefficient


def gnielinski_sherwood_number(
    reynolds: float,
    schmidt: float,
    bed_porosity: float,
) -> float:
    """Sherwood number from the packed-bed Gnielinski correlation.

    Sh = [1 + 1.5 * (1 - bed_porosity)]
         * [2 + 0.644 * Re**0.5 * Sc**(1/3)]

    This is useful for estimating the external film mass-transfer
    coefficient for GAC and ion-exchange fixed beds.

    Reference:
        MWH, Chapter 7, p. 423, Table 7-5.
    """
    assert reynolds > 0, f"reynolds must be positive, got {reynolds}"
    assert schmidt > 0, f"schmidt must be positive, got {schmidt}"
    assert (
        0 < bed_porosity < 1
    ), f"bed_porosity must be between 0 and 1, got {bed_porosity}"

    return (1 + 1.5 * (1 - bed_porosity)) * (
        2 + 0.644 * reynolds**0.5 * schmidt ** (1 / 3)
    )


def film_transfer_coefficient_from_sherwood(
    sherwood: float,
    diffusion_coefficient: float,
    particle_diameter: float,
) -> float:
    """External film mass-transfer coefficient from Sherwood number.

    kf = Sh * diffusion_coefficient / particle_diameter

    Reference:
        Rearranged from MWH, Chapter 7, p. 420, Eq. 7-52.
    """
    assert sherwood > 0, f"sherwood must be positive, got {sherwood}"
    assert (
        diffusion_coefficient > 0
    ), f"diffusion_coefficient must be positive, got {diffusion_coefficient}"
    assert (
        particle_diameter > 0
    ), f"particle_diameter must be positive, got {particle_diameter}"

    return sherwood * diffusion_coefficient / particle_diameter


def peclet_number(
    interstitial_velocity: float,
    length: float,
    axial_dispersion_coefficient: float,
) -> float:
    """Axial Peclet number for fixed-bed GAC modeling.

    Pe = length * interstitial_velocity / axial_dispersion_coefficient

    Important:
        For GAC fixed-bed modeling, the denominator is the axial
        dispersion coefficient, not the molecular diffusion coefficient.

        The velocity should be the interstitial/pore-water velocity,
        not the superficial velocity.

    Reference:
        MWH, Chapter 15, p. 1211, Eq. 15-158.

    """
    assert (
        interstitial_velocity > 0
    ), f"interstitial_velocity must be positive, got {interstitial_velocity}"
    assert length > 0, f"length must be positive, got {length}"
    assert axial_dispersion_coefficient > 0, (
        "axial_dispersion_coefficient must be positive, "
        f"got {axial_dispersion_coefficient}"
    )

    return length * interstitial_velocity / axial_dispersion_coefficient


def packed_bed_mass_transfer_peclet_number(
    reynolds: float,
    schmidt: float,
) -> float:
    """Packed-bed mass-transfer Peclet number used as a Gnielinski constraint.

    Pe = Re * Sc

    This is not the same as the axial-dispersion Peclet number used in
    the GAC fixed-bed model.

    Reference:
        MWH, Chapter 7, p. 423, Table 7-5.
    """
    assert reynolds > 0, f"reynolds must be positive, got {reynolds}"
    assert schmidt > 0, f"schmidt must be positive, got {schmidt}"

    return reynolds * schmidt


def stanton_number(
    mass_transfer_coefficient: float,
    residence_time: float,
    bed_porosity: float,
    particle_radius: float,
) -> float:
    """Fixed-bed Stanton number for GAC and ion-exchange mass transfer.

    St = kf * residence_time * (1 - bed_porosity)
         / (bed_porosity * particle_radius)

    Here, residence_time is the pore-water residence time in the bed:

        residence_time = bed_porosity * EBCT

    Reference:
        MWH, Chapter 15, p. 1211, Eq. 15-159.
    """
    assert mass_transfer_coefficient > 0, (
        f"mass_transfer_coefficient must be positive, "
        f"got {mass_transfer_coefficient}"
    )
    assert residence_time > 0, f"residence_time must be positive, got {residence_time}"
    assert (
        0 < bed_porosity < 1
    ), f"bed_porosity must be between 0 and 1, got {bed_porosity}"
    assert (
        particle_radius > 0
    ), f"particle_radius must be positive, got {particle_radius}"

    return (
        mass_transfer_coefficient
        * residence_time
        * (1 - bed_porosity)
        / (bed_porosity * particle_radius)
    )


def stanton_number_from_ebct(
    mass_transfer_coefficient: float,
    empty_bed_contact_time: float,
    bed_porosity: float,
    particle_radius: float,
) -> float:
    """Fixed-bed Stanton number using EBCT directly.

    Since residence_time = bed_porosity * EBCT:

    St = kf * EBCT * (1 - bed_porosity) / particle_radius

    Reference:
        MWH, Chapter 15, p. 1211, Eqs. 15-153 and 15-159.
    """
    assert mass_transfer_coefficient > 0, (
        f"mass_transfer_coefficient must be positive, "
        f"got {mass_transfer_coefficient}"
    )
    assert (
        empty_bed_contact_time > 0
    ), f"empty_bed_contact_time must be positive, got {empty_bed_contact_time}"
    assert (
        0 < bed_porosity < 1
    ), f"bed_porosity must be between 0 and 1, got {bed_porosity}"
    assert (
        particle_radius > 0
    ), f"particle_radius must be positive, got {particle_radius}"

    return (
        mass_transfer_coefficient
        * empty_bed_contact_time
        * (1 - bed_porosity)
        / particle_radius
    )


def helfferich_number(
    total_resin_capacity: float,
    intraparticle_diffusion_coefficient: float,
    film_thickness: float,
    liquid_concentration: float,
    liquid_diffusion_coefficient: float,
    particle_radius: float,
    separation_factor: float,
) -> float:
    """Helfferich number for ion-exchange rate-control evaluation.

    He = (qT * Dp * film_thickness)
         / (C * Dl * particle_radius)
         * (5 + 2 * separation_factor)

    Interpretation:
        He << 1: intraparticle diffusion controls.
        He >> 1: liquid-film diffusion controls.
        He near 1: both contribute.

    Reference:
        MWH, Chapter 16, p. 1298, Eq. 16-38.
    """
    assert (
        total_resin_capacity > 0
    ), f"total_resin_capacity must be positive, got {total_resin_capacity}"
    assert intraparticle_diffusion_coefficient > 0, (
        "intraparticle_diffusion_coefficient must be positive, "
        f"got {intraparticle_diffusion_coefficient}"
    )
    assert film_thickness > 0, f"film_thickness must be positive, got {film_thickness}"
    assert (
        liquid_concentration > 0
    ), f"liquid_concentration must be positive, got {liquid_concentration}"
    assert liquid_diffusion_coefficient > 0, (
        "liquid_diffusion_coefficient must be positive, "
        f"got {liquid_diffusion_coefficient}"
    )
    assert (
        particle_radius > 0
    ), f"particle_radius must be positive, got {particle_radius}"
    assert (
        separation_factor > 0
    ), f"separation_factor must be positive, got {separation_factor}"

    return (
        total_resin_capacity
        * intraparticle_diffusion_coefficient
        * film_thickness
        / (liquid_concentration * liquid_diffusion_coefficient * particle_radius)
        * (5 + 2 * separation_factor)
    )
