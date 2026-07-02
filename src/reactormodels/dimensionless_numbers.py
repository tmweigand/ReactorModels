"""dimensionless_numbers.py"""


def reynolds_number(
    density: float,
    velocity: float,
    diameter: float,
    viscosity: float,
) -> float:
    """Reynolds number. (Re = density * velocity * diameter / viscosity)"""
    assert density > 0, f"density must be positive, got {density}"
    assert velocity > 0, f"velocity must be positive, got {velocity}"
    assert diameter > 0, f"diameter must be positive, got {diameter}"
    assert viscosity > 0, f"viscosity must be positive, got {viscosity}"

    return density * velocity * diameter / viscosity


def schmidt_number(
    viscosity: float,
    density: float,
    diffusion_coefficient: float,
) -> float:
    """Schmidt number. (Sc = viscosity / (density * diffusion_coefficient))"""
    assert viscosity > 0, f"viscosity must be positive, got {viscosity}"
    assert density > 0, f"density must be positive, got {density}"
    assert (
        diffusion_coefficient > 0
    ), f"diffusion_coefficient must be positive, got {diffusion_coefficient}"

    return viscosity / (density * diffusion_coefficient)


def peclet_number(
    velocity: float,
    length: float,
    diffusion_coefficient: float,
) -> float:
    """Peclet number. (Pe = velocity * length / diffusion_coefficient)"""
    assert velocity > 0, f"velocity must be positive, got {velocity}"
    assert length > 0, f"length must be positive, got {length}"
    assert (
        diffusion_coefficient > 0
    ), f"diffusion_coefficient must be positive, got {diffusion_coefficient}"

    return velocity * length / diffusion_coefficient


def sherwood_number(
    mass_transfer_coefficient: float,
    diameter: float,
    diffusion_coefficient: float,
) -> float:
    """Sherwood number. (Sh = mass_transfer_coefficient * diameter / diffusion_coefficient)"""  # noqa: E501
    assert mass_transfer_coefficient > 0, (
        f"mass_transfer_coefficient must be positive, "
        f"got {mass_transfer_coefficient}"
    )
    assert diameter > 0, f"diameter must be positive, got {diameter}"
    assert (
        diffusion_coefficient > 0
    ), f"diffusion_coefficient must be positive, got {diffusion_coefficient}"

    return mass_transfer_coefficient * diameter / diffusion_coefficient


def stanton_number(
    mass_transfer_coefficient: float,
    velocity: float,
) -> float:
    """Stanton number. (St = mass_transfer_coefficient / velocity)"""
    assert mass_transfer_coefficient > 0, (
        f"mass_transfer_coefficient must be positive, "
        f"got {mass_transfer_coefficient}"
    )
    assert velocity > 0, f"velocity must be positive, got {velocity}"

    return mass_transfer_coefficient / velocity
