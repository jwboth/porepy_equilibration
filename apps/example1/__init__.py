"""Elastic medium with random network under compression."""

from __future__ import annotations

from pathlib import Path
import porepy as pp

from ncp_mechanics.presets.base_params import BaseParameters, BaseParameterRange
from ncp_mechanics.presets.logging import print_stats
from ncp_mechanics.presets.parameter_test import run_parameter_test

from ncp_mechanics.geometry.dim_2d import SquareWithImportedFractures as Geometry
from ncp_mechanics.geometry.dim_2d import SquareWithImportedFracturesConfig as GeometryConfig

from ncp_mechanics.bc.tensile_background_stress_2d import TensileBackgroundStress2D as BC
from ncp_mechanics.bc.tensile_background_stress_2d import TensileBackgroundStress2DConfig as BCConfig
from ncp_mechanics.model.mechanics_model import (
    MechanicsModel,
    MechanicsModelWithReference,
    MechanicsModelConfig,
)
from ncp.formulations.utils import ContactConfig

from ncp_mechanics.common import DefaultParameters
from ncp_mechanics.presets.abbreviations import short, elastic_folder_name
from ncp_mechanics.common.utils import add_windows_long_path_prefix

from dataclasses import dataclass, field
from itertools import product
import numpy as np

description = """Run 2D Utah Forge (Salt Cove) simulation (compression).\n\n"""

from porepy.viz.data_saving_model_mixin import IterationExporting


class Model(IterationExporting, Geometry, BC, MechanicsModelWithReference):
    """Mechanics model for under unidirectional compression of 2D Salt Cove outcrop."""


# class Model(IterationExporting, Geometry, BC, MechanicsModel):
#    """Mechanics model for under unidirectional compression of 2D Salt Cove outcrop."""


class CaseParameterConfig:
    """Overwrite default parameters in the set config classes:

    - Mechanical material parameters
    - Numerical material parameters

    NOTE: Parameters from Both et al. 2025 (EGC proceeding paper).

    """

    @classmethod
    def default_config(cls) -> dict[str, int | float | bool]:
        config = {}

        # Overwrite BCConfig parameters
        config.update(
            {
                "bc": {
                    "background_stress": [-0.2 * pp.MEGA, -2 * pp.MEGA],
                },
            }
        )

        # Overwrite GeometryConfig
        config.update(
            {
                "geometry": {
                    "csv_geometry_file": Path(__file__).parent
                    / "geometries"
                    / "network_outcrop_salt_cove_1.csv",
                    "domain_size": 1000.0,
                    "cell_size": 30.0,
                    "cell_size_fracture": 1,
                },
            }
        )

        # Overwrite ContactConfig
        ...

        # Overwrite MechanicsModelConfig
        config.update(
            {
                "solid_constants": {
                    # Guessed
                    "dilation_angle": 0.1,  # guessed
                    # Literature values
                    "biot_coefficient": 1,  # guessed by Vaezi et al.
                    "permeability": 4.35e-6 * pp.DARCY,  # X.Ma et al.
                    "normal_permeability": 4.35e-6 * pp.DARCY,  # X.Ma et al.
                    "residual_aperture": 1e-4,  # Computed from transmissivities (X. Ma et al.) and cubic law
                    "porosity": 1.36e-2,  # X.Ma et al.
                    "shear_modulus": 16.8 * pp.GIGA,  # X.Ma et al.
                    "lame_lambda": 19.73 * pp.GIGA,  # X.Ma et al.
                    "density": 2653,  # X.Ma et al.
                    "friction_coefficient": 0.6,  # X.Ma et al.
                    # Well
                    "well_radius": 0.10,  # m
                },
                "numerics": {
                    "open_state_tolerance": 1e-10,  # Numerical method parameter
                    "characteristic_contact_traction": np.nan,  # See BC - requires to overwrite!
                },
                "units": {
                    "kg": 16.8 * pp.GIGA,  # see Young's/shear modulus
                    "m": 1.0,
                    "s": 1.0,
                    "rad": 1.0,
                },
            }
        )

        # Overwrite DefaultParameters
        config.update(
            {
                "max_iterations": 100,
                "time": {
                    "end_time": 10.0,
                    "dt": 1.0,  # Effectively single step
                    "constant_dt": True,
                    "dt_min_max": [0.1, 1.0],
                    "iter_max": 15,
                    "iter_optimal_range": [5, 10],
                    "iter_relax_factors": (0.7, 1.3),
                },
            }
        )

        return config


class Parameters(BaseParameters):
    """Default model parameters.

    This class:
    - Aggregates default configs from PhysicsConfig and GeometryConfig
    - Registers Config classes for parsing via config_classes()
    - Provides custom folder naming via _make_folder_name()

    """

    @classmethod
    def default_config(cls) -> dict:
        """Combine physics and geometry default configs."""
        config = {}
        for klass in reversed(Parameters.config_classes()):
            config = BaseParameters._deep_update(config, klass.default_config())
        return config

    @staticmethod
    def config_classes() -> list[type]:
        """Return Config classes that handle parsing.

        Each class must have a parse_model_params_from_config classmethod.
        Order matters: Use mixin logic.

        Returns:
            list[type]: List of Config classes.

        """
        return [
            CaseParameterConfig,  # Overwrite default physics parameters
            BCConfig,  # Handles BC params (if any)
            GeometryConfig,  # Handles geometry params
            ContactConfig,  # Handles contact params
            MechanicsModelConfig,  # Handles BC params (if any)
            DefaultParameters,  # Handles time, materials, units
        ]

    def _make_folder_name(self) -> Path:
        """Generate descriptive folder name based on config."""
        base = self.base_path if self.base_path else Path.cwd()
        # Physics
        physics_folder = elastic_folder_name(self.config)
        # BC
        bc_folder = BC.short(self.config["bc"])
        # Numerics
        numerics_folder = f"cnum_{self.config['numerics']['characteristic_contact_traction']:.1e}"
        # Geometry
        geo_folder = Geometry.short(self.config["geometry"])
        # Contact
        normal_formulation = self.config["contact"]["normal_formulation"]
        tangential_formulation = self.config["contact"]["tangential_formulation"]
        contact_folder = f"{short(normal_formulation)}_{short(tangential_formulation)}"

        path = base / physics_folder / bc_folder / numerics_folder / geo_folder / contact_folder
        long_path = add_windows_long_path_prefix(path.resolve())
        return long_path


@dataclass
class ParameterRange(BaseParameterRange):
    """Defines ranges for parameter sweeps.

    Each attribute can be a single value or a list of values to iterate over.

    Example (identical to the default):

    parameter_range = ParameterRange(
        friction_coefficient=[0.6],
        dilation_angle=[0.0, 0.1],
        contact_formulation=[
            ("alart_curnier", "alart_curnier"),
            ("alart_curnier", "hueber"),
            ("bipotential_orthogonal_return", "bipotential_orthogonal_return"),
            ("ncp_min", "ncp_min"),
            ("ncp_fb", "ncp_fb"),
            ("alart_curnier", "constant_scaled_alart_curnier"),
            ("alart_curnier", "random_scaled_alart_curnier"),
            ("constant_scaled_alart_curnier", "constant_scaled_alart_curnier"),
            ("random_scaled_alart_curnier", "random_scaled_alart_curnier"),
            ("alart_curnier", "random_scaled_alart_curnier"),
            ("random_weighted_return", "random_weighted_return"),
        ],
        num_fractures=[[i, i] for i in range(1, 11)],
        fracture_seed=[i for i in range(0, 20)],
        use_processed_fracture_network=[False, True],
    )

    """

    # Contact formulations: list of (normal, tangential) tuples
    contact_formulation: list[tuple[str, str]] = field(
        default_factory=lambda: [
            ("alart_curnier", "alart_curnier"),
            ("alart_curnier", "hueber"),
            ("bipotential_orthogonal_return", "bipotential_orthogonal_return"),
            ("ncp_min", "ncp_min"),
            ("ncp_fb", "ncp_fb"),
            ("alart_curnier", "constant_scaled_alart_curnier"),
            ("alart_curnier", "random_scaled_alart_curnier"),
            ("constant_scaled_alart_curnier", "constant_scaled_alart_curnier"),
            ("random_scaled_alart_curnier", "random_scaled_alart_curnier"),
            ("alart_curnier", "random_scaled_alart_curnier"),
            ("random_weighted_return", "random_weighted_return"),
        ]
    )

    # Solid constants
    friction_coefficient: list[float] = field(default_factory=lambda: [0.6])
    dilation_angle: list[float] = field(default_factory=lambda: [])

    # BC parameters
    background_stress: list[list[float]] = field(
        default_factory=lambda: [
            [-0.2 * pp.MEGA, -2 * pp.MEGA],
            [-0.2 * pp.MEGA, -20 * pp.MEGA],
            # [-0.2 * pp.MEGA, -0.2 * pp.MEGA],
            # [-0.2 * pp.MEGA, -200 * pp.MEGA],
        ]
    )

    # Numerical parameters
    characteristic_contact_traction: list[float] = field(
        default_factory=lambda: [0.1 * pp.MEGA, 1 * pp.MEGA, 10 * pp.MEGA]
    )

    def __iter__(self):
        """Iterate over all parameter combinations."""
        # Get all parameter names and their values
        params = {
            "contact_formulation": self.contact_formulation,
            "friction_coefficient": self.friction_coefficient,
            "dilation_angle": self.dilation_angle,
            "background_stress": self.background_stress,
            "characteristic_contact_traction": self.characteristic_contact_traction,
        }

        # Generate cartesian product of all parameter values
        keys = list(params.keys())
        values = list(params.values())

        for combo in product(*values):
            yield dict(zip(keys, combo))

    def __len__(self):
        """Return total number of parameter combinations."""
        return (
            len(self.contact_formulation)
            * len(self.friction_coefficient)
            * len(self.dilation_angle)
            * len(self.background_stress)
            * len(self.characteristic_contact_traction)
        )

    def to_config_update(self, combo: dict) -> dict:
        """Convert a parameter combination to a config update dict."""
        return {
            "contact": {
                "normal_formulation": combo["contact_formulation"][0],
                "tangential_formulation": combo["contact_formulation"][1],
            },
            "solid_constants": {
                "friction_coefficient": combo["friction_coefficient"],
                "dilation_angle": combo["dilation_angle"],
            },
            "bc": {
                "background_stress": combo["background_stress"],
            },
            "numerics": {
                "characteristic_contact_traction": combo["characteristic_contact_traction"],
            },
        }

    @classmethod
    def from_config(cls, config: dict, overwrite: bool = False) -> ParameterRange:
        """Create ParameterRange from (possibly incomplete) config dict.

        Can be used to create a ParameterRange instance from toml.

        Example toml:

        [contact]
        formulation = [("alart_curnier", "alart_curnier"), ("ncp_min", "ncp_min")]

        [solid_constants]
        friction_coefficient = [0.4, 0.6, 0.8]
        dilation_angle = [0.0, 0.1]

        Args:
            config (dict): Parsed TOML configuration dictionary.

        Returns:
            ParameterRange: Instance with parameters from config or defaults.

        """

        # Helper to get nested value with fallback and overwrite logic
        def get_nested(cfg, keys, default, overwrite):
            d = cfg
            for k in keys[:-1]:
                d = d.get(k, {}) if isinstance(d, dict) else {}
            if isinstance(d, dict):
                if overwrite and keys[-1] in d:
                    return d[keys[-1]]
                if not overwrite:
                    return d.get(keys[-1], default)
            return default

        default_range = cls()

        return cls(
            contact_formulation=get_nested(
                config,
                ["contact", "formulation"],
                default_range.contact_formulation,
                overwrite,
            ),
            friction_coefficient=get_nested(
                config,
                ["solid_constants", "friction_coefficient"],
                default_range.friction_coefficient,
                overwrite,
            ),
            dilation_angle=get_nested(
                config,
                ["solid_constants", "dilation_angle"],
                default_range.dilation_angle,
                overwrite,
            ),
            background_stress=get_nested(
                config,
                ["bc", "background_stress"],
                default_range.background_stress,
                overwrite,
            ),
            characteristic_contact_traction=get_nested(
                config,
                ["numerics", "characteristic_contact_traction"],
                default_range.characteristic_contact_traction,
                overwrite,
            ),
        )


def main():
    run_parameter_test(
        Model=Model,
        Parameters=Parameters,
        ParameterRange=ParameterRange,
        print_stats=print_stats,
        base_path=Path(__file__).parent,
        description=description,
    )
