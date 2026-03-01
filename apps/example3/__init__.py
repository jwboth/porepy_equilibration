"""2D Salt Cove outcrop simulation under compression."""

from __future__ import annotations

import porepy as pp
from porepy.applications.material_values.fluid_values import water
from porepy.viz.data_saving_model_mixin import IterationExporting
from porepy_equilibration.geometry import SaltCoveVertical as Geometry
from porepy_equilibration.gravity import Gravity
# from porepy_equilibration.model import FlowModel as Physics
from porepy_equilibration.model import FlowModelWithReference as Physics


class Model(
    # IterationExporting,
    Geometry,
    Gravity,
    Physics,
):
    """Model for the 2D Salt Cove outcrop simulation under gravity, 3000m underground."""


def define_model_params():
    return {
        "csv_file": "salt_cove_fractures.csv",
        "material_constants": {
            "solid": pp.SolidConstants(
                **{
                    "dilation_angle": 0.1,
                    "biot_coefficient": 1.0,
                    "permeability": 4.35e-6 * pp.DARCY,
                    "normal_permeability": 4.35e-6 * pp.DARCY,
                    "residual_aperture": 1e-4,
                    "porosity": 1.36e-2,
                    "shear_modulus": 16.8 * pp.GIGA,
                    "lame_lambda": 19.73 * pp.GIGA,
                    "density": 2653,
                    "friction_coefficient": 0.6,
                    # "well_radius": 0.10,
                }
            ),
            "fluid": pp.FluidComponent(**water),
            "numerical": pp.NumericalConstants(
                **{
                    "open_state_tolerance": 1e-10,
                    "characteristic_contact_traction": 1.0 * pp.MEGA,
                }
            ),
        },
        "time_manager": pp.TimeManager(
            schedule=[0.0, 100.0 * pp.YEAR],
            dt_init=1.0 * pp.YEAR,
            constant_dt=False,
            dt_min_max=[0.1 * pp.YEAR, 100.0 * pp.YEAR],
            iter_optimal_range=[10, 20],
            iter_relax_factors=(0.7, 5.0),
            iter_max=100,
        ),
        "units": pp.Units(kg=16.8 * pp.GIGA, m=1.0, s=1.0),
        "solver_statistics_file": "solver_statistics.json",
        "folder_name": "output/example3",
    }


def define_solver_params():
    return {"nl_max_iterations": 100}


def main():
    """Run single simulation."""
    model_params = define_model_params()
    model = Model(model_params)
    solver_params = define_solver_params()
    pp.run_time_dependent_model(model, solver_params)
    print("Simulation completed.")
    print("Solver statistics:")
    print(model.nonlinear_solver_statistics.num_iterations_history)


if __name__ == "__main__":
    main()
