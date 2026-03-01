"""2D Salt Cove outcrop simulation under compression."""

from __future__ import annotations

import argparse
import porepy as pp
from porepy_equilibration.bc import GradualTensileBackgroundStress2D as BC1
from porepy_equilibration.bc import TensileBackgroundStress2D as BC2
from porepy_equilibration.geometry import SaltCove as Geometry
from porepy_equilibration.model import MechanicsModel as PhysicsNoRef
from porepy_equilibration.model import MechanicsModelWithReference as PhysicsRef


def create_model_class(use_reference: bool, use_gradual_bc: bool) -> type:
    """Factory function to create model class with selected BC.

    Parameters:
        use_gradual_bc: If True, uses GradualTensileBackgroundStress2D;
                        if False, uses TensileBackgroundStress2D.

    Returns:
        A model class combining geometry, BC, and physics.
    """
    bc_class = BC1 if use_gradual_bc else BC2

    physics_class = PhysicsRef if use_reference else PhysicsNoRef

    class Model(Geometry, bc_class, physics_class):
        """Model for 2D Salt Cove outcrop under compression."""

        pass

    return Model


def create_folder_name(use_reference: bool, use_gradual_bc: bool) -> str:
    """Create folder name based on selected options."""
    ref_str = "with_ref" if use_reference else "no_ref"
    bc_str = "gradual_bc" if use_gradual_bc else "instant_bc"
    return f"output/example1_{ref_str}_{bc_str}"


def define_model_params(use_reference: bool, use_gradual_bc: bool) -> dict:
    """Define hardcoded model parameters."""
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
                }
            ),
            "fluid": pp.FluidComponent(*{}),
            "numerical": pp.NumericalConstants(
                **{
                    "open_state_tolerance": 1e-10,
                    "characteristic_contact_traction": 1.0 * pp.MEGA,
                }
            ),
        },
        "bc": {
            "background_stress": [-0.2 * pp.MEGA, -2 * pp.MEGA],
        },
        "time_manager": pp.TimeManager(
            schedule=[0.0, 10.0],
            dt_init=1.0,
            constant_dt=True,
            dt_min_max=[0.1, 1.0],
            iter_optimal_range=[8, 20],
            iter_relax_factors=(0.7, 1.3),
            iter_max=100,
        ),
        "units": pp.Units(kg=16.8 * pp.GIGA, m=1.0, s=1.0),
        "solver_statistics_file_name": "solver_statistics.json",
        "folder_name": create_folder_name(use_reference, use_gradual_bc),
    }


def define_solver_params() -> dict:
    """Define solver parameters."""
    return {"nl_max_iterations": 100}


def main():
    """Run single simulation with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description="2D Salt Cove outcrop simulation under compression.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=("Examples:\n  example1 --gradual-bc\n  example1 --instant-bc"),
    )
    parser.add_argument(
        "--with-reference-states",
        action="store_true",
        default=False,
        help="Use reference states in the model.",
    )
    parser.add_argument(
        "--without-reference-states",
        action="store_true",
        default=False,
        help="Do not use reference states in the model.",
    )
    parser.add_argument(
        "--gradual-bc",
        action="store_true",
        default=False,
        help="Use gradual tensile background stress BC.",
    )
    parser.add_argument(
        "--instant-bc",
        action="store_true",
        default=False,
        help="Use instant tensile background stress BC.",
    )

    args = parser.parse_args()

    # Validate that exactly one flag is provided
    if not (args.with_reference_states + args.without_reference_states) == 1:
        parser.error(
            "Must specify either --with-reference-states or --without-reference-states"
        )
    if not (args.gradual_bc + args.instant_bc) == 1:
        parser.error("Must specify either --gradual-bc or --instant-bc")

    use_reference = args.with_reference_states
    use_gradual_bc = args.gradual_bc

    # Create model and run
    model_class = create_model_class(use_reference, use_gradual_bc)
    model_params = define_model_params(use_reference, use_gradual_bc)
    model = model_class(model_params)

    solver_params = define_solver_params()
    pp.run_time_dependent_model(model, solver_params)

    print("Simulation completed.")
    print("Solver statistics:")
    print(model.nonlinear_solver_statistics.num_iterations_history)


if __name__ == "__main__":
    main()
