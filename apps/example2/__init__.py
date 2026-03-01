"""2D Salt Cove outcrop simulation under compression."""

from __future__ import annotations

import argparse
import porepy as pp
from porepy.applications.material_values.fluid_values import water
from porepy_equilibration.geometry import SaltCoveVertical as Geometry
from porepy_equilibration.gravity import GradualGravity
from porepy_equilibration.gravity import Gravity
from porepy_equilibration.model import PoromechanicsModel as PoromechanicsNoRef
from porepy_equilibration.model import (
    PoromechanicsModelWithReference as PoromechanicsWithRef,
)
from porepy_equilibration.solver_statistics import CustomSolverStatistics


class InitMech:
    """Mixin to constrain pressure in flow equations."""

    def after_nonlinear_convergence(self) -> None:
        """Define indicator for constraining flow equations."""
        super().after_nonlinear_convergence()
        self.single_phase_flow_constraint_indicator.set_value(0.0)

    def mass_balance_equation(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Overwrites mass balance equation to constrain the pressure."""
        if not hasattr(self, "single_phase_flow_constraint_indicator"):
            self.single_phase_flow_constraint_indicator = pp.ad.Scalar(1.0)
        eq = super().mass_balance_equation(subdomains)
        constrained_eq = self.pressure(subdomains) - self.initial_pressure(subdomains)
        indicator = self.single_phase_flow_constraint_indicator
        combined_eq = eq + indicator * (constrained_eq - eq)
        combined_eq.set_name(
            pp.fluid_mass_balance.FluidMassBalanceEquations.primary_equation_name()
        )
        return combined_eq


def create_model_class(use_reference_states: bool, use_gradual_bc: bool) -> type:
    """Factory function to create model class with selected physics.

    Parameters:
        use_reference_states: If True, uses PoromechanicsWithRef;
                              if False, uses PoromechanicsNoRef.

    Returns:
        A model class combining geometry, gravity, and selected physics.
    """
    if use_reference_states:
        physics_class = PoromechanicsWithRef
    else:
        physics_class = PoromechanicsNoRef

    if use_gradual_bc:
        gravity_class = GradualGravity
    else:
        gravity_class = Gravity

    class Model(
        InitMech, Geometry, CustomSolverStatistics, gravity_class, physics_class
    ):
        """Model for 2D Salt Cove outcrop under gravity."""

        pass

    return Model


def create_folder_name(use_reference_states: bool, use_gradual_bc: bool) -> str:
    """Create folder name based on selected options."""
    ref_str = "with_ref" if use_reference_states else "no_ref"
    bc_str = "gradual_bc" if use_gradual_bc else "instant_bc"
    return f"output/example2_{ref_str}_{bc_str}"


def define_model_params(use_reference_states: bool, use_gradual_bc: bool) -> dict:
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
        "solver_statistics_file_name": "solver_statistics.json",
        "folder_name": create_folder_name(use_reference_states, use_gradual_bc),
    }


def define_solver_params() -> dict:
    """Define solver parameters."""
    return {"nl_max_iterations": 100}


def main():
    """Run single simulation with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description="2D Salt Cove outcrop simulation under gravity.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  example2 --with-reference-states\n"
            "  example2 --without-reference-states"
        ),
    )
    parser.add_argument(
        "--with-reference-states",
        action="store_true",
        default=False,
        help="Use physics model with reference states.",
    )
    parser.add_argument(
        "--without-reference-states",
        action="store_true",
        default=False,
        help="Use physics model without reference states.",
    )
    parser.add_argument(
        "--gradual-bc",
        action="store_true",
        default=False,
        help="Use gradual background stress boundary condition.",
    )
    parser.add_argument(
        "--instant-bc",
        action="store_true",
        default=False,
        help="Use instant background stress boundary condition.",
    )

    args = parser.parse_args()

    # Validate that exactly one flag is provided
    if not (args.with_reference_states + args.without_reference_states) == 1:
        parser.error(
            "Must specify either --with-reference-states or --without-reference-states"
        )
    if not (args.gradual_bc + args.instant_bc) == 1:
        parser.error("Must specify either --gradual-bc or --instant-bc")

    use_reference_states = args.with_reference_states
    use_gradual_bc = args.gradual_bc

    # Create model and run
    model_class = create_model_class(use_reference_states, use_gradual_bc)
    model_params = define_model_params(use_reference_states, use_gradual_bc)
    model = model_class(model_params)

    solver_params = define_solver_params()
    pp.run_time_dependent_model(model, solver_params)

    print("Simulation completed.")
    print("Solver statistics:")
    print(model.nonlinear_solver_statistics.num_iterations_history)


if __name__ == "__main__":
    main()
