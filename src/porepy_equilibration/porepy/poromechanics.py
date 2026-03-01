r"""Poromechanics with reference state."""

from __future__ import annotations

import porepy as pp

from . import (constitutive_laws, contact_mechanics, fluid_mass_balance,
               momentum_balance)


class ConstitutiveLawsPoromechanics(
    constitutive_laws.DisplacementJump,
    constitutive_laws.PoroMechanicsPorosity,
): ...


# Exclude classes that are used above as base classes in ConstitutiveLawsPoromechanics.
class ppConstitutiveLawsPoromechanics(
    # Combined effects
    pp.constitutive_laws.DisplacementJumpAperture,
    pp.constitutive_laws.BiotCoefficient,
    pp.constitutive_laws.PressureStress,
    pp.constitutive_laws.PoroMechanicsPorosity,
    # Fluid mass balance subproblem
    pp.constitutive_laws.ZeroGravityForce,
    pp.constitutive_laws.SecondOrderTensorUtils,
    pp.constitutive_laws.DarcysLaw,
    pp.constitutive_laws.DimensionReduction,
    pp.constitutive_laws.AdvectiveFlux,
    pp.constitutive_laws.FluidMobility,
    pp.constitutive_laws.PeacemanWellFlux,
    pp.constitutive_laws.ConstantPermeability,
    pp.constitutive_laws.FluidDensityFromPressure,
    pp.constitutive_laws.ConstantViscosity,
    # Mechanical subproblem
    pp.constitutive_laws.ElasticModuli,
    pp.constitutive_laws.CharacteristicTractionFromDisplacement,
    pp.constitutive_laws.ElasticTangentialFractureDeformation,
    pp.constitutive_laws.LinearElasticMechanicalStress,
    pp.constitutive_laws.ConstantSolidDensity,
    pp.constitutive_laws.FractureGap,
    pp.constitutive_laws.CoulombFrictionBound,
    # pp.constitutive_laws.DisplacementJump,
):
    """Class for combined constitutive laws for poromechanics."""

    def stress(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Stress operator.

        Parameters:
            subdomains: List of subdomains where the stress is defined.

        Returns:
            Operator for the stress.

        """
        # Method from constitutive library's LinearElasticRock.
        return self.mechanical_stress(subdomains) + self.pressure_stress(subdomains)


class VariablesPoromechanics(
    momentum_balance.VariablesMomentumBalance,
    fluid_mass_balance.VariablesSinglePhaseFlow,
    contact_mechanics.ContactTractionVariable,
): ...


class InitialConditionsPoromechanics(
    fluid_mass_balance.InitialConditionsSinglePhaseFlow,
    momentum_balance.InitialConditionsMomentumBalance,
    contact_mechanics.InitialConditionsContactTraction,
): ...


class DataSavingPoromechanics(
    fluid_mass_balance.DataSavingFluidMassBalance,
    momentum_balance.DataSavingMomentumBalance,
    contact_mechanics.DataSavingContactMechanics,
): ...


# Use the same design as for :class:`pp.Poromechanics`.
class Poromechanics(  # type: ignore[misc]
    DataSavingPoromechanics,
    pp.poromechanics.EquationsPoromechanics,
    VariablesPoromechanics,
    ConstitutiveLawsPoromechanics,
    ppConstitutiveLawsPoromechanics,
    # pp.poromechanics.ConstitutiveLawsPoromechanics,
    pp.poromechanics.BoundaryConditionsPoromechanics,
    InitialConditionsPoromechanics,
    pp.poromechanics.SolutionStrategyPoromechanics,
    pp.FluidMixin,
    pp.ModelGeometry,
    pp.DataSavingMixin,
): ...
