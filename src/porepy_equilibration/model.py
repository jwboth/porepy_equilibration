from porepy_equilibration.porepy.fluid_mass_balance import (
    SinglePhaseFlow as SinglePhaseFlowWithReference,
)
from porepy_equilibration.porepy.momentum_balance import (
    MomentumBalance as MomentumBalanceWithReference,
)
from porepy_equilibration.porepy.poromechanics import (
    Poromechanics as PoromechanicsWithReference,
)
from porepy_equilibration.porepy.solution_strategy import (
    NewtonReferenceUpdateStrategy as UpdateStrategy,
)
from porepy.viz.data_saving_model_mixin import FractureDeformationExporting

import porepy as pp

from .auxiliary import AuxiliaryContact
from .contact import AlartCurnierContact
from .fracture_states import FractureStates


class CommonIngredients(
    AuxiliaryContact,  # Yield function, orthognality, and alignment
    FractureStates,  # physics-based conact states and their logging
    FractureDeformationExporting,
    pp.constitutive_laws.CharacteristicDisplacementFromTraction,
):
    """Common ingredients for mechanics and poromechanics models."""


class MechanicsModel(
    AlartCurnierContact,
    CommonIngredients,
    pp.momentum_balance.MomentumBalance,
):
    """Base mechanics model."""


class MechanicsModelWithReference(
    AlartCurnierContact,
    CommonIngredients,
    UpdateStrategy,
    MomentumBalanceWithReference,
):
    """Base mechanics model with reference state."""


class PoromechanicsModel(
    # AlartCurnierContact,
    CommonIngredients,
    pp.constitutive_laws.CubicLawPermeability,
    pp.poromechanics.Poromechanics,
):
    """Base poromechanics model."""


class PoromechanicsModelWithReference(
    # AlartCurnierContact,
    CommonIngredients,
    pp.constitutive_laws.CubicLawPermeability,
    UpdateStrategy,
    PoromechanicsWithReference,
):
    """Base mechanics model with reference."""


class FlowModel(
    pp.SinglePhaseFlow,
):
    """Base flow model."""


class FlowModelWithReference(
    UpdateStrategy,
    SinglePhaseFlowWithReference,
):
    """Base flow model with reference state."""
