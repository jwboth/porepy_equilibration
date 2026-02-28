import porepy as pp
from porepy.viz.data_saving_model_mixin import (
    FractureDeformationExporting,
    IterationExporting,
)

from porepy.models.with_reference.momentum_balance import (
    MomentumBalance as MomentumBalanceWithReference,
)

from .auxiliary import AuxiliaryContact
from .fracture_states import FractureStates
from .contact import AlartCurnierContact


class MechanicsModel(
    AuxiliaryContact,  # Yield function, orthognality, and alignment
    FractureStates,  # physics-based conact states and their logging
    FractureDeformationExporting,
    IterationExporting,
    pp.constitutive_laws.CharacteristicDisplacementFromTraction,
    AlartCurnierContact,
    pp.momentum_balance.MomentumBalance,
):
    """Base mechanics model."""


class MechanicsModelWithReference(
    AuxiliaryContact,  # Yield function, orthognality, and alignment
    FractureDeformationExporting,
    FractureStates,  # physics-based conact states and their logging
    pp.constitutive_laws.CharacteristicDisplacementFromTraction,
    AlartCurnierContact,
    MomentumBalanceWithReference,
):
    """Base mechanics model with reference state."""
