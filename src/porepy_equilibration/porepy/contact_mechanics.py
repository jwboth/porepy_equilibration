"""Momentum balance with reference state."""

import logging
from typing import Callable, cast

import numpy as np

import porepy as pp

from . import constitutive_laws

logger = logging.getLogger(__name__)


class DataSavingContactMechanics:
    """Auxiliary class for storing contact tractions with reference state."""

    def data_to_export(self):
        """Add reference state to export data."""
        data = super().data_to_export()

        # Get characteristic traction for scaling to Pa
        sds = self.mdg.subdomains(dim=self.nd - 1)
        if len(sds) > 0:
            char = self.evaluate_and_scale(sds, "characteristic_contact_traction", "Pa")
            cell_offsets = np.cumsum([0] + [sd.num_cells for sd in sds])

            # Ensure characteristic traction is an array
            size = sum([sd.num_cells for sd in sds])
            if isinstance(char, float):
                char = char * np.ones(size)

        # Add variants of contact tractions
        for id, sd in enumerate(sds):
            # Reference contact traction (dimensionless)
            contact_traction_ref = self.reference_contact_traction([sd]).value(
                self.equation_system
            )
            data.append(
                (sd, self.contact_traction_variable + "_ref", contact_traction_ref)
            )

            # Delta contact traction (dimensionless)
            contact_traction_inc = self.delta_contact_traction([sd]).value(
                self.equation_system
            )
            data.append(
                (sd, self.contact_traction_variable + "_inc", contact_traction_inc)
            )

            # Find and remove the original (sd, contact_traction_variable) entry
            data = [
                entry
                for entry in data
                if not (entry[0] == sd and entry[1] == self.contact_traction_variable)
            ]
            # Full contact traction (dimensionless)
            contact_traction = self.contact_traction([sd]).value(self.equation_system)
            data.append((sd, self.contact_traction_variable, contact_traction))

            # Add Pa-scaled variants
            if len(sds) > 0:
                # Reference traction in Pa
                traction_ref_pa = contact_traction_ref.reshape((self.nd, -1), order="F")
                traction_ref_pa *= char[cell_offsets[id] : cell_offsets[id + 1]]
                data.append(
                    (
                        sd,
                        self.contact_traction_variable + "_ref_in_Pa",
                        traction_ref_pa.ravel("F"),
                    )
                )

                # Delta traction in Pa
                traction_inc_pa = contact_traction_inc.reshape((self.nd, -1), order="F")
                traction_inc_pa *= char[cell_offsets[id] : cell_offsets[id + 1]]
                data.append(
                    (
                        sd,
                        self.contact_traction_variable + "_inc_in_Pa",
                        traction_inc_pa.ravel("F"),
                    )
                )

                # Full traction in Pa
                traction_pa = contact_traction.reshape((self.nd, -1), order="F")
                traction_pa *= char[cell_offsets[id] : cell_offsets[id + 1]]
                data.append(
                    (
                        sd,
                        self.contact_traction_variable + "_in_Pa",
                        traction_pa.ravel("F"),
                    )
                )

        return data


class ContactTractionVariable(pp.contact_mechanics.ContactTractionVariable):
    """Contact traction variable for contact mechanics."""

    contact_traction_variable: str
    """Name of the primary variable representing the contact traction on a fracture
    subdomain. Normally defined in a mixin of instance
    :class:`~porepy.models.contact_mechanics.SolutionStrategyContactMechanics`.

    """
    set_reference_tractions = False

    def reference_contact_traction(
        self, domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        """Reference displacement in the matrix."""

        # Initalize if required.
        self.init_reference_tractions()

        return pp.ad.TimeDependentDenseArray(
            name=f"reference_" + self.contact_traction_variable,
            domains=domains,
        )

    def init_reference_tractions(self) -> None:
        """Initialization of reference tractions."""

        # Employ a hack to initialize the reference state before it is used,
        # since the reference state is needed in the creation of the variables,
        # but the initialization of the reference state needs to be done after the
        # variables are created. This is a bit of a circular dependency, but it works
        # in practice. The alternative would be to separate the creation of the
        # variables and the initialization of the reference state into two separate
        # steps, but that would be more cumbersome to use.
        # Make sure to only initialize once.
        if self.set_reference_tractions:
            return
        self.set_reference_tractions = True

        # Set reference state for matrix displacement
        for sd, data in self.mdg.subdomains(return_data=True, dim=self.nd - 1):
            for index in [{"iterate_index": 0}, {"time_step_index": 0}]:
                pp.set_solution_values(
                    name="reference_" + self.contact_traction_variable,
                    values=np.zeros(self.nd * sd.num_cells),
                    data=data,
                    **index,
                )

        # Write logging info in yellow color
        logger.info("\033[93m" + "Initialized reference traction to zero." + "\033[0m")

    def update_reference(self) -> None:
        """Updating of reference tractions."""

        # If super has an update_reference method, call it for compatibility with multi-physics.
        if hasattr(super(), "update_reference"):
            super().update_reference()

        for sd, data in self.mdg.subdomains(return_data=True, dim=self.nd - 1):
            for index in [{"iterate_index": 0}, {"time_step_index": 0}]:
                for name, values in [
                    (
                        "reference_" + self.contact_traction_variable,
                        self.contact_traction([sd]).value(self.equation_system),
                    ),
                    (
                        self.contact_traction_variable,
                        np.zeros(self.nd * sd.num_cells),
                    ),
                ]:
                    pp.set_solution_values(
                        name=name,
                        values=values,
                        data=data,
                        **index,
                    )

        # Write logging info in yellow color
        logger.info("\033[93m" + "Updated reference state." + "\033[0m")

    def delta_contact_traction(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Fracture contact traction increment [-].

        Parameters:
            subdomains: List of subdomains where the contact traction is defined. Should
                be of co-dimension one, i.e. fractures.

        Returns:
            Variable for nondimensionalized fracture contact traction.

        """
        # Check that the subdomains are fractures
        for sd in subdomains:
            if sd.dim != self.nd - 1:
                raise ValueError("Contact traction only defined on fractures")

        return self.equation_system.md_variable(
            self.contact_traction_variable, subdomains
        )

    def contact_traction(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Fracture contact traction [-].

        Parameters:
            subdomains: List of subdomains where the contact traction is defined. Should
                be of co-dimension one, i.e. fractures.

        Returns:
            Variable for nondimensionalized fracture contact traction.

        """
        return self.reference_contact_traction(
            subdomains
        ) + self.delta_contact_traction(subdomains)


class InitialConditionsContactTraction(pp.InitialConditionMixin):
    """Mixin for providing initial values for contact traction.

    Similiar to :class:`pp.contact_mechanics.InitialConditionsContactTraction`, but for
    redirecting the initial condition towards the contact traction increment.

    """

    delta_contact_traction: Callable[[list[pp.Grid]], pp.ad.Operator]
    """See :class:`ContactTractionVariable`."""

    def set_initial_values_primary_variables(self) -> None:
        """Method to set initial values for contact traction."""
        # Super call for compatibility with multi-physics.
        super().set_initial_values_primary_variables()

        for sd in self.mdg.subdomains(dim=self.nd - 1):
            self.equation_system.set_variable_values(
                self.ic_values_delta_contact_traction(sd),
                [cast(pp.ad.Variable, self.delta_contact_traction([sd]))],
                iterate_index=0,
            )

    def ic_values_delta_contact_traction(self, sd: pp.Grid) -> np.ndarray:
        """Fetch initial values for contact traction.

        Note: This is only consistent with zero reference states.

        """
        if hasattr(self, "ic_values_contact_traction"):
            return self.ic_values_contact_traction(sd)
        else:
            return pp.contact_mechanics.InitialConditionsContactTraction.ic_values_contact_traction(
                self, sd
            )


# Use the same design pattern as for pp.MomentumBalance. Mainly keep the equations,
# boundary conditions, and some constitutive laws and overwrite aperture specific
# laws in addition to new variables. They are updated via solution strategies.
class ContactMechanics(
    DataSavingContactMechanics,
    pp.contact_mechanics.ContactMechanicsEquations,
    ContactTractionVariable,
    constitutive_laws.DisplacementJump,
    pp.contact_mechanics.ConstitutiveLawsContactMechanics,
    InitialConditionsContactTraction,
    pp.contact_mechanics.SolutionStrategyContactMechanics,
    pp.FluidMixin,
    pp.ModelGeometry,
    pp.DataSavingMixin,
):
    """Class for mixed-dimensional momentum balance with contact mechanics."""
