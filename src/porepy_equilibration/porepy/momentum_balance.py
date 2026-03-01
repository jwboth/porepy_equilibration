"""Momentum balance with reference state."""

import logging
from typing import Callable, Sequence, cast

import numpy as np

import porepy as pp

from . import constitutive_laws, contact_mechanics

logger = logging.getLogger(__name__)


class VariablesMomentumBalance(pp.momentum_balance.VariablesMomentumBalance):
    """Variables for mixed-dimensional deformation.

    NOTE: Do not inherit from :class:`VariablesMomentumBalance` to circumvent double
    definition.

    The variables are:
        - Displacement in matrix
        - Displacement on fracture-matrix interfaces

    """

    displacement_variable: str
    interface_displacement_variable: str
    set_reference_displacements: bool = False

    def reference_displacement(
        self, domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        """Reference displacement in the matrix."""

        # Initalize if required.
        self.init_reference_displacements()

        return pp.ad.TimeDependentDenseArray(
            name="reference_" + self.displacement_variable,
            domains=domains,
        )

    def reference_interface_displacement(
        self, interfaces: list[pp.MortarGrid]
    ) -> pp.ad.Operator:
        """Reference displacement on fracture-matrix interfaces."""

        # Initalize if required.
        self.init_reference_displacements()

        return pp.ad.TimeDependentDenseArray(
            name="reference_" + self.interface_displacement_variable,
            domains=interfaces,
        )

    def init_reference_displacements(self) -> None:
        """Initialization of reference displacements."""

        # Employ a hack to initialize the reference state before it is used,
        # since the reference state is needed in the creation of the variables,
        # but the initialization of the reference state needs to be done after the
        # variables are created. This is a bit of a circular dependency, but it works
        # in practice. The alternative would be to separate the creation of the
        # variables and the initialization of the reference state into two separate
        # steps, but that would be more cumbersome to use.
        # Make sure to only initialize once.
        if self.set_reference_displacements:
            return
        self.set_reference_displacements = True

        # Set reference state for matrix displacement
        for sd, data in self.mdg.subdomains(return_data=True, dim=self.nd):
            for index in [{"iterate_index": 0}, {"time_step_index": 0}]:
                pp.set_solution_values(
                    name="reference_" + self.displacement_variable,
                    values=np.zeros(self.nd * sd.num_cells),
                    data=data,
                    **index,
                )

        # Set reference state for interface displacement
        for intf, data in self.mdg.interfaces(return_data=True, dim=self.nd - 1):
            for index in [{"iterate_index": 0}, {"time_step_index": 0}]:
                pp.set_solution_values(
                    name="reference_" + self.interface_displacement_variable,
                    values=np.zeros(self.nd * intf.num_cells),
                    data=data,
                    **index,
                )

        # Write logging info in yellow color
        logger.info(
            "\033[93m" + "Initialized reference displacements to zero." + "\033[0m"
        )

    def update_reference(self) -> None:
        """Updating of reference displacements."""

        # If super class has an update reference method, call it for compatibility with multi-physics.
        if hasattr(super(), "update_reference"):
            super().update_reference()

        # Update reference state for matrix displacement
        for sd, data in self.mdg.subdomains(return_data=True, dim=self.nd):
            for index in [{"iterate_index": 0}, {"time_step_index": 0}]:
                for name, values in [
                    (
                        "reference_" + self.displacement_variable,
                        self.displacement([sd]).value(self.equation_system),
                    ),
                    (
                        self.displacement_variable,
                        np.zeros(self.nd * sd.num_cells),
                    ),
                ]:
                    pp.set_solution_values(
                        name=name,
                        values=values,
                        data=data,
                        **index,
                    )

        # Update reference state for interface displacement
        for intf, data in self.mdg.interfaces(return_data=True, dim=self.nd - 1):
            for index in [{"iterate_index": 0}, {"time_step_index": 0}]:
                for name, values in [
                    (
                        "reference_" + self.interface_displacement_variable,
                        self.interface_displacement([intf]).value(self.equation_system),
                    ),
                    (
                        self.interface_displacement_variable,
                        np.zeros(self.nd * intf.num_cells),
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

    def delta_displacement(self, domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        """Increment of displacement in the matrix."""
        if len(domains) == 0 or all(
            isinstance(grid, pp.BoundaryGrid) for grid in domains
        ):
            domains = cast(Sequence[pp.BoundaryGrid], domains)
            return self.create_boundary_operator(
                name=self.displacement_variable, domains=domains
            )
        # Check that the subdomains are grids
        if not all(isinstance(grid, pp.Grid) for grid in domains):
            raise ValueError(
                "Method called on a mixture of subdomain and boundary grids."
            )
        # Now we can cast to Grid
        domains = cast(list[pp.Grid], domains)

        if not all([grid.dim == self.nd for grid in domains]):
            raise ValueError(
                "Displacement is only defined in subdomains of dimension nd."
            )

        return self.equation_system.md_variable(self.displacement_variable, domains)

    def displacement(self, domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        """Displacement in the matrix.

        Parameters:
            domains: List of subdomains or interface grids where the displacement is
                defined. Should be the matrix subdomains.

        Returns:
            Variable for the displacement.

        """
        if not all([grid.dim == self.nd for grid in domains]):
            # Hack the boundary treatment
            return self.delta_displacement(domains)
        return self.reference_displacement(domains) + self.delta_displacement(domains)

    def delta_interface_displacement(
        self, interfaces: list[pp.MortarGrid]
    ) -> pp.ad.Variable:
        """Displacement on fracture-matrix interfaces.

        Parameters:
            interfaces: List of interface grids where the displacement is defined.
                Should be between the matrix and fractures.

        Returns:
            Variable for the displacement.

        Raises:
            ValueError: If the dimension of the interfaces is not equal to the ambient
                dimension of the problem minus one.

        """
        if not all([intf.dim == self.nd - 1 for intf in interfaces]):
            raise ValueError(
                "Interface displacement is only defined on interfaces of dimension "
                "nd - 1."
            )

        return self.equation_system.md_variable(
            self.interface_displacement_variable, interfaces
        )

    def interface_displacement(self, interfaces: list[pp.MortarGrid]) -> pp.ad.Operator:
        """Displacement on fracture-matrix interfaces.

        Parameters:
            interfaces: List of interface grids where the displacement is defined.
                Should be between the matrix and fractures.

        Returns:
            Variable for the displacement.

        Raises:
            ValueError: If the dimension of the interfaces is not equal to the ambient
                dimension of the problem minus one.

        """
        return self.reference_interface_displacement(
            interfaces
        ) + self.delta_interface_displacement(interfaces)


class InitialConditionsMomentumBalance(pp.InitialConditionMixin):
    """Mixin for providing initial values for displacement."""

    delta_displacement: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    """See :class:`VariablesMomentumBalance`."""

    delta_interface_displacement: Callable[[list[pp.MortarGrid]], pp.ad.Operator]
    """See :class:`VariablesMomentumBalance`."""

    def set_initial_values_primary_variables(self) -> None:
        """Method to set initial values for displacement, contact traction and interface
        displacement at iterate index 0 after the super-call.

        See also:

            - :meth:`ic_values_displacement`
            - :meth:`ic_values_interface_displacement`
            - :meth:`ic_values_contact_traction`

        """
        # Super call for compatibility with multi-physics.
        super().set_initial_values_primary_variables()

        for sd in self.mdg.subdomains():
            # Displacement is only defined on grids with ambient dimension.
            if sd.dim == self.nd:
                # Need to cast the return value to variable, because it is typed as
                # operator.
                self.equation_system.set_variable_values(
                    self.ic_values_delta_displacement(sd),
                    [cast(pp.ad.Variable, self.delta_displacement([sd]))],
                    iterate_index=0,
                )

        # interface dispacement is only defined on fractures with codimension 1
        for intf in self.mdg.interfaces(dim=self.nd - 1, codim=1):
            self.equation_system.set_variable_values(
                self.ic_values_delta_interface_displacement(intf),
                [cast(pp.ad.Variable, self.delta_interface_displacement([intf]))],
                iterate_index=0,
            )

    def ic_values_delta_displacement(self, sd: pp.Grid) -> np.ndarray:
        """Redirect initial values for displacement to the displacement increment."""
        if hasattr(self, "ic_values_displacement"):
            return self.ic_values_displacement(sd)
        else:
            return pp.momentum_balance.InitialConditionsMomentumBalance.ic_values_displacement(
                self, sd
            )

    def ic_values_delta_interface_displacement(self, intf: pp.MortarGrid) -> np.ndarray:
        """Redirect initial values for interface displacement to the interface displacement
        increment."""
        if hasattr(self, "ic_values_interface_displacement"):
            return self.ic_values_interface_displacement(intf)
        else:
            return pp.momentum_balance.InitialConditionsMomentumBalance.ic_values_interface_displacement(
                self, intf
            )


class DataSavingMomentumBalance:
    """Auxiliary class for momentum balance with reference state."""

    def data_to_export(self):
        """Add reference state to export data."""
        data = super().data_to_export()

        # Add variants of displacements
        for sd in self.mdg.subdomains(return_data=False, dim=self.nd):
            # Reference displacement
            displacement_ref = self.reference_displacement([sd]).value(
                self.equation_system
            )
            data.append((sd, self.displacement_variable + "_ref", displacement_ref))

            # Delta displacement
            displacement_inc = self.delta_displacement([sd]).value(self.equation_system)
            data.append((sd, self.displacement_variable + "_inc", displacement_inc))

            # Full displacement
            # Find and remove the original (sd, self.displacement_variable) entry
            data = [
                entry
                for entry in data
                if not (entry[0] == sd and entry[1] == self.displacement_variable)
            ]
            # Append the new entry with updated displacement
            displacement = self.displacement([sd]).value(self.equation_system)
            data.append((sd, self.displacement_variable, displacement))

        # Add variants of interface displacements
        for intf in self.mdg.interfaces(return_data=False, dim=self.nd - 1):
            # Reference interface displacement
            interface_displacement_ref = self.reference_interface_displacement(
                [intf]
            ).value(self.equation_system)
            data.append(
                (
                    intf,
                    self.interface_displacement_variable + "_ref",
                    interface_displacement_ref,
                )
            )

            # Delta interface displacement
            interface_displacement_inc = self.delta_interface_displacement(
                [intf]
            ).value(self.equation_system)
            data.append(
                (
                    intf,
                    self.interface_displacement_variable + "_inc",
                    interface_displacement_inc,
                )
            )

            # Full interface displacement
            data = [
                entry
                for entry in data
                if not (
                    entry[0] == intf
                    and entry[1] == self.interface_displacement_variable
                )
            ]
            interface_displacement = self.interface_displacement([intf]).value(
                self.equation_system
            )
            data.append(
                (intf, self.interface_displacement_variable, interface_displacement)
            )

        return data


# Use the same design pattern as for pp.MomentumBalance. Mainly keep the equations,
# boundary conditions, and some constitutive laws and overwrite aperture specific
# laws in addition to new variables. They are updated via solution strategies.
class MomentumBalance(
    DataSavingMomentumBalance,
    contact_mechanics.DataSavingContactMechanics,
    pp.contact_mechanics.ContactMechanicsEquations,
    pp.momentum_balance.MomentumBalanceEquations,
    VariablesMomentumBalance,
    contact_mechanics.ContactTractionVariable,
    constitutive_laws.DisplacementJump,
    pp.contact_mechanics.ConstitutiveLawsContactMechanics,
    pp.momentum_balance.ConstitutiveLawsMomentumBalance,
    pp.momentum_balance.BoundaryConditionsMomentumBalance,
    InitialConditionsMomentumBalance,
    contact_mechanics.InitialConditionsContactTraction,
    pp.contact_mechanics.SolutionStrategyContactMechanics,
    pp.momentum_balance.SolutionStrategyMomentumBalance,
    pp.FluidMixin,
    pp.ModelGeometry,
    pp.DataSavingMixin,
):
    """Class for mixed-dimensional momentum balance with contact mechanics."""
