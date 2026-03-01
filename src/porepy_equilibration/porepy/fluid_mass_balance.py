"""Single-phase fluid mass balance equations with reference state."""

from __future__ import annotations

import logging
from typing import Callable, Sequence, cast

import numpy as np

import porepy as pp

logger = logging.getLogger(__name__)


class InitialConditionsSinglePhaseFlow(pp.InitialConditionMixin):
    """Mixin for providing initial values for pressure and fluxes with reference state.

    Redirects initial values for pressure and fluxes to their increment (delta) versions.
    """

    delta_pressure: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    """See :class:`VariablesSinglePhaseFlow`."""

    delta_interface_darcy_flux: Callable[[list[pp.MortarGrid]], pp.ad.Operator]
    """See :class:`VariablesSinglePhaseFlow`."""

    delta_well_flux: Callable[[list[pp.MortarGrid]], pp.ad.Operator]
    """See :class:`VariablesSinglePhaseFlow`."""

    def initial_condition(self):
        """After the super-call, it sets initial values for the interface darcy flux
        and well flux.

        Note:
            Pressure is considered primary and the interface fluxes can in theory be
            constructed from it in simple cases. This makes them secondary variables
            indirectly but in practice they are treated as primary variables.

            Uses cases requiring a consistent initialization will benefit from the
            order here in the initialization routine.

        See also:

            - :meth:`ic_values_interface_darcy_flux`
            - :meth:`ic_values_well_flux`

        """
        # NOTE IMPORTANT: Super-call placed on top to ensure that variables considered
        # primary in the initialization (like pressure) are available before various
        # fluxes are initialized (see also set_initial_values_primary_variables, whose
        # super-call must be resolved first by the IC base mixin).
        super().initial_condition()

        for intf in self.mdg.interfaces():
            if intf.codim == 1:
                self.equation_system.set_variable_values(
                    self.ic_values_interface_darcy_flux(intf),
                    [cast(pp.ad.Variable, self.delta_interface_darcy_flux([intf]))],
                    iterate_index=0,
                )

            if intf.codim == 2:
                self.equation_system.set_variable_values(
                    self.ic_values_well_flux(intf),
                    [cast(pp.ad.Variable, self.delta_well_flux([intf]))],
                    iterate_index=0,
                )

        for sd, data in self.mdg.subdomains(return_data=True):
            pp.initialize_data(
                data,
                self.mobility_keyword,
                {"darcy_flux": np.zeros(sd.num_faces)},
            )
        for intf, data in self.mdg.interfaces(return_data=True):
            pp.initialize_data(
                data,
                self.mobility_keyword,
                {"darcy_flux": np.zeros(intf.num_cells)},
            )

    def set_initial_values_primary_variables(self) -> None:
        """Method to set initial values for pressure increment at iterate index 0.

        See also:

            - :meth:`ic_values_delta_pressure`

        """
        # Super call for compatibility with multi-physics.
        super().set_initial_values_primary_variables()

        for sd in self.mdg.subdomains():
            # Need to cast the return value to variable, because it is typed as
            # operator.
            self.equation_system.set_variable_values(
                self.ic_values_delta_pressure(sd),
                [cast(pp.ad.Variable, self.delta_pressure([sd]))],
                iterate_index=0,
            )

    def ic_values_delta_pressure(self, sd: pp.Grid) -> np.ndarray:
        """Redirect initial values for pressure to the pressure increment.

        Override this method to provide different initial conditions.

        Parameters:
            sd: A subdomain in the md-grid.

        Returns:
            The initial pressure increment values on that subdomain with
            ``shape=(sd.num_cells,)``. Defaults to zero array.

        """
        if hasattr(self, "ic_values_pressure"):
            return self.ic_values_pressure(sd)
        else:
            return pp.fluid_mass_balance.InitialConditionsSinglePhaseFlow.ic_values_pressure(
                self, sd
            )

    def ic_values_interface_darcy_flux(self, intf: pp.MortarGrid) -> np.ndarray:
        """Method returning the initial interface Darcy flux increment values.

        Redirect to parent class if ic_values_interface_darcy_flux exists.

        Override this method to customize the initialization.

        Note:
            This method is only called for interfaces with codimension 1.

        Parameters:
            intf: A mortar grid in the md-grid.

        Returns:
            The initial interface Darcy flux increment values with
            ``shape=(interface.num_cells,)``. Defaults to zero array.

        """
        if hasattr(self, "ic_values_interface_darcy_flux"):
            # Avoid infinite recursion by checking if we have this as a callable
            # on the parent class
            try:
                return pp.fluid_mass_balance.InitialConditionsSinglePhaseFlow.ic_values_interface_darcy_flux(
                    self, intf
                )
            except AttributeError:
                return np.zeros(intf.num_cells)
        return np.zeros(intf.num_cells)

    def ic_values_well_flux(self, intf: pp.MortarGrid) -> np.ndarray:
        """Method returning the initial well flux increment values.

        Redirect to parent class if ic_values_well_flux exists.

        Override this method to customize the initialization.

        Note:
            This method is only called for interfaces with codimension 2.

        Parameters:
            intf: A mortar grid in the md-grid.

        Returns:
            The initial well flux increment values with
            ``shape=(interface.num_cells,)``. Defaults to zero array.

        """
        if hasattr(self, "ic_values_well_flux"):
            # Avoid infinite recursion by checking if we have this as a callable
            # on the parent class
            try:
                return pp.fluid_mass_balance.InitialConditionsSinglePhaseFlow.ic_values_well_flux(
                    self, intf
                )
            except AttributeError:
                return np.zeros(intf.num_cells)
        return np.zeros(intf.num_cells)


class VariablesSinglePhaseFlow(pp.fluid_mass_balance.VariablesSinglePhaseFlow):
    """Variables for single-phase flow with reference state."""

    pressure_variable: str
    interface_darcy_flux_variable: str
    well_flux_variable: str
    set_reference_flow_variables: bool = False

    def reference_pressure(self, domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        """Reference pressure on subdomains."""

        # Initialize if required.
        self.init_reference_flow_variables()

        return pp.ad.TimeDependentDenseArray(
            name="reference_" + self.pressure_variable,
            domains=domains,
        )

    def reference_interface_darcy_flux(
        self, interfaces: list[pp.MortarGrid]
    ) -> pp.ad.Operator:
        """Reference Darcy flux on codimension-1 interfaces."""

        # Initialize if required.
        self.init_reference_flow_variables()

        return pp.ad.TimeDependentDenseArray(
            name="reference_" + self.interface_darcy_flux_variable,
            domains=interfaces,
        )

    def reference_well_flux(self, interfaces: list[pp.MortarGrid]) -> pp.ad.Operator:
        """Reference well flux on codimension-2 interfaces."""

        # Initialize if required.
        self.init_reference_flow_variables()

        return pp.ad.TimeDependentDenseArray(
            name="reference_" + self.well_flux_variable,
            domains=interfaces,
        )

    def init_reference_flow_variables(self) -> None:
        """Initialization of reference flow variables."""

        # Employ a hack to initialize the reference state before it is used,
        # since the reference state is needed in the creation of the variables,
        # but the initialization of the reference state needs to be done after the
        # variables are created. This is a bit of a circular dependency, but it works
        # in practice. The alternative would be to separate the creation of the
        # variables and the initialization of the reference state into two separate
        # steps, but that would be more cumbersome to use.
        # Make sure to only initialize once.
        if self.set_reference_flow_variables:
            return
        self.set_reference_flow_variables = True

        # Set reference state for pressure on all subdomains
        for sd, data in self.mdg.subdomains(return_data=True):
            for index in [{"iterate_index": 0}, {"time_step_index": 0}]:
                pp.set_solution_values(
                    name="reference_" + self.pressure_variable,
                    values=np.zeros(sd.num_cells),
                    data=data,
                    **index,
                )

        # Set reference state for interface Darcy flux on codim-1 interfaces
        for intf, data in self.mdg.interfaces(return_data=True, codim=1):
            for index in [{"iterate_index": 0}, {"time_step_index": 0}]:
                pp.set_solution_values(
                    name="reference_" + self.interface_darcy_flux_variable,
                    values=np.zeros(intf.num_cells),
                    data=data,
                    **index,
                )

        # Set reference state for well flux on codim-2 interfaces
        for intf, data in self.mdg.interfaces(return_data=True, codim=2):
            for index in [{"iterate_index": 0}, {"time_step_index": 0}]:
                pp.set_solution_values(
                    name="reference_" + self.well_flux_variable,
                    values=np.zeros(intf.num_cells),
                    data=data,
                    **index,
                )

        # Write logging info in yellow color
        logger.info(
            "\033[93m" + "Initialized reference flow variables to zero." + "\033[0m"
        )

    def update_reference(self) -> None:
        """Updating of reference flow variables."""

        # If super has an update_reference method, call it for compatibility with multi-physics.
        if hasattr(super(), "update_reference"):
            super().update_reference()

        # Update reference state for pressure
        for sd, data in self.mdg.subdomains(return_data=True):
            for index in [{"iterate_index": 0}, {"time_step_index": 0}]:
                for name, values in [
                    (
                        "reference_" + self.pressure_variable,
                        self.pressure([sd]).value(self.equation_system),
                    ),
                    (
                        self.pressure_variable,
                        np.zeros(sd.num_cells),
                    ),
                ]:
                    pp.set_solution_values(
                        name=name,
                        values=values,
                        data=data,
                        **index,
                    )

        # Update reference state for interface Darcy flux
        for intf, data in self.mdg.interfaces(return_data=True, codim=1):
            for index in [{"iterate_index": 0}, {"time_step_index": 0}]:
                for name, values in [
                    (
                        "reference_" + self.interface_darcy_flux_variable,
                        self.interface_darcy_flux([intf]).value(self.equation_system),
                    ),
                    (
                        self.interface_darcy_flux_variable,
                        np.zeros(intf.num_cells),
                    ),
                ]:
                    pp.set_solution_values(
                        name=name,
                        values=values,
                        data=data,
                        **index,
                    )

        # Update reference state for well flux
        for intf, data in self.mdg.interfaces(return_data=True, codim=2):
            for index in [{"iterate_index": 0}, {"time_step_index": 0}]:
                for name, values in [
                    (
                        "reference_" + self.well_flux_variable,
                        self.well_flux([intf]).value(self.equation_system),
                    ),
                    (
                        self.well_flux_variable,
                        np.zeros(intf.num_cells),
                    ),
                ]:
                    pp.set_solution_values(
                        name=name,
                        values=values,
                        data=data,
                        **index,
                    )

        # Write logging info in yellow color
        logger.info("\033[93m" + "Updated reference flow variables." + "\033[0m")

    def delta_pressure(self, domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        """Pressure increment on subdomains."""
        if len(domains) > 0 and isinstance(domains[0], pp.BoundaryGrid):
            return self.create_boundary_operator(
                name=self.pressure_variable,
                domains=cast(Sequence[pp.BoundaryGrid], domains),
            )
        # Check that all domains are subdomains.
        if not all(isinstance(g, pp.Grid) for g in domains):
            raise ValueError("grids must consist entirely of subdomains.")
        # Now we can cast the grids
        domains = cast(list[pp.Grid], domains)

        return self.equation_system.md_variable(self.pressure_variable, domains)

    def pressure(self, domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        """Pressure term. Either a primary variable if subdomains are provided a
        boundary condition operator if boundary grids are provided.

        Parameters:
            domains: List of subdomains or boundary grids.

        Raises:
            ValueError: If the grids are not all subdomains or all boundary grids.

        Returns:
            Operator representing the pressure [Pa].

        """
        if len(domains) > 0 and isinstance(domains[0], pp.BoundaryGrid):
            return self.delta_pressure(domains)
        # Check that all domains are subdomains.
        if not all(isinstance(g, pp.Grid) for g in domains):
            raise ValueError("grids must consist entirely of subdomains.")
        # Now we can cast the grids
        domains = cast(list[pp.Grid], domains)

        return self.reference_pressure(domains) + self.delta_pressure(domains)

    def delta_interface_darcy_flux(
        self, interfaces: list[pp.MortarGrid]
    ) -> pp.ad.Operator:
        """Interface Darcy flux increment on codimension-1 interfaces."""
        return self.equation_system.md_variable(
            self.interface_darcy_flux_variable, interfaces
        )

    def interface_darcy_flux(
        self, interfaces: list[pp.MortarGrid]
    ) -> pp.ad.MixedDimensionalVariable:
        """Interface Darcy flux.

        Integrated over faces in the mortar grid.

        Parameters:
            interfaces: List of interface grids.

        Returns:
            Variable representing the interface Darcy flux [kg * m^2 * s^-2].

        """
        return self.reference_interface_darcy_flux(
            interfaces
        ) + self.delta_interface_darcy_flux(interfaces)

    def delta_well_flux(self, interfaces: list[pp.MortarGrid]) -> pp.ad.Operator:
        """Well flux increment on codimension-2 interfaces."""
        return self.equation_system.md_variable(self.well_flux_variable, interfaces)

    def well_flux(
        self, interfaces: list[pp.MortarGrid]
    ) -> pp.ad.MixedDimensionalVariable:
        """Variable for the volumetric well flux.

        Parameters:
            interfaces: List of interface grids.

        Returns:
            Variable representing the Darcy-like well flux [kg * m^2 * s^-2].

        """
        return self.reference_well_flux(interfaces) + self.delta_well_flux(interfaces)


class DataSavingFluidMassBalance:
    """Auxiliary class for fluid mass balance with reference state."""

    def data_to_export(self):
        """Add reference state variants to export data."""
        data = super().data_to_export()

        # Add variants of pressure
        for sd in self.mdg.subdomains(return_data=False):
            # Reference pressure
            pressure_ref = self.evaluate_and_scale([sd], "reference_pressure", "Pa")
            data.append((sd, self.pressure_variable + "_ref", pressure_ref))

            # Delta pressure
            pressure_inc = self.evaluate_and_scale([sd], "delta_pressure", "Pa")
            data.append((sd, self.pressure_variable + "_inc", pressure_inc))

            # Full pressure
            # Find and remove the original (sd, self.pressure_variable) entry
            data = [
                entry
                for entry in data
                if not (entry[0] == sd and entry[1] == self.pressure_variable)
            ]
            # Append the new entry with updated pressure
            pressure = self.evaluate_and_scale([sd], "pressure", "Pa")
            data.append((sd, self.pressure_variable, pressure))

        # Add variants of interface Darcy flux (codim-1)
        for intf in self.mdg.interfaces(return_data=False, codim=1):
            # Reference interface Darcy flux
            flux_ref = self.evaluate_and_scale([intf], "reference_interface_darcy_flux", "kg * m^-2 * s^-1")
            data.append((intf, self.interface_darcy_flux_variable + "_ref", flux_ref))

            # Delta interface Darcy flux
            flux_inc = self.evaluate_and_scale([intf], "delta_interface_darcy_flux", "kg * m^-2 * s^-1")
            data.append((intf, self.interface_darcy_flux_variable + "_inc", flux_inc))

            # Full interface Darcy flux
            # Find and remove the original (intf, self.interface_darcy_flux_variable) entry
            data = [
                entry
                for entry in data
                if not (
                    entry[0] == intf and entry[1] == self.interface_darcy_flux_variable
                )
            ]
            # Append the new entry with updated flux
            flux = self.evaluate_and_scale([intf], "interface_darcy_flux", "kg * m^-2 * s^-1")
            data.append((intf, self.interface_darcy_flux_variable, flux))

        # Add variants of well flux (codim-2)
        for intf in self.mdg.interfaces(return_data=False, codim=2):
            # Reference well flux
            well_ref = self.evaluate_and_scale([intf], "reference_well_flux", "kg * m^-2 * s^-1")
            data.append((intf, self.well_flux_variable + "_ref", well_ref))

            # Delta well flux
            well_inc = self.evaluate_and_scale([intf], "delta_well_flux", "kg * m^-2 * s^-1")
            data.append((intf, self.well_flux_variable + "_inc", well_inc))

            # Full well flux
            # Find and remove the original (intf, self.well_flux_variable) entry
            data = [
                entry
                for entry in data
                if not (entry[0] == intf and entry[1] == self.well_flux_variable)
            ]
            # Append the new entry with updated well flux
            well = self.evaluate_and_scale([intf], "well_flux", "kg * m^-2 * s^-1")
            data.append((intf, self.well_flux_variable, well))

        return data


# Use the same design as for :class:`pp.SinglePhaseFlow`.
class SinglePhaseFlow(  # type: ignore[misc]
    DataSavingFluidMassBalance,
    pp.fluid_mass_balance.FluidMassBalanceEquations,
    VariablesSinglePhaseFlow,
    pp.fluid_mass_balance.ConstitutiveLawsSinglePhaseFlow,
    pp.fluid_mass_balance.BoundaryConditionsSinglePhaseFlow,
    InitialConditionsSinglePhaseFlow,
    pp.fluid_mass_balance.SolutionStrategySinglePhaseFlow,
    pp.FluidMixin,
    pp.ModelGeometry,
    pp.DataSavingMixin,
):
    """Class for single-phase flow in mixed-dimensional porous media."""
