"""Constitutive laws in light of reference states."""

from __future__ import annotations

from typing import Callable, Sequence, TypeVar

import numpy as np

import porepy as pp

number = pp.number
Scalar = pp.ad.Scalar

ArrayType = TypeVar("ArrayType", pp.ad.AdArray, np.ndarray)


class MechanicalAperture:
    normal_component: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Operator giving the normal component of a vector on fractures."""

    displacement_jump: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Displacement jump on fractures."""

    def mechanical_aperture(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Mechanical aperture of fractures.

        Parameters:
            subdomains: List of subdomains where the mechanical aperture is defined.
                Should be a fracture subdomain.

        Returns:
            Operator for the mechanical aperture.

        Raises:
             AssertionError: If the subdomains are not fractures, i.e. have dimension
                `nd - 1`.

        """
        if not all([sd.dim == self.nd - 1 for sd in subdomains]):
            raise ValueError("Mechanical aperture only defined on fractures")

        reference_opening = self.reference_mechanical_aperture(subdomains)
        normal_jump = self.normal_component(subdomains) @ self.displacement_jump(
            subdomains
        )

        f_max = pp.ad.Function(pp.ad.maximum, "maximum_function")
        zero = Scalar(0.0, "zero")
        aperture = f_max(reference_opening + normal_jump, zero)
        aperture.set_name("mechanical_aperture")
        return aperture

    def reference_mechanical_aperture(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        """Reference mechanical aperture of fractures.

        Parameters:
            subdomains: List of subdomains where the reference mechanical aperture is
                defined. Should be a fracture subdomain.

        Returns:
            Operator for the reference mechanical aperture.

        Raises:
             AssertionError: If the subdomains are not fractures, i.e. have dimension
                `nd - 1`.

        """
        return Scalar(0.0, "reference_mechanical_aperture")


class DisplacementJump(pp.constitutive_laws.DisplacementJump):
    """Displacement jump on fractures taking into account reference state."""

    delta_interface_displacement: Callable[[list[pp.MortarGrid]], pp.ad.Operator]
    """Operator giving the displacement on interfaces."""

    @pp.ad.cached_method
    def displacement_jump(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Displacement jump on fracture-matrix interfaces.

        Parameters:
            subdomains: List of subdomains where the displacement jump is defined.
                Should be a fracture subdomain.

        Returns:
            Operator for the displacement jump.

        Raises:
             AssertionError: If the subdomains are not fractures, i.e. have dimension
                `nd - 1`.

        """
        if not all([sd.dim == self.nd - 1 for sd in subdomains]):
            raise ValueError("Displacement jump only defined on fractures")

        interfaces = self.subdomains_to_interfaces(subdomains, [1])
        # Only use matrix-fracture interfaces
        interfaces = [intf for intf in interfaces if intf.dim == self.nd - 1]
        mortar_projection = pp.ad.MortarProjections(
            self.mdg, subdomains, interfaces, self.nd
        )
        # The displacement jmup is expressed in the local coordinates of the fracture.
        # First use the sign of the mortar sides to get a difference, then map first
        # from the interface to the fracture, and finally to the local coordinates.
        rotated_jumps: pp.ad.Operator = (
            self.local_coordinates(subdomains)
            @ mortar_projection.mortar_to_secondary_avg()
            @ mortar_projection.sign_of_mortar_sides()
            @ self.delta_interface_displacement(interfaces)
        )
        rotated_jumps.set_name("Rotated_displacement_jump")
        return rotated_jumps


class DisplacementJumpAperture(
    pp.constitutive_laws.DisplacementJumpAperture, MechanicalAperture
):
    """Displacement jump aperture using mechanical aperture."""

    @pp.ad.cached_method
    def aperture(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Aperture [m].

        The aperture computation depends on the dimension of the subdomain. For the
        matrix, the aperture is one. For intersections, the aperture is given by the
        average of the apertures of the adjacent fractures. For fractures, the aperture
        equals displacement jump plus residual aperture.

        Parameters:
            subdomains: List of subdomain grids.

        Returns:
            Operator representing apertures.

        """
        # For now, assume no intersections
        projection = pp.ad.SubdomainProjections(subdomains, dim=1)
        # Subdomains of the top dimension
        nd_subdomains = [sd for sd in subdomains if sd.dim == self.nd]

        num_cells_nd_subdomains = sum(sd.num_cells for sd in nd_subdomains)
        # For the matrix, use unitary aperture in SI units, then convert to the model's
        # units.
        one = pp.wrap_as_dense_ad_array(
            self.units.convert_units(1, "m"), size=num_cells_nd_subdomains, name="one"
        )
        # Start with nd, where aperture is one.
        apertures = projection.cell_prolongation(nd_subdomains) @ one

        # NOTE: The loop is reversed, to ensure that the subdomains are processed in the
        # same order as will be returned by an iteration over the subdomains of the
        # mixed-dimensional grid. If the order in input argument subdomains is
        # different, the result will likely be wrong.
        # Only consider subdomains of lower dimension, there is no aperture for the top
        # dimension.
        for dim in range(self.nd - 1, -1, -1):
            subdomains_of_dim = [sd for sd in subdomains if sd.dim == dim]
            if len(subdomains_of_dim) == 0:
                continue
            if dim == self.nd - 1:
                # NOTE: The only difference to the original class.
                a_ref = self.residual_aperture(subdomains_of_dim)
                a_mech = self.mechanical_aperture(subdomains_of_dim)
                apertures_of_dim = a_ref + a_mech
                apertures_of_dim.set_name("aperture_maximum_function")
                apertures = (
                    apertures
                    + projection.cell_prolongation(subdomains_of_dim) @ apertures_of_dim
                )
            else:
                if dim == self.nd - 2:
                    well_subdomains = [
                        sd for sd in subdomains_of_dim if self.is_well_grid(sd)
                    ]
                    if len(well_subdomains) > 0:
                        # Wells. Aperture is given by well radius.
                        radii = [self.grid_aperture(sd) for sd in well_subdomains]
                        well_apertures = pp.wrap_as_dense_ad_array(
                            np.hstack(radii), name="well apertures"
                        )
                        apertures = (
                            apertures
                            + projection.cell_prolongation(well_subdomains)
                            @ well_apertures
                        )
                        # Well subdomains need not be considered further.
                        subdomains_of_dim = [
                            sd for sd in subdomains_of_dim if sd not in well_subdomains
                        ]

                # Intersection aperture is average of apertures of intersecting
                # fractures.
                interfaces_dim = self.subdomains_to_interfaces(subdomains_of_dim, [1])
                # Only consider interfaces of the current dimension, i.e. those related
                # to higher-dimensional neighbors.
                interfaces_dim = [intf for intf in interfaces_dim if intf.dim == dim]
                # Get the higher-dimensional neighbors.
                parent_subdomains = self.interfaces_to_subdomains(interfaces_dim)
                # Only consider the higher-dimensional neighbors, i.e. disregard the
                # intersections with the current dimension.
                parent_subdomains = [
                    sd for sd in parent_subdomains if sd.dim == dim + 1
                ]

                # Define the combined set of subdomains of this dimension and the
                # parents. Sort this according to the MixedDimensionalGrid's order of
                # the subdomains.
                parent_and_this_dim_subdomains = self.mdg.sort_subdomains(
                    subdomains_of_dim + parent_subdomains
                )

                # Create projection operator between the subdomains involved in the
                # computation, i.e. the current dimension and the parents.
                mortar_projection = pp.ad.MortarProjections(
                    self.mdg, parent_and_this_dim_subdomains, interfaces_dim
                )
                # Also create projections between the subdomains we act on.
                parent_and_subdomain_projection = pp.ad.SubdomainProjections(
                    parent_and_this_dim_subdomains
                )

                # Get the apertures of the higher-dimensional neighbors by calling this
                # method on the parents.
                parent_apertures = self.aperture(parent_subdomains)

                # The apertures on the lower-dimensional subdomains are the mean
                # apertures from the higher-dimensional neighbors. This requires both a
                # projection of the actual apertures and counting the number of
                # higher-dimensional neighbors.

                # Define a trace operator. This is needed to go from the cell-based
                # apertures among the parents to the faces (which are accessible to the
                # mortar projections).
                trace = pp.ad.Trace(parent_subdomains)

                # Projection from parents to intersections via the mortar grid. This is
                # a convoluted operation: Map from the trace (only defined on the
                # parents) to the full set of subdomains. Project first to the mortars
                # and then to the lower-dimensional subdomains. The resulting compound
                # projection is used  to map apertures and to count the number of
                # neighbors.
                parent_cells_to_intersection_cells = (
                    mortar_projection.mortar_to_secondary_avg()
                    @ mortar_projection.primary_to_mortar_avg()
                    @ parent_and_subdomain_projection.face_prolongation(
                        parent_subdomains
                    )
                    @ trace.trace
                )

                # Average weights are the number of cells in the parent subdomains
                # contributing to each intersection cells.
                weight_value = self.equation_system.evaluate(
                    parent_cells_to_intersection_cells
                )

                assert isinstance(weight_value, (sps.spmatrix, sps.sparray))  # for mypy
                average_weights = np.ravel(weight_value.sum(axis=1))
                nonzero = average_weights > 0
                average_weights[nonzero] = 1 / average_weights[nonzero]
                # Wrap as a DenseArray
                divide_by_num_neighbors = pp.wrap_as_dense_ad_array(
                    average_weights, name="average_weights"
                )

                # Project apertures from the parents and divide by the number of
                # higher-dimensional neighbors.
                apertures_of_dim = divide_by_num_neighbors * (
                    parent_cells_to_intersection_cells @ parent_apertures
                )
                # Above matrix is defined on intersections and parents. Restrict to
                # intersections.
                intersection_subdomain_projection = pp.ad.SubdomainProjections(
                    parent_and_this_dim_subdomains
                )
                apertures_of_dim = (
                    intersection_subdomain_projection.cell_restriction(
                        subdomains_of_dim
                    )
                    @ apertures_of_dim
                )
                # Set a name for the apertures of this dimension
                apertures_of_dim.set_name(f"Displacement_jump_aperture_dim_{dim}")

                # Add to total aperture.
                apertures += (
                    projection.cell_prolongation(subdomains_of_dim) @ apertures_of_dim
                )

        # Give the operator a name
        apertures.set_name("Displacement_jump_apertures")

        return apertures


class PoroMechanicsPorosity(pp.constitutive_laws.PoroMechanicsPorosity):
    r"""Porosity for poromechanical models taking into account reference states."""

    biot_coefficient: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Biot coefficient. Normally defined in a mixin instance of
    :class:`~porepy.models.constitutive_laws.BiotCoefficient`.

    """
    bulk_modulus: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Bulk modulus. Normally defined in a mixin instance of
    :class:`~porepy.models.constitutive_laws.ElasticModuli`.

    """
    stress_keyword: str
    """Keyword used to identify the stress discretization. Normally set by a mixin
    instance of
    :class:`~porepy.models.momentum_balance.SolutionStrategyMomentumBalance`.

    """
    delta_pressure: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    """Perturbation of pressure from reference."""

    delta_displacement: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    """Displacement increment variable. Normally defined in a mixin instance of
    :class:`~porepy.models.momentum_balance.VariablesMomentumBalance`.

    """
    delta_interface_displacement: Callable[[list[pp.MortarGrid]], pp.ad.Operator]
    """Displacement increment on interfaces. Normally defined in a mixin instance of
    :class:`~porepy.models.momentum_balance.VariablesMomentumBalance`.

    """
    combine_boundary_operators_mechanical_stress: Callable[
        [Sequence[pp.Grid]], pp.ad.Operator
    ]
    """Combine mechanical stress boundary operators for different types of boundary
    conditions. Can be provided by a mixin class of type
    :class:`~porepy.models.constitutive_laws.LinearElasticMechanicalStress`.
    """

    def porosity_change_from_pressure(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        """Contribution of the pressure changes to the matrix porosity [-].

        Parameters:
            subdomains: List of subdomains where the porosity is defined.

        Returns:
            Cell-wise contribution of the pressure changes to the matrix porosity [-].

        """

        # Retrieve material parameters.
        alpha = self.biot_coefficient(subdomains)
        phi_ref = self.reference_porosity(subdomains)
        bulk_modulus = self.bulk_modulus(subdomains)

        # Pressure changes
        dp = self.delta_pressure(subdomains)

        # Compute 1/N as defined in Coussy, 2004, https://doi.org/10.1002/0470092718.
        n_inv = (alpha - phi_ref) * (Scalar(1) - alpha) / bulk_modulus

        # Pressure change contribution
        pressure_contribution = n_inv * dp
        pressure_contribution.set_name("Porosity change from pressure")

        return pressure_contribution

    def displacement_divergence(
        self,
        subdomains: list[pp.Grid],
    ) -> pp.ad.Operator:
        """Divergence of displacement [-].

        This is ``alpha : grad(u)`` where ``alpha`` is the Biot tensor and ``u`` is
        the displacement. If the tensor is isotropic, the expression simplifies to
        ``alpha * div(u)``, where ``alpha`` can be interpreted as a scalar.

        Parameters:
            subdomains: List of subdomains where the divergence is defined.

        Returns:
            Divergence operator accounting from contributions from interior of the
            domain and from internal and external boundaries.

        """
        # Sanity check on dimension
        if not all(sd.dim == self.nd for sd in subdomains):
            raise ValueError("Displacement divergence only defined in nd.")

        # Obtain neighbouring interfaces
        interfaces = self.subdomains_to_interfaces(subdomains, [1])
        # Mock discretization (empty `discretize` method), used to access discretization
        # matrices computed by Biot discretization.
        discr = pp.ad.BiotAd(self.stress_keyword, subdomains)
        # Projections
        sd_projection = pp.ad.SubdomainProjections(subdomains, dim=self.nd)
        mortar_projection = pp.ad.MortarProjections(
            self.mdg, subdomains, interfaces, dim=self.nd
        )

        boundary_operator = self.combine_boundary_operators_mechanical_stress(
            subdomains
        )

        # Compose operator.
        displacement_divergence_integrated = discr.displacement_divergence(
            self.darcy_keyword
        ) @ self.delta_displacement(subdomains) + discr.bound_displacement_divergence(
            self.darcy_keyword
        ) @ (
            boundary_operator
            + sd_projection.face_restriction(subdomains)
            @ mortar_projection.mortar_to_primary_avg()
            @ self.delta_interface_displacement(interfaces)
        )
        # Divide by cell volumes to counteract integration. The displacement_divergence
        # discretization contains a volume integral. Since this is used here together
        # with intensive quantities, we need to divide by cell volumes.
        cell_volumes_inv = Scalar(1) / self.wrap_grid_attribute(
            subdomains, "cell_volumes", dim=1
        )
        displacement_divergence = cell_volumes_inv * displacement_divergence_integrated
        displacement_divergence.set_name("displacement_divergence")
        return displacement_divergence

    def _mpsa_consistency(
        self, subdomains: list[pp.Grid], physics_name: str, variable_name: str
    ) -> pp.ad.Operator:
        """Consistency term for Mpsa discretizations of coupled problems.

        This function returns a diffusion-type term that is needed to ensure an
        MPSA-type discretization of poromechanics (or
        thermomechanics/thermoporomechanics) is stable in the limit of vanishing time
        steps and permeability. The term arises naturally from the MPSA discretization,
        see Nordbotten 2016 (doi:10.1137/15M1014280) for details.

        Parameters:
            subdomains: List of subdomains where the consistency is defined.
            physics_name: The physics keyword for which the consistency is computed.
                This is the keyword used in the scalar_vector_mapping provided to the
                Mpsa Biot discretization.
            variable_name: Name of the variable which should have a consistency term.

        Returns:
            Biot consistency operator.

        """
        # Sanity check on dimension
        if not all(sd.dim == self.nd for sd in subdomains):
            raise ValueError("Mpsa consistency only defined in nd.")

        discr = pp.ad.BiotAd(self.stress_keyword, subdomains)

        # The consistency is based on perturbation. If the variable is used directly,
        # results will not match if the reference state is not zero, see
        # :func:`test_without_fracture` in test_poromechanics.py.
        dp = self.delta_pressure(subdomains)
        consistency_integrated = discr.consistency(physics_name) @ dp

        # Divide by cell volumes to counteract integration.
        # The consistency discretization contains a volume integral. Since the
        # consistency term is used here together with intensive quantities, we need to
        # divide by cell volumes.
        cell_volumes_inverse = Scalar(1) / self.wrap_grid_attribute(
            subdomains, "cell_volumes", dim=1
        )
        consistency = cell_volumes_inverse * consistency_integrated
        consistency.set_name("mpsa_consistency")
        return consistency
