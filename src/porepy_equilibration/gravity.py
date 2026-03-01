from typing import Callable

import numpy as np
from numpy.typing import NDArray
from porepy.grids.grid import Grid

import porepy as pp


class LithostaticBackgroundStress:
    def lithostatic_pressure(self, grid: pp.Grid) -> np.ndarray:
        """Lithostatic pressure."""
        gravity = self.units.convert_units(pp.GRAVITY_ACCELERATION, "m*s^-2")
        rho_g = self.solid.density * gravity
        z = grid.cell_centers[self.nd - 1]
        return -rho_g * z

    def vertical_background_stress(self, grid: pp.Grid) -> np.ndarray:
        """Vertical background stress."""
        # NOTE: The convention in PP is that compressive stresses are negative.
        return -self.lithostatic_pressure(grid)

    def horizontal_background_stress(self, grid: pp.Grid) -> np.ndarray:
        """Zero horizontal background stress."""
        s_v = self.vertical_background_stress(grid)
        s_h = np.zeros((self.nd - 1, self.nd - 1, grid.num_cells))
        scaling = 0.25
        for i, j in np.ndindex(self.nd - 1, self.nd - 1):
            s_h[i, j] = scaling * s_v
        return s_h

    def background_stress(self, grid: pp.Grid) -> np.ndarray:
        """Combination of vertical (lithostatic) and horizontal stress."""

        s_h = self.horizontal_background_stress(grid)
        s_v = self.vertical_background_stress(grid)
        s = np.zeros((self.nd, self.nd, grid.num_cells))
        for i, j in np.ndindex(self.nd - 1, self.nd - 1):
            s[i, j] = s_h[i, j]
        s[-1, -1] = s_v
        return s


class HydrostaticPressure(
    pp.applications.boundary_conditions.model_boundary_conditions.HydrostaticPressureValues,
    # pp.applications.initial_conditions.model_initial_conditions.InitialConditionHydrostaticPressureValues,
    # pp.applications.boundary_conditions.model_boundary_conditions.HydrostaticBoundaryPressureValues,
):
    def ic_values_pressure(self, sd: Grid) -> NDArray:
        """Pressure values.

        Parameters:
            sd: Subdomain grid for which initial values are to be returned.

        Returns:
            Array of initial values, with one value for each cell in the subdomain.

        """
        depth = self.depth(sd.cell_centers)

        values = self.hydrostatic_pressure(depth)
        return values

    def initial_pressure(self, domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        # Set reference state for pressure on all subdomains
        for sd, data in self.mdg.subdomains(return_data=True):
            for index in [{"iterate_index": 0}, {"time_step_index": 0}]:
                pp.set_solution_values(
                    name="initial_pressure",
                    values=self.ic_values_pressure(sd),
                    data=data,
                    **index,
                )

        return pp.ad.TimeDependentDenseArray(
            name="initial_pressure",
            domains=domains,
        )

    def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """Pressure values.

        Parameters:
            boundary_grid: Boundary grid for which boundary values are to be returned.

        Returns:
            Array of boundary values, with one value for each face in the subdomain.

        """
        depth = self.depth(boundary_grid.cell_centers)
        values = self.hydrostatic_pressure(depth)
        return values


class Gravity(
    HydrostaticPressure,
    pp.constitutive_laws.GravityForce,
    LithostaticBackgroundStress,
):
    """Mechanical BC based on pure stress description in 2D, and holding the mid-points
    of each side fixed. Resembles 2D horizontal plane strain setup with tensile background
    stress.

    Default parameters based on unidirectional compression example (Ex. 5.1)
    https://doi.org/10.1002/nme.707.

    Takes model parameters:
        - background_stress: Magnitude of the tensile stress in Pa for [sigma_xx, sigma_yy, sigma_xy]

    """

    units: pp.Units
    nd: int
    domain_boundary_sides: Callable[[pp.Grid | pp.BoundaryGrid], pp.domain.DomainSides]
    time_manager: pp.TimeManager
    onset: bool
    params: dict

    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        boundary_faces = self.domain_boundary_sides(sd).all_bf
        bc = pp.BoundaryConditionVectorial(sd, boundary_faces, "dir")
        bc.internal_to_dirichlet(sd)

        # Set all boundary faces to Neumann initially
        domain_sides = self.domain_boundary_sides(sd)
        bc.is_dir[:, domain_sides.all_bf] = False
        bc.is_neu[:, domain_sides.all_bf] = True

        center = [500, -2500]

        # Fix x/y coordinates in mid cells of faces
        if np.any(domain_sides.north):
            north_center = np.where(domain_sides.north)[0][
                np.argmin(np.abs(sd.face_centers[0, domain_sides.north] - center[0]))
            ]
            bc.is_dir[0, north_center] = True
            bc.is_neu[0, north_center] = False
        if np.any(domain_sides.south):
            south_center = np.where(domain_sides.south)[0][
                np.argmin(np.abs(sd.face_centers[0, domain_sides.south] - center[0]))
            ]
            bc.is_dir[0, south_center] = True
            bc.is_neu[0, south_center] = False
        if np.any(domain_sides.east):
            east_center = np.where(domain_sides.east)[0][
                np.argmin(np.abs(sd.face_centers[1, domain_sides.east] - center[1]))
            ]
            bc.is_dir[1, east_center] = True
            bc.is_neu[1, east_center] = False
        if np.any(domain_sides.west):
            west_center = np.where(domain_sides.west)[0][
                np.argmin(np.abs(sd.face_centers[1, domain_sides.west] - center[1]))
            ]
            bc.is_dir[1, west_center] = True
            bc.is_neu[1, west_center] = False
        return bc

    @property
    def onset(self) -> bool:
        return self.time_manager.time_index > 0

    def bc_values_stress(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        vals = np.zeros((self.nd, boundary_grid.num_cells))
        if boundary_grid.dim == self.nd - 1 and self.onset:
            background_stress_tensor = self.background_stress(boundary_grid)
            domain_sides = self.domain_boundary_sides(boundary_grid)
            if np.any(domain_sides.east):
                normal = np.array([1, 0])
                east_center = np.where(domain_sides.east)[0][
                    np.argmin(
                        np.abs(boundary_grid.cell_centers[1, domain_sides.east] - 0)
                    )
                ]
                east_others = np.array(
                    list(set(np.where(domain_sides.east)[0].tolist()) - {east_center})
                )
                vals[:, east_others] = (
                    np.einsum(
                        "ijk,j->ik", background_stress_tensor[:, :, east_others], normal
                    )
                    * boundary_grid.cell_volumes[east_others]
                )
            if np.any(domain_sides.west):
                normal = np.array([-1, 0])
                west_center = np.where(domain_sides.west)[0][
                    np.argmin(
                        np.abs(boundary_grid.cell_centers[1, domain_sides.west] - 0)
                    )
                ]
                west_others = list(
                    set(np.where(domain_sides.west)[0].tolist()) - {west_center}
                )
                vals[:, west_others] = (
                    np.einsum(
                        "ijk,j->ik", background_stress_tensor[:, :, west_others], normal
                    )
                    * boundary_grid.cell_volumes[west_others]
                )
            if np.any(domain_sides.south):
                normal = np.array([0, -1])
                south_center = np.where(domain_sides.south)[0][
                    np.argmin(
                        np.abs(boundary_grid.cell_centers[0, domain_sides.south] - 0)
                    )
                ]
                south_others = list(
                    set(np.where(domain_sides.south)[0].tolist()) - {south_center}
                )
                vals[:, south_others] = (
                    np.einsum(
                        "ijk,j->ik",
                        background_stress_tensor[:, :, south_others],
                        normal,
                    )
                    * boundary_grid.cell_volumes[south_others]
                )
            if np.any(domain_sides.north):
                normal = np.array([0, 1])
                north_center = np.where(domain_sides.north)[0][
                    np.argmin(
                        np.abs(boundary_grid.cell_centers[0, domain_sides.north] - 0)
                    )
                ]
                north_others = np.array(
                    list(set(np.where(domain_sides.north)[0].tolist()) - {north_center})
                )
                vals[:, north_others] = (
                    np.einsum(
                        "ijk,j->ik",
                        background_stress_tensor[:, :, north_others],
                        normal,
                    )
                    * boundary_grid.cell_volumes[north_others]
                )
        return vals.ravel("F")


class GradualGravity(Gravity):
    """Gravity BC with gradual onset, where the background stress is scaled by time manager's time value."""

    def horizontal_background_stress(self, grid: pp.Grid) -> np.ndarray:
        """Zero horizontal background stress."""
        s_v = self.vertical_background_stress(grid)
        s_h = np.zeros((self.nd - 1, self.nd - 1, grid.num_cells))
        loading_step = min(
            1.0,
            (self.time_manager.time - self.time_manager.dt_init)
            / (self.time_manager.time_final - self.time_manager.dt_init),
        )
        scaling = 1.0 - loading_step * 0.75
        for i, j in np.ndindex(self.nd - 1, self.nd - 1):
            s_h[i, j] = scaling * s_v
        return s_h
