from typing import Callable

import numpy as np

import porepy as pp


class TensileBackgroundStress2D:
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

    def background_stress(self, grid: pp.Grid) -> np.ndarray:
        s = np.zeros((self.nd, self.nd, grid.num_cells))
        bx = self.params["bc"]["background_stress"][0]
        by = self.params["bc"]["background_stress"][1]
        bxy = (
            self.params["bc"]["background_stress"][2]
            if len(self.params["bc"]["background_stress"]) == 3
            else 0.0
        )
        s[0, 0] = self.units.convert_units(bx, "Pa")
        s[1, 1] = self.units.convert_units(by, "Pa")
        s[0, 1] = self.units.convert_units(bxy, "Pa")
        s[1, 0] = self.units.convert_units(bxy, "Pa")
        return s

    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        boundary_faces = self.domain_boundary_sides(sd).all_bf
        bc = pp.BoundaryConditionVectorial(sd, boundary_faces, "dir")
        bc.internal_to_dirichlet(sd)

        # Set all boundary faces to Neumann initially
        domain_sides = self.domain_boundary_sides(sd)
        bc.is_dir[:, domain_sides.all_bf] = False
        bc.is_neu[:, domain_sides.all_bf] = True

        # Fix x/y coordinates in mid cells of faces
        if np.any(domain_sides.north):
            north_center = np.where(domain_sides.north)[0][
                np.argmin(np.abs(sd.face_centers[0, domain_sides.north] - 0))
            ]
            bc.is_dir[0, north_center] = True
            bc.is_neu[0, north_center] = False
        if np.any(domain_sides.south):
            south_center = np.where(domain_sides.south)[0][
                np.argmin(np.abs(sd.face_centers[0, domain_sides.south] - 0))
            ]
            bc.is_dir[0, south_center] = True
            bc.is_neu[0, south_center] = False
        if np.any(domain_sides.east):
            east_center = np.where(domain_sides.east)[0][
                np.argmin(np.abs(sd.face_centers[1, domain_sides.east] - 0))
            ]
            bc.is_dir[1, east_center] = True
            bc.is_neu[1, east_center] = False
        if np.any(domain_sides.west):
            west_center = np.where(domain_sides.west)[0][
                np.argmin(np.abs(sd.face_centers[1, domain_sides.west] - 0))
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


class GradualTensileBackgroundStress2D(TensileBackgroundStress2D):
    """Same as TensileBackgroundStress2D, but with gradual strength of the
    background stress over time.

    """

    time_manager: pp.TimeManager
    params: dict

    def background_stress(self, grid: pp.Grid) -> np.ndarray:
        s = np.zeros((self.nd, self.nd, grid.num_cells))
        bx = self.params["bc"]["background_stress"][0]
        by = self.params["bc"]["background_stress"][1]
        bxy = (
            self.params["bc"]["background_stress"][2]
            if len(self.params["bc"]["background_stress"]) == 3
            else 0.0
        )
        loading_step = min(
            1.0, 2.0 * self.time_manager.time / self.time_manager.time_final
        )
        by *= loading_step
        # Add 10% noise to shear components of the background stress.
        if loading_step < 1.0 and not np.isclose(loading_step, 1.0):
            noise = by * 0.1 * np.random.rand(grid.num_cells)  # Add up to 10% noise
            bxy += noise
        else:
            print("No noise")
        s[0, 0] = self.units.convert_units(bx, "Pa")
        s[1, 1] = self.units.convert_units(by, "Pa")
        s[0, 1] = self.units.convert_units(bxy, "Pa")
        s[1, 0] = self.units.convert_units(bxy, "Pa")
        return s


class RotatingTensileBackgroundStress2D(TensileBackgroundStress2D):
    """Same as TensileBackgroundStress2D, but with gradual strength of the
    background stress over time.

    """

    time_manager: pp.TimeManager
    params: dict

    def background_stress(self, grid: pp.Grid) -> np.ndarray:
        s = np.zeros((self.nd, self.nd, grid.num_cells))
        _bx = self.params["bc"]["background_stress"][0]
        _by = self.params["bc"]["background_stress"][1]
        _bxy = (
            self.params["bc"]["background_stress"][2]
            if len(self.params["bc"]["background_stress"]) == 3
            else 0.0
        )
        loading_step = self.time_manager.time_index
        angle_ratios = [0.0, 0.25, 0.5, 0.75, 1.0, 0.75, 0.5, 0.25, 0.0, 0.0]
        angle = (
            angle_ratios[loading_step - 1] * np.pi / 4
        )  # Rotate up to 45 degrees and back
        bx = _bx * np.cos(angle) - _by * np.sin(angle)
        by = _bx * np.sin(angle) + _by * np.cos(angle)
        bxy = _bxy
        print()
        print(
            f"Loading step {loading_step}, rotation angle: {np.degrees(angle):.1f} degrees"
        )
        print(f"Background stress components: bx={bx:.2e}, by={by:.2e}, bxy={bxy:.2e}")
        print()
        s[0, 0] = self.units.convert_units(bx, "Pa")
        s[1, 1] = self.units.convert_units(by, "Pa")
        s[0, 1] = self.units.convert_units(bxy, "Pa")
        s[1, 0] = self.units.convert_units(bxy, "Pa")
        return s
