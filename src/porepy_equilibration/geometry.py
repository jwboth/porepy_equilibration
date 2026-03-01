"""Generic geometry class for reading SVG files and extracting fractures."""

from pathlib import Path

import porepy as pp


class SaltCove:
    params: dict
    units: pp.Units

    def meshing_arguments(self) -> dict:
        mesh_args = {}
        mesh_args["cell_size"] = self.units.convert_units(30.0, "m")
        mesh_args["cell_size_fracture"] = self.units.convert_units(1.0, "m")
        return mesh_args

    def grid_type(self) -> str:
        return "simplex"

    def set_domain(self) -> None:
        """Set the domain based on the CSV."""
        self.create_fracture_network()
        self._domain = self.fracture_network.domain

    def set_fractures(self) -> None:
        self.create_fracture_network()
        self._fractures = self.fracture_network.fractures

    def _csv_file_path(self) -> Path:
        """Get the path to the CSV geometry file."""
        return Path(__file__).parent / "data" / "salt_cove_fractures.csv"

    def create_fracture_network(self) -> None:
        """Set the fracture network from the CSV geometry file."""
        csv_geometry_file = self._csv_file_path()
        expected_domain_size = 1000
        self.fracture_network = pp.fracture_importer.network_from_csv(
            Path(csv_geometry_file),
            has_domain=True,
            tol=expected_domain_size * 1e-6,
        )

class SaltCoveVertical(SaltCove):
    def _csv_file_path(self) -> Path:
        """Get the path to the CSV geometry file."""
        return Path(__file__).parent / "data" / "salt_cove_fractures_vertical.csv"