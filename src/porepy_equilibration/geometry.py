"""Generic geometry class for reading SVG files and extracting fractures."""

import porepy as pp
from pathlib import Path


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

    def create_fracture_network(self) -> None:
        """Set the fracture network from the CSV geometry file."""
        # start path from parent of this current file
        csv_geometry_file = Path(__file__).parent / "data" / "salt_cove_fractures.csv"
        expected_domain_size = 1000
        self.fracture_network = pp.fracture_importer.network_from_csv(
            Path(csv_geometry_file),
            has_domain=True,
            tol=expected_domain_size * 1e-6,
        )
