class NewtonReferenceUpdateStrategy:
    """Update the reference state at the beginning of each Newton loop."""

    def before_nonlinear_loop(self) -> None:
        """Update the reference state at the beginning of each Newton loop."""
        super().before_nonlinear_loop()
        self.update_reference()


class InexactNewtonReferenceUpdateStrategy:
    """Update the reference state at the beginning of each Newton iteration."""

    def before_nonlinear_iteration(self) -> None:
        """Update the reference state at the beginning of each Newton iteration."""
        super().before_nonlinear_iteration()
        self.update_reference()
