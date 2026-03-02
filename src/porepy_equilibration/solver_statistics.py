import porepy as pp
import numpy as np


class CustomSolverStatistics:
    """Track the norm of the increments in time."""

    equation_system: pp.EquationSystem
    nonlinear_solver_statistics: pp.NonlinearSolverStatistics

    def after_nonlinear_iteration(self, nonlinear_increment: np.ndarray) -> None:
        """Store the norm of the nonlinear increment."""
        super().after_nonlinear_iteration(nonlinear_increment)

        # Compute the increment in time
        current_solution = self.equation_system.get_variable_values(iterate_index=0)
        previous_solution = self.equation_system.get_variable_values(time_step_index=0)
        time_increment = current_solution - previous_solution

        # Compute the Lebesgue norm of the time increment
        metric = pp.VariableBasedLebesgueMetric(self)
        metric_value = metric(time_increment)

        # Update the keys of metric value to start with "time_increment_"
        metric_value = {
            f"time_increment_{key}": value for key, value in metric_value.items()
        }

        # Log custom data.
        self.nonlinear_solver_statistics.log_custom_data(append=False, **metric_value)
