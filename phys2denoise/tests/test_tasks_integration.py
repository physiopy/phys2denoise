"""Tests for phys2denoise.tasks and their integration."""
from loguru import logger
from physutils import physio
from physutils.tasks import transform_to_physio
from pydra import Workflow

import phys2denoise.tasks as tasks

# def test_integration(fake_phys):
#     """Test the integration of phys2denoise tasks."""
#     # Create physio object data
#     transform_to_physio.inputs.data = fake_phys

#     wf = Workflow(name="metric_calculation", input_spec=["data"])
#     wf.add(transform_to_physio)

#     # Test the integration of the tasks
#     tasks.select_input_args(phys, {"physio": phys})
#     tasks.select_input_args(phys, {"physio": phys, "window": 4})
#     tasks.select_input_args(phys, {"physio": phys, "window": 4, "buffer": 2})
#     tasks.select_input_args(phys, {"physio": phys, "window": 4, "buffer": 2, "inverse": False})
#     tasks.compute_metrics(phys, metrics=["crf", "respiratory_variance"])
