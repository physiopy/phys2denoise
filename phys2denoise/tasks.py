from typing import Union

import pydra
from loguru import logger
from physutils.io import Physio

from phys2denoise.metrics.utils import export_metrics


@pydra.mark.task
def export_metrics(phys: Physio, metrics: Union[list, str], outdir: str) -> None:
    if metrics == "all":
        logger.info("Exporting all metrics")
