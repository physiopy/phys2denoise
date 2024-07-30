from inspect import _empty, signature
from typing import Union

import pydra
from loguru import logger
from physutils.io import Physio

from phys2denoise.metrics.cardiac import *  # noqa
from phys2denoise.metrics.chest_belt import *  # noqa
from phys2denoise.metrics.multimodal import *  # noqa
from phys2denoise.metrics.utils import export_metric

_available_metrics = [
    "cardiac_phase",
    "respiratory_phase",
    "heart_rate",
    "heart_rate_variability",
    "heart_beat_interval",
    "respiratory_variance_time",
    "respiratory_pattern_variability",
    "env",
    "respiratory_variance",
    "retroicor",
]


def select_input_args(metric, metric_args):
    """
    Retrieve required args for metric from a dictionary of possible arguments.

    This function checks what parameters are accepted by a metric.
    Then, for each parameter, check if the user provided it or not.
    If they did not, but the parameter is required, throw an error -
    unless it's "physio", reserved name for the timeseries input to a metric.
    Otherwise, use the default.

    Parameters
    ----------
    metric : function
        Metric function to retrieve arguments for
    metric_args : dict
        Dictionary containing all arguments for all functions requested by the
        user

    Returns
    args : dict
        Arguments to provide as input to metric

    Raises
    ------
    ValueError
        If a required argument is missing

    """
    args = {}

    # Check the parameters required by the metric and given by the user (see docstring)
    for param in signature(metric).parameters.values():
        if param.name not in metric_args:
            if param.default == _empty and param.name != "physio":
                raise ValueError(
                    f"Missing parameter {param} required " f"to run {metric}"
                )
            else:
                args[param.name] = param.default
        else:
            args[param.name] = metric_args[param.name]

    return args


@pydra.mark.task
def compute_metrics(phys: Physio, metrics: Union[list, str]) -> Physio:
    if isinstance(metrics, list):
        for metric in metrics:
            if metric not in _available_metrics:
                logger.warning(f"Metric {metric} not available. Skipping")
                continue

            args = select_input_args(metric, {})
            phys, _ = globals()[metric](phys, **args)
            logger.info(f"Computed {metric}")
    else:
        raise ValueError("metrics must be a list of strings")
    return phys


@pydra.mark.task
def export_metrics(
    phys: Physio, metrics: Union[list, str], outdir: str, tr: Union[int, float]
) -> None:
    if metrics == "all":
        logger.info("Exporting all computed metrics")
        for metric in phys.computed_metrics.keys():
            has_lags = True if "has_lags" in phys.computed_metrics[metric] else False
            export_metric(
                phys.computed_metrics[metric], phys.fs, tr, outdir, has_lags=has_lags
            )
            logger.info(f"Exported {metric}")
    elif isinstance(metrics, list):
        for metric in metrics:
            if metric not in phys.computed_metrics.keys():
                logger.warning(f"Metric {metric} not computed. Skipping")
                continue
            has_lags = True if "has_lags" in phys.computed_metrics[metric] else False
            export_metric(
                phys.computed_metrics[metric], phys.fs, tr, outdir, has_lags=has_lags
            )
            logger.info(f"Exported {metric}")
    else:
        raise ValueError("metrics must be a list of strings or 'all'")
