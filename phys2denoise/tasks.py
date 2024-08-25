import logging
from inspect import _empty, signature
from typing import Union

import pydra
from loguru import logger
from physutils.physio import Physio

from phys2denoise.metrics.cardiac import *  # noqa
from phys2denoise.metrics.chest_belt import respiratory_variance  # noqa
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

LGR = logging.getLogger(__name__)
LGR.setLevel(logging.DEBUG)


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
        if param.name == "data" or param.name == "kwargs":
            continue
        if param.name not in metric_args:
            if param.default == _empty:
                raise ValueError(
                    f"Missing parameter {param} required " f"to run {metric}"
                )
            else:
                args[param.name] = param.default
        else:
            args[param.name] = metric_args[param.name]

    return args


@pydra.mark.task
def compute_metrics(phys: Physio, metrics: Union[list, str], args: dict) -> Physio:
    from phys2denoise.metrics.cardiac import (  # noqa
        cardiac_phase,
        heart_beat_interval,
        heart_rate,
        heart_rate_variability,
    )
    from phys2denoise.metrics.chest_belt import (  # noqa
        env,
        respiratory_pattern_variability,
        respiratory_phase,
        respiratory_variance,
        respiratory_variance_time,
    )
    from phys2denoise.metrics.multimodal import retroicor  # noqa

    if isinstance(metrics, list) or isinstance(metrics, str):
        for metric in metrics:
            if metric not in _available_metrics:
                LGR.warning(f"Metric {metric} not available. Skipping")
                continue
            LGR.debug(f"Computing {metric}")

            if metric not in args or args[metric] is None:
                metric_args = {}
            else:
                metric_args = args[metric]

            input_args = select_input_args(locals()[metric], metric_args)
            phys = locals()[metric](phys, **input_args)
    return phys


@pydra.mark.task
def export_metrics(
    phys: Physio, metrics: Union[list, str], outdir: str, tr: Union[int, float]
) -> None:
    if metrics == "all":
        LGR.info("Exporting all computed metrics")
        for metric in phys.computed_metrics.keys():
            has_lags = (
                True if "has_lags" in phys.computed_metrics[metric].args else False
            )
            prefix = outdir + f"/{metric}"
            export_metric(
                phys.computed_metrics[metric], phys.fs, tr, prefix, has_lags=has_lags
            )
            LGR.info(f"Exported {metric}")
    elif isinstance(metrics, list):
        for metric in metrics:
            if metric not in phys.computed_metrics.keys():
                LGR.warning(f"Metric {metric} not computed. Skipping")
                continue
            has_lags = (
                True if "has_lags" in phys.computed_metrics[metric].args else False
            )
            prefix = outdir + f"/{metric}"
            export_metric(
                phys.computed_metrics[metric],
                phys.fs,
                tr,
                prefix,
                has_lags=has_lags,
            )
            LGR.info(f"Exported {metric}")
    else:
        raise ValueError("metrics must be a list of strings or 'all'")
