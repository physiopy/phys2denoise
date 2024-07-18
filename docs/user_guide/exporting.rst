.. _usage_exporting:

Exporting physiological data metrics
------------------------------------
Another feature of the :py:mod:`phys2denoise` package is the ability to export the computed physiological data metrics to a file, with various parameters.
This can be done using the :py:func:`export_metric` function, which provides the following capabilities:

- Exporting the computed metrics, resampled at the TR of the fMRI data, along with the original data.
- Flagging if the exported data is the convolved version or if the metric contains lags of itself, resulting in appropriate file naming.
- Defining the output file extension and file prefix.
- Defining the number of timepoints to be considered.

The following example shows how to export the computed respiratory variance time using a Physio object.

.. code-block:: python

  RVT = respiratory_variance_time(
      resp.data, resp.peaks, resp.troughs, resp.fs, lags=(0, 4, 8, 12)
  )

  export_metric(
      RVT,
      resp.fs,
      tr=1.5,
      fileprefix="data/pk-p2d/sub-007_ses-05_task-rest_run-01_RVT",
      ntp=400,
      is_convolved=False,
      has_lags=True,
  )
