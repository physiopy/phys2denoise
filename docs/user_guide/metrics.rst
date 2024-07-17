.. _usage_metrics:

Computing physiological data metrics
-------------------------------------
The :py:mod:`phys2denoise` package provides a set of functions to compute physiological data metrics. The set of supported metrics
includes:

- Cardiac metrics
    - Cardiac phase
    - Heart rate
    - Heart rate variability
    - Heart beat interval
- Respiratory metrics
    - Respiratory phase
    - Respiratory variance
    - Respiratory pattern variability
    - Envelope
- Multimodal metrics
    - RETROICOR

All of the metrics computation function definitions, descriptions and references can be found in :ref:`api_ref`.


Using a Physio object
#####################

Physiological data metrics can be easily computed using Physio objects, from the :py:mod:`physutils` module,
on which the physiological data will be loaded.

The following example shows how to compute the respiratory variance time using a Physio object.

.. code-block:: python

    from physutils import io
    from phys2denoise.metrics.chest_belt import respiratory_variance_time
    # peakdet is an example package that provides peak/trough detection for the respiratory signal
    from peakdet import operations

    # Load the physiological data
    sample_rate = 1000
    physio = io.load_physio('path/to/physiological/data', fs=sample_rate)

    # Peak/trough detection for the respiratory signal, using the peakdet package
    physio = operations.peakfind_physio(physio)

    # Compute RVT
    physio, rvt = respiratory_variance_time(physio)

:py:func:`respiratory_variance_time` returns a tuple with the updated Physio object and the computed respiratory variance time.

:py:mod:`peakdet` is used in this example as it is also compatible with the Physio object. However, any other peak/trough detection
package can be used. In this case, the peak and trough values should be stored in the Physio object manually as follows:

.. code-block:: python

    # Store the peak and trough values in the Physio object
    physio._metadata["peaks"] = peaks
    physio._metadata["troughs"] = troughs

The benefit of using a Physio object other than the encapsulation of all the desired parameters in a single object is the fact that
the object retains a history of all the operations performed on it. This allows for easy debugging and reproducibility of the results.
For further information refer to the :py:mod:`physutils` documentation.

Without using a Physio object
#############################

However, if the use of :py:mod:`physutils` is not preferred, the metrics can be also computed without it. The following
example shows how to compute the heart rate and the heart rate variability using the :py:mod:`phys2denoise` package.

.. code-block:: python

    from phys2denoise.metrics.chest_belt import respiratory_variance_time

    # Given that the respiratory signal is stored in `data`, the peaks in `peaks`, the troughs in `troughs`
    # and the sample rate in `sample_rate`
    _, rvt = respiratory_variance_time(physio, peaks, troughs, sample_rate)
