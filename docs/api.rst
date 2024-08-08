.. _api_ref:
API
===

.. py:module:: phys2denoise

Cardiac data
------------

.. automodule:: phys2denoise.metrics.cardiac
   :members: heart_rate, heart_rate_variability, heart_beat_interval, cardiac_phase

Respiratory data
----------------

.. automodule:: phys2denoise.metrics.chest_belt
   :members: respiratory_cariance_time, respiratory_pattern_variability, env, respiratory_variance, respiratory_phase

Multimodal data
---------------

.. autofunction:: phys2denoise.multimodal.retroicor

Response functions
------------------

.. automodule:: phys2denoise.metrics.responses
   :members: crf, icrf, rrf

Utilities
---------

.. automodule:: phys2denoise.metrics.utils
   :members: print_metric_call, mirrorpad_1d, rms_envelope_1d, apply_lags, apply_function_in_sliding_window, convolve_and_rescale, export_metric
