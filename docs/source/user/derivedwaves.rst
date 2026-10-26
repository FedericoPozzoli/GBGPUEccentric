Extending GBGPU Waveforms
~~~~~~~~~~~~~~~~~~~~~~~~~

The base class :class:`gbgpu.gbgpu.GBGPUBase` can be inherited in order to build other waveforms in the model of FastGB. To do this, there are required methods that need to be added to the new waveform class. They are described in the Abstract Base Class :class:`gbgpu.gbgpu.GBGPUBase`. After the base class, waveform models that have already been extended beyond the base are described. 


``GBGPUBase`` base class
-----------------------------

.. autoclass:: gbgpu.gbgpu.GBGPUBase
    :members:
    :show-inheritance:


Third-body inclusion
-----------------------

:class:`gbgpu.gbgpu.GBGPUBase` has been extended to include a third-body in long orbit around the inner binary. This waveform was first built in `arXiv:1806.00500 <https://arxiv.org/pdf/1806.00500.pdf>`_. It was more recently used and adapted into this code base for # TODO: add arxiv. Please cite both papers, as well as the base FastGB papers, if you use this waveform. 

Third-body waveform
******************************

.. autoclass:: gbgpu.thirdbody.GBGPUThirdBody
    :members:
    :show-inheritance:


Third-body utility functions
******************************

.. autofunction:: gbgpu.thirdbody.third_body_factors

.. autofunction:: gbgpu.thirdbody.get_T2



Eccentric binaries
-----------------------

:class:`gbgpu.gbgpu.GBGPUBase` has been extended to eccentric quasi-monochromatic binaries with the multi-harmonic model of `Moore et al. 2024, MNRAS 531, 2817 (arXiv:2310.06568) <https://arxiv.org/abs/2310.06568>`_. Harmonic ``n`` is a circular FastGB source with amplitude ``A alpha_n(e)``, frequency ``n/2 f_GW + (n - 2) f_PP``, frequency derivative ``n/2 fdot_GW`` and phase ``phi + (n - 2) Phi0``. The binaries of a batch are mapped on the device to one row per harmonic and evaluated by the shared-memory C++/CUDA kernels (waveforms, likelihood, swap likelihood, global templates), with one ``N`` per harmonic; inner products between harmonics are added wherever their frequency bands overlap. Data and templates must cover every harmonic band. Please cite that paper, as well as the base FastGB papers, if you use this waveform.

Eccentric waveform
******************************

.. autoclass:: gbgpu.eccentric.GBGPUEccentric
    :members:
    :show-inheritance:

Eccentric utility functions
******************************

.. autofunction:: gbgpu.eccentric.eccentric_harmonic_amplitudes

.. autofunction:: gbgpu.eccentric.eccentric_amplitude_coefficients

.. autofunction:: gbgpu.eccentric.eccentric_to_circular_harmonics

.. autofunction:: gbgpu.eccentric.eccentricity_enhancement

.. autofunction:: gbgpu.eccentric.get_fdot_eccentric

.. autofunction:: gbgpu.eccentric.get_fPP

.. autofunction:: gbgpu.eccentric.get_masses_from_eccentric_parameters

Circular-eccentric reversible jump (Eryn)
******************************************

:mod:`gbgpu.eccentric_moves` provides `Eryn <https://github.com/mikekatz04/Eryn>`_ moves to sample circular (:class:`gbgpu.gbgpu.GBGPU`) and eccentric (:class:`gbgpu.eccentric.GBGPUEccentric`) binaries together. The models are nested: harmonic ``m`` (``m = 2`` by default) of an eccentric binary is a circular GBGPU source, and the reversible jump keeps it fixed, drawing ``(e, Phi0, fPP)`` from a proposal distribution when a circular binary is promoted and dropping them when an eccentric binary is demoted. The number of binaries can change too: :class:`gbgpu.eccentric_moves.BirthDeathMove` adds and removes circular or eccentric binaries, and :class:`gbgpu.eccentric_moves.MergeSplitMove` merges two circular binaries that are harmonics 2 and 3 of the same eccentric binary into it (or splits it). All jumps follow Eryn's leaf-count convention and can be combined (a non-zero ``log_prior_odds`` of the circular-eccentric jump must be matched by ``log_leaf_weights`` in the others). The module also contains in-model redraws of ``(e, Phi0, fPP)`` and of whole binaries, a differential-evolution move, the ``(psi, phi) -> (psi + pi/2, phi + pi)`` symmetry move, likelihoods for walkers holding one or many binaries, and Savage-Dickey estimators of the Bayes factor. It requires ``eryn`` (``gbgpu[sampling]``). ``examples/eccentric_rj_pe.py`` / ``examples/eccentric_rj_pe.ipynb`` (circular <-> eccentric jump), ``examples/eccentric_multi_source_pe.py`` / ``examples/eccentric_multi_source_pe.ipynb`` (birth/death, merge/split) and ``examples/eccentric_global_fit_pe.py`` / ``examples/eccentric_global_fit_pe.ipynb`` (one eccentric and three circular binaries in noise, starting from a circular search) run parameter-estimation tests.

.. autoclass:: gbgpu.eccentric_moves.CircularEccentricRJMove
    :members:
    :show-inheritance:

.. autoclass:: gbgpu.eccentric_moves.BirthDeathMove
    :members:
    :show-inheritance:

.. autoclass:: gbgpu.eccentric_moves.MergeSplitMove
    :members:
    :show-inheritance:

.. autoclass:: gbgpu.eccentric_moves.CircularEccentricMap
    :members:

.. autoclass:: gbgpu.eccentric_moves.HarmonicPairMap
    :members:

.. autoclass:: gbgpu.eccentric_moves.GaussianMixtureProposal
    :members:

.. autoclass:: gbgpu.eccentric_moves.DistributionProposal
    :members:

.. autoclass:: gbgpu.eccentric_moves.PromotedProposal
    :members:

.. autoclass:: gbgpu.eccentric_moves.EccentricRedrawMove
    :members:
    :show-inheritance:

.. autoclass:: gbgpu.eccentric_moves.LeafRedrawMove
    :members:
    :show-inheritance:

.. autoclass:: gbgpu.eccentric_moves.PolarizationPhaseFlipMove
    :members:
    :show-inheritance:

.. autoclass:: gbgpu.eccentric_moves.DifferentialEvolutionMove
    :members:
    :show-inheritance:

.. autoclass:: gbgpu.eccentric_moves.CircularEccentricLikelihood
    :members:

.. autoclass:: gbgpu.eccentric_moves.MultiSourceLikelihood
    :members:

.. autofunction:: gbgpu.eccentric_moves.circular_to_gbgpu

.. autofunction:: gbgpu.eccentric_moves.eccentric_to_gbgpu

.. autofunction:: gbgpu.eccentric_moves.eccentric_model_probability

.. autofunction:: gbgpu.eccentric_moves.savage_dickey_log_bayes_factor

.. autofunction:: gbgpu.eccentric_moves.rao_blackwell_savage_dickey_log_bayes_factor
