"""XD-RobustPVOS / DAG-RS reference implementation.

Cross-Domain and Compound-Degradation Robust Promptable Video Object Segmentation
via Downstream-Agreement-Guided Restoration (training-free).

Package layout
--------------
xdrp.degradations : reproducible compound-degradation engine
xdrp.operators    : training-free restoration operator bank (ROB)
xdrp.evidence     : downstream agreement scores + Dirichlet fusion + degradation profile
xdrp.sam_backend  : promptable segmentation backends (SAM 2.1 image predictor / dummy)
xdrp.flow         : optical flow utilities
xdrp.pipeline     : the DAG-RS inference loop
xdrp.metrics      : J / F metrics
xdrp.iqa          : image quality statistics
xdrp.stats        : bootstrap CI, Wilcoxon, Holm, Spearman
xdrp.datasets     : DAVIS / MOSE / YouTube-VOS readers
xdrp.viz          : qualitative panels and trajectory plots
"""

__version__ = "0.1.0"
