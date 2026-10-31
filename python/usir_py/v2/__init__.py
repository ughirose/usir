"""USIR v2.0 Paper-Grade Architecture Package."""

from .adaptive_baoab import (
    AdaptiveBAOABIntegrator,
    ClauseTwistedSMC,
    JacobianLogitProjector,
)
from .hth_lexicon import (
    CLOSED_ADVERSATIVE_CONJUNCTIONS,
    CLOSED_LOGICAL_MARKERS,
    CLOSED_NEGATION_MARKERS,
    HTHEntry,
    HTHLexicon,
    MusicNoteEvent,
    TokenAnalysis,
    clz30,
    compute_sensory_bridge,
    discrete_jump_distance,
    extract_music_functor_signals,
    lca_depth_30bit,
    poincare_hyperbolic_distance,
)
from .thermodynamics_so8 import (
    HamiltonFristonResonance,
    KacMoreauYosidaThermodynamics,
    SO8PullbackMetric,
)
from .wavelet_hermite_encoder import (
    MCAResult,
    RQAResult,
    WaveletDecompositionResult,
    WaveletHermiteEncoder,
    build_haar_orthonormal_matrix,
    compute_rqa_metrics,
    extract_24d_v2,
    orthogonal_wavelet_decompose_1d,
    solve_mca_fourier_soliton,
    wiener_hermite_polynomials,
)

__all__ = [
    # Module 1
    "clz30",
    "lca_depth_30bit",
    "discrete_jump_distance",
    "poincare_hyperbolic_distance",
    "compute_sensory_bridge",
    "CLOSED_NEGATION_MARKERS",
    "CLOSED_ADVERSATIVE_CONJUNCTIONS",
    "CLOSED_LOGICAL_MARKERS",
    "HTHEntry",
    "TokenAnalysis",
    "MusicNoteEvent",
    "HTHLexicon",
    "extract_music_functor_signals",
    # Module 2
    "WaveletDecompositionResult",
    "MCAResult",
    "RQAResult",
    "wiener_hermite_polynomials",
    "build_haar_orthonormal_matrix",
    "orthogonal_wavelet_decompose_1d",
    "solve_mca_fourier_soliton",
    "compute_rqa_metrics",
    "WaveletHermiteEncoder",
    "extract_24d_v2",
    # Module 3
    "SO8PullbackMetric",
    "KacMoreauYosidaThermodynamics",
    "HamiltonFristonResonance",
    # Module 4
    "AdaptiveBAOABIntegrator",
    "ClauseTwistedSMC",
    "JacobianLogitProjector",
]
