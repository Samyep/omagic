import os
import random
import numpy as np


def set_seeds(seed: int) -> None:
    """Seed Python, NumPy, and TensorFlow (if present) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"

    try:
        import tensorflow as tf  # type: ignore

        tf.random.set_seed(seed)
        try:
            tf.keras.utils.set_random_seed(seed)
        except Exception:
            pass
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass
    except ImportError:
        # TensorFlow is optional and imported lazily in workers.
        pass
