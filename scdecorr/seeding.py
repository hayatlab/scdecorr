# seed_everything.py
import os
import random
import warnings

def seed_everything(seed: int = 42, cuda: bool = True, workers: bool = False):
    """
    Set random seeds for Python, NumPy, PyTorch (CPU/CUDA)
    - seed: integer seed
    - cuda: set torch.cuda seeds (if torch is available)
    - workers: if True, attempts to set up worker seeding for DataLoader workers (PyTorch)
    Returns: None
    """
    # 1) Python stdlib
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # 2) NumPy
    import numpy as np
    np.random.seed(seed)

    # 3) PyTorch (CPU + optional CUDA)
    try:
        import torch
        torch.manual_seed(seed)
        # ensure all devices (if available)
        try:
            if cuda and torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass

        # make cuDNN deterministic where possible (may slow training)
        # Note: some ops are still nondeterministic even with these flags.
        #torch.backends.cudnn.deterministic = True
        #torch.backends.cudnn.benchmark = False

        # Optionally prepare worker_init_fn for DataLoader workers
        if workers:
            # This helper returns a worker_init_fn you can pass to DataLoader(worker_init_fn=...)
            def _worker_init_fn(worker_id):
                # combine global seed + worker_id to derive per-worker seed
                worker_seed = seed + worker_id
                random.seed(worker_seed)
                np.random.seed(worker_seed)
                torch.manual_seed(worker_seed)
            # attach to module so user can import it:
            # from seed_everything import worker_init_fn
            globals()['worker_init_fn'] = _worker_init_fn

    except Exception:
        pass

    # 6) Warn user about limitations
    warnings.warn(
        "Seeds set for Python, NumPy and available ML libs (PyTorch/TF/JAX). "
        "Full bitwise reproducibility is not guaranteed across different hardware/drivers/versions."
    )
