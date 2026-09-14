import logging
import os
from typing import  List, Optional, Sequence, Tuple
import numpy as np


def random_train_valid_split(
    items: Sequence,
    valid_fraction: float,
    seed: int,
    work_dir: str,
    prefix: Optional[str] = None,
) -> Tuple[List, List]:
    assert 0.0 < valid_fraction < 1.0

    size = len(items)
    # guarantee at least one validation, mostly for tests with tiny fitting databases
    assert size > 1
    train_size = min(size - int(valid_fraction * size), size - 1)

    indices = list(range(size))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    if len(indices[train_size:]) < 10:
        logging.info(
            f"Using random {100 * valid_fraction:.0f}% of training set for validation with following indices: {indices[train_size:]}"
        )
    else:
        # Save indices to file (optionally prefixed with experiment name)
        filename = f"valid_indices_{seed}.txt"
        if prefix is not None and len(prefix) > 0:
            filename = f"{prefix}_" + filename
        path = os.path.join(work_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            for index in indices[train_size:]:
                f.write(f"{index}\n")

        logging.info(
            f"Using random {100 * valid_fraction:.0f}% of training set for validation with indices saved in: {path}"
        )

    return (
        [items[i] for i in indices[:train_size]],
        [items[i] for i in indices[train_size:]],
    )
