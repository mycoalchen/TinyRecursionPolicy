import importlib
import inspect
from typing import Optional
import torch
import random


def load_model_class(identifier: str, prefix: str = "models."):
    module_path, class_name = identifier.split("@")

    # Import the module
    module = importlib.import_module(prefix + module_path)
    cls = getattr(module, class_name)

    return cls


def get_model_source_path(identifier: str, prefix: str = "models."):
    module_path, class_name = identifier.split("@")

    module = importlib.import_module(prefix + module_path)
    return inspect.getsourcefile(module)


def sample_dataset_batch_indices(
    num_batches: int,
    rank: int,
    world_size: int,
    subset_seed: int,
    subset_fraction: Optional[float] = None,
    subset_max_batches: Optional[int] = None,
):
    """
    Pick a random sample of batch indices from [0, num_batches), with a hard size cap of max_batches, and broadcast it to all processes
    Return None if we should just use the entire dataset
    """

    if (subset_fraction is None and subset_max_batches is None) or (
        subset_fraction is None and subset_max_batches >= num_batches
    ):
        return None

    subset_size = num_batches
    if subset_fraction is not None:
        subset_size = max(1, int(num_batches * subset_fraction))
    if subset_max_batches is not None:
        subset_size = min(subset_size, subset_max_batches)

    if rank == 0:
        rng = random.Random(subset_seed)
        chosen = rng.sample(range(num_batches), k=subset_size)

    # Use torch.distributed to broadcast chosen batch indices
    if world_size > 1:
        if rank == 0:
            chosen_tensor = torch.tensor(chosen, dtype=torch.int64, device="cuda")
        else:
            chosen_tensor = torch.empty(subset_size, dtype=torch.int64, device="cuda")
        torch.distributed.broadcast(chosen_tensor, src=0)
        chosen = chosen_tensor.cpu().tolist()
    
    return set(chosen)