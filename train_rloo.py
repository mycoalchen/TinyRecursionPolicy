import datetime
from typing import Optional, Any, Sequence, List, Dict
from dataclasses import dataclass
import os
import math
import copy

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch import nn
import torch.utils.checkpoint as checkpoint
import tqdm
import wandb
import hydra
import pydantic
from omegaconf import DictConfig
from adam_atan2_pytorch import AdamAtan2
from einops import repeat

from puzzle_dataset import PuzzleDatasetMetadata
from utils.functions import load_model_class, sample_dataset_batch_indices
from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed
from pretrain import create_dataloader, load_checkpoint, save_train_state, save_code_and_config

from rloo import run_forward_step


class LossConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")
    name: str


class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")
    name: str
    loss: LossConfig
    halt_max_steps: int = 16
    H_cycles: int = 3


class RLOOParams(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")
    epsilon: float = 0.2
    reward_type: str = "cell_match"
    entropy_coef: float


class RLOOConfig(pydantic.BaseModel):
    arch: ArchConfig
    rloo: RLOOParams
    data_paths: List[str]
    data_paths_test: List[str] = []

    global_batch_size: int
    num_rollouts_per_input: int = 8
    training_steps: int

    lr: float
    lr_min_ratio: float
    lr_warmup_steps: int

    beta1: float
    beta2: float
    weight_decay: float

    puzzle_emb_lr: float
    puzzle_emb_weight_decay: float

    seed: int = 0
    use_kl: bool = False

    project_name: Optional[str] = None
    run_name: Optional[str] = None
    load_checkpoint: Optional[str] = None
    checkpoint_path: Optional[str] = None

    # Eval params
    evaluators: List[Any] = []
    eval_interval: int = 100
    eval_max_batches: Optional[int] = None
    eval_dataset_fraction: Optional[float] = None
    eval_save_outputs: List[str] = []
    checkpoint_every_eval: bool = False
    eval_batch_size: int

    ema: bool = False
    ema_rate: float = 0.999
    freeze_weights: bool = False


@dataclass
class TrainState:
    model: nn.Module
    optimizers: Sequence[torch.optim.Optimizer]
    optimizer_lrs: Sequence[float]
    model_state: Any
    training_step: int
    total_steps: int

# --- Model Creation ---


def create_model(config: RLOOConfig, train_metadata: PuzzleDatasetMetadata):
    # Calculate the actual batch size the model will see (Batch * Rollouts)
    expanded_batch_size = config.global_batch_size * config.num_rollouts_per_input
    print(f"Batch size {config.global_batch_size} * num rollouts {config.num_rollouts_per_input} = expanded batch size {expanded_batch_size}")
    model_cfg = dict(
        **config.arch.model_dump(exclude={'loss'}),
        batch_size=expanded_batch_size,
        vocab_size=train_metadata.vocab_size,
        seq_len=train_metadata.seq_len,
        num_puzzle_identifiers=train_metadata.num_puzzle_identifiers,
        causal=False,
    )

    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)

    with torch.device("cuda"):
        model: nn.Module = model_cls(model_cfg)
        model = loss_head_cls(model, config.arch.loss.loss_type,
                              config.arch.loss.act_loss_weight)
        if "DISABLE_COMPILE" not in os.environ:
            model = torch.compile(model)
        load_checkpoint(model, config)

    if config.arch.model_dump().get("puzzle_emb_ndim", 0) == 0:
        optimizers = [AdamAtan2(model.parameters(
        ), lr=config.lr, weight_decay=config.weight_decay, betas=(config.beta1, config.beta2))]
        optimizer_lrs = [config.lr]
    else:
        optimizers = [
            CastedSparseEmbeddingSignSGD_Distributed(
                model.model.puzzle_emb.buffers(), lr=config.puzzle_emb_lr, weight_decay=config.puzzle_emb_weight_decay, world_size=1
            ),
            AdamAtan2(
                model.parameters(), lr=config.lr, weight_decay=config.weight_decay, betas=(config.beta1, config.beta2)
            ),
        ]
        optimizer_lrs = [config.puzzle_emb_lr, config.lr]

    return model, optimizers, optimizer_lrs


def compute_lr(base_lr: float, config: RLOOConfig, train_state: TrainState):
    current_step = train_state.training_step
    num_warmup_steps = config.lr_warmup_steps
    num_training_steps = config.training_steps
    min_ratio = config.lr_min_ratio

    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / \
        float(max(1, num_training_steps - num_warmup_steps))
    return base_lr * (min_ratio + max(0.0, (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * 0.5 * 2.0 * progress))))

# --- Evaluation Logic (Ported from pretrain.py) ---

def evaluate(
    config: RLOOConfig,
    train_state: TrainState,
    eval_loader: torch.utils.data.DataLoader,
    eval_metadata: PuzzleDatasetMetadata,
    rank: int,
    world_size: int,
    subset_sampling_seed: int,
    use_subset_of_data: bool = True,
):
    reduced_metrics = None

    if rank == 0:
        print(f"Starting Evaluation at step {train_state.training_step}...")

    # we might not want to do eval with the entire test dataset every time
    if use_subset_of_data:
        data_subset_batch_indices = sample_dataset_batch_indices(
            len(eval_loader),
            rank,
            world_size,
            subset_sampling_seed,
            config.eval_dataset_fraction,
            config.eval_max_batches,
        )
    else:
        data_subset_batch_indices = None

    with torch.inference_mode():
        return_keys = set(config.eval_save_outputs) # Empty
        return_keys.add("logits")
        return_keys.add("q_halt_logits")

        set_ids = {k: idx for idx, k in enumerate(eval_metadata.sets)} # 'all': 0

        save_outputs = {}
        metric_keys = []
        metric_values = None

        processed_batches = 0

        for batch_idx, (set_name, batch, _) in enumerate(eval_loader):
            if (
                data_subset_batch_indices is not None
                and batch_idx not in data_subset_batch_indices
            ):
                continue

            processed_batches += 1
            if rank == 0 and processed_batches % 10 == 0:
                print(f"Processing eval batch {processed_batches}: {set_name}")

            # inputs: (B, seq_len), labels: (B, seq_len), puzzle_identifiers: (B)
            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.device("cuda"):
                state = train_state.model.model.initial_state_eval(batch)
            
            # Forward
            outputs = None
            for inference_step in range(config.arch.halt_max_steps):
                state, loss, metrics, outputs, all_finish = train_state.model(
                    state=state, batch=batch, prev_outputs=outputs, return_keys=return_keys
                )
                if all_finish:
                    break

            for collection in (batch, outputs):
                for k, v in collection.items():
                    if k in config.eval_save_outputs:
                        save_outputs.setdefault(k, [])
                        save_outputs[k].append(v.cpu())  # Move to CPU for saving GPU memory

            del state, loss, outputs, batch, all_finish

            # Aggregate metrics
            set_id = set_ids[set_name]

            if metric_values is None:
                metric_keys = list(
                    sorted(metrics.keys())
                )  # Sort keys to guarantee all processes use the same order.
                metric_values = torch.zeros(
                    (len(set_ids), len(metrics.values())),
                    dtype=torch.float32,
                    device="cuda",
                )

            metric_values[set_id] += torch.stack([metrics[k] for k in metric_keys])

            del metrics

        # concatenate save preds
        save_outputs = {k: torch.cat(v, dim=0) for k, v in save_outputs.items()}

        # Save preds
        if config.checkpoint_path is not None and len(save_outputs):
            # Each rank save predictions independently
            os.makedirs(os.path.dirname(config.checkpoint_path), exist_ok=True)
            torch.save(
                save_outputs,
                os.path.join(
                    config.checkpoint_path, f"step_{train_state.training_step}_all_preds.{rank}"
                ),
            )

        del save_outputs

        # Reduce to rank 0
        if metric_values is not None:
            if world_size > 1:
                dist.reduce(metric_values, dst=0)

            if rank == 0:
                reduced_metrics = metric_values.cpu().numpy()
                reduced_metrics = {
                    set_name: {
                        metric_name: reduced_metrics[set_id, metric_id]
                        for metric_id, metric_name in enumerate(metric_keys)
                    }
                    for set_id, set_name in enumerate(set_ids)
                }

                # Postprocess
                for set_name, m in reduced_metrics.items():
                    count = max(m.pop("count"), 1)  # Avoid NaNs
                    reduced_metrics[set_name] = {k: v / count for k, v in m.items()}

    return reduced_metrics


def eval_log_and_checkpoint(
    RANK: int,
    WORLD_SIZE: int,
    config: RLOOConfig,
    train_state: TrainState,
    eval_loader: torch.utils.data.DataLoader,
    eval_metadata: PuzzleDatasetMetadata,
    checkpoint: bool,
    use_subset_of_data: bool = False,
):
    if RANK == 0:
        print("EVALUATE")
    
    # Switch to eval mode
    train_state.model.eval()
    
    metrics = evaluate(
        config,
        train_state,
        eval_loader,
        eval_metadata,
        rank=RANK,
        world_size=WORLD_SIZE,
        subset_sampling_seed=torch.randint(low=0, high=100, size=(1,)).item(),
        use_subset_of_data=use_subset_of_data,
    )

    if RANK == 0 and metrics is not None:
        # Flatten metrics for wandb
        wandb_metrics = {}
        for set_name, values in metrics.items():
            for k, v in values.items():
                wandb_metrics[f"test/{set_name}/{k}"] = v
        print(wandb_metrics)
        wandb.log(wandb_metrics, step=train_state.training_step)

    if checkpoint and RANK == 0:
        print("SAVE CHECKPOINT")
        save_train_state(config, train_state)
    
    # Switch back to train mode
    train_state.model.train()


# --- RLOO Training Logic ---


def train_batch_rloo(config: RLOOConfig, train_state: TrainState, batch: Dict[str, Any]):
    train_state.training_step += 1

    # 1. Setup Batch
    inputs = batch["inputs"].cuda()
    pids = batch["puzzle_identifiers"].cuda()
    gt_labels = batch["labels"].cuda()

    B = inputs.shape[0]
    G = config.num_rollouts_per_input
    N = B * G

    # Expand for K rollouts
    group_inputs = repeat(inputs, 'b ... -> (b g) ...', g=G)
    group_pids = repeat(pids, 'b ... -> (b g) ...', g=G)

    inner = train_state.model.model.inner
    H_cycles = inner.config.H_cycles
    Halt_Max = inner.config.halt_max_steps

    # Lists to store gradients components
    log_probs_list = []
    masks_list = []

    all_metrics = {"z_L_entropy": [], "z_L_sigma": [], "z_H_entropy": [], "z_H_sigma": []}
    halt_logits_list = []

    # 2. Forward Pass (Online Collection)
    input_embeddings = inner._input_embeddings(group_inputs, group_pids)
    latents = inner.initial_latents(N)
    z_L, z_H = latents.z_L, latents.z_H
    seq_info = dict(cos_sin=inner.rotary_emb()
                    if hasattr(inner, "rotary_emb") else None)

    active_mask = torch.ones(N, dtype=torch.bool, device="cuda")
    act_step_num = torch.zeros(N, dtype=torch.long, device="cuda")
    final_z_H = z_H

    # --- ROLLOUT LOOP ---
    for step in range(Halt_Max * H_cycles):
        current_macro_step = min(step // H_cycles, Halt_Max - 1)
        act_step_num = torch.full((inputs.shape[0] * config.num_rollouts_per_input,),
                                    current_macro_step,
                                    dtype=torch.long,
                                    device="cuda")

        # Step returns log_prob attached to the graph
        new_z_L, new_z_H, step_log_prob, halt_dist, exploration_metrics, q_logits = checkpoint.checkpoint(
            run_forward_step,
            inner,
            z_L,
            z_H,
            input_embeddings,
            seq_info,
            act_step_num,
            use_reentrant=False
        )

        for k, v in exploration_metrics.items():
            all_metrics[k].append(v)
        halt_logits_list.append(q_logits)

        halt_action = halt_dist.sample()
        halt_log_prob = halt_dist.log_prob(halt_action)

        total_step_log_prob = step_log_prob + halt_log_prob

        # Store valid log probs [N]
        log_probs_list.append(total_step_log_prob)
        masks_list.append(active_mask.float())

        # Update State
        should_halt = (halt_action == 0)
        active_mask = active_mask & (~should_halt)

        # Update hidden states (Detached for next step input, but new_z graphs still connected for log_prob grads)
        z_L = new_z_L.detach()
        z_H = new_z_H.detach()

        final_z_H = torch.where(
            active_mask[:, None, None], new_z_H, final_z_H)

        
        if active_mask.sum() == 0:
            break

    # 3. Reward Calculation
    lm_logits = inner.lm_head(final_z_H)[:, inner.puzzle_emb_len:, :]
    preds = torch.argmax(lm_logits, dim=-1)

    group_gts = repeat(gt_labels, 'b l -> (b g) l', g=G)
    valid_mask = (group_inputs == 1)  # Mask for Sudoku empty cells
    is_correct = (preds == group_gts) & valid_mask
    scored_total = valid_mask.sum(dim=1).float().clamp(min=1.0)
    seq_is_correct = (is_correct.sum(dim=1) == valid_mask.sum(dim=1))

    # Raw Rewards [B*G]
    scores = is_correct.sum(dim=1).float() / scored_total

    # 4. RLOO Baseline Calculation
    # Reshape to [B, G]
    scores_view = scores.view(B, G)

    # Calculate sum of scores for each group
    sum_scores = scores_view.sum(dim=1, keepdim=True)  # [B, 1]

    # RLOO Baseline: (Sum - current) / (K - 1)
    rloo_baseline = (sum_scores - scores_view) / (G - 1)
    advantages = scores_view - rloo_baseline
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    advantages = advantages.view(-1)  # Flatten back to [N]

    # 5. Compute Loss
    # Loss = - (Advantage * Sum_of_Log_Probs)
    # We sum log probs over the valid trajectory

    # Stack lists: [Time, N]
    step_log_probs = torch.stack(log_probs_list, dim=0)
    step_masks = torch.stack(masks_list, dim=0)

    # Calculate Mean Entropy for Bonus (averaged over time and batch)
    # We use z_L_entropy and z_H_entropy from the dictionary
    avg_z_L_entropy = torch.stack(all_metrics["z_L_entropy"]).mean()
    avg_z_H_entropy = torch.stack(all_metrics["z_H_entropy"]).mean()
    total_entropy = avg_z_L_entropy + avg_z_H_entropy

    # Multiply by mask
    masked_log_probs = step_log_probs * step_masks
    trajectory_log_prob = masked_log_probs.sum(dim=0)

    # Sum over time [N] -> Total trajectory log prob
    trajectory_log_prob = masked_log_probs.sum(dim=0)

    # Loss = - (RL_Objective + Entropy_Bonus)
    # We maximize entropy, so we minimize -entropy
    pg_loss = -(trajectory_log_prob * advantages.detach()).mean()
    entropy_loss = -config.rloo.entropy_coef * total_entropy
    loss = pg_loss + entropy_loss

    # Auxiliary metrics
    # Gather logits for active steps
    avg_steps = step_masks.sum(dim=0).mean().item()

    # 6. Backward & Opt
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(train_state.model.parameters(), 1.0)
    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
        print(f"Skipping step! Gradient is {grad_norm}")
        for optim in train_state.optimizers:
            optim.zero_grad()
        return {"loss": loss.item(), "reward": scores.mean().item(), "lr": 0.0}

    lr_this_step = None
    for optim, base_lr in zip(train_state.optimizers, train_state.optimizer_lrs):
        lr_this_step = compute_lr(base_lr, config, train_state)
        for param_group in optim.param_groups:
            param_group["lr"] = lr_this_step
        optim.step()
        optim.zero_grad()

    metrics_to_log = {
        "loss": loss.item(), 
        "reward": scores.mean().item(), 
        "lr": lr_this_step,
        "metrics/steps": avg_steps,
        "metrics/z_L_sigma": torch.stack(all_metrics["z_L_sigma"]).mean().item(),
        "metrics/z_L_entropy": avg_z_L_entropy.item(),
        "metrics/z_H_sigma": torch.stack(all_metrics["z_H_sigma"]).mean().item(),
        "metrics/z_H_entropy": avg_z_H_entropy.item(),
    }
    
    print(metrics_to_log)
    return metrics_to_log

# --- Main Launch ---


def load_synced_config(hydra_config: DictConfig, rank: int, world_size: int) -> RLOOConfig:
    objects = [None]
    if rank == 0:
        config = RLOOConfig(**hydra_config)
        if config.project_name is None:
            config.project_name = "GTRM-RLOO"
        if config.checkpoint_path is None:
            config.checkpoint_path = os.path.join(
                "checkpoints", config.project_name, config.run_name or "run")
        objects = [config]
    if world_size > 1:
        dist.broadcast_object_list(objects, src=0)
    return objects[0]


@hydra.main(config_path="config", config_name="cfg_rloo", version_base=None)
def launch(hydra_config: DictConfig):

    RANK = 0
    WORLD_SIZE = 1
    CPU_PROCESS_GROUP = None

    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        
        # CPU GLOO process group for eval synchronization
        CPU_PROCESS_GROUP = dist.new_group(backend="gloo")

    config = load_synced_config(hydra_config, rank=RANK, world_size=WORLD_SIZE)
    torch.random.manual_seed(config.seed + RANK)

    # Training DataLoader
    train_loader, train_metadata = create_dataloader(
        config, "train", RANK, WORLD_SIZE,
        epochs_per_iter=1,
        global_batch_size=config.global_batch_size,
        test_set_mode=False
    )
    
    # Eval DataLoader
    try:
        eval_loader, eval_metadata = create_dataloader(
            config,
            "test",
            rank=RANK, 
            world_size=WORLD_SIZE,
            test_set_mode=True,
            epochs_per_iter=1,
            global_batch_size=config.eval_batch_size,
        )
    except Exception as e:
        print(f"NO EVAL DATA FOUND: {e}")
        eval_loader = eval_metadata = None

    train_state = TrainState(
        model=None, optimizers=[], optimizer_lrs=[], model_state=None,
        training_step=0, total_steps=config.training_steps
    )

    model, optimizers, optimizer_lrs = create_model(config, train_metadata)
    train_state.model = model
    train_state.optimizers = optimizers
    train_state.optimizer_lrs = optimizer_lrs

    if RANK == 0:
        wandb.init(project=config.project_name,
                   name=config.run_name, config=config.model_dump())
        save_code_and_config(config)

    iter_count = 0
    progress_bar = tqdm.tqdm(
        total=config.training_steps) if RANK == 0 else None

    # Initial Eval
    # if eval_loader is not None:
    #      eval_log_and_checkpoint(
    #         RANK, WORLD_SIZE, config, train_state, eval_loader, eval_metadata, 
    #         checkpoint=False, use_subset_of_data=True
    #     )

    while train_state.training_step < config.training_steps:
        iter_count += 1
        if RANK == 0:
            progress_bar.set_description(f"Iter {iter_count}")

        train_state.model.train()

        for set_name, batch, global_batch_size in train_loader:
            metrics = train_batch_rloo(config, train_state, batch)

            if RANK == 0:
                wandb.log(metrics, step=train_state.training_step)
                progress_bar.set_postfix(
                    loss=f"{metrics['loss']:.4f}", reward=f"{metrics['reward']:.4f}")
                progress_bar.update(1)

            # Periodic Evaluation
            if config.eval_interval > 0 and train_state.training_step % config.eval_interval == 1 and eval_loader is not None:
                eval_log_and_checkpoint(
                    RANK, 
                    WORLD_SIZE, 
                    config, 
                    train_state, 
                    eval_loader, 
                    eval_metadata, 
                    checkpoint=config.checkpoint_every_eval,
                    use_subset_of_data=True
                )

            if train_state.training_step >= config.training_steps:
                break

        if RANK == 0:
            save_train_state(config, train_state)
            
    # Final Full Evaluation
    if eval_loader is not None:
        print("FINAL EVALUATION WITH FULL TEST DATASET")
        eval_log_and_checkpoint(
            RANK, 
            WORLD_SIZE, 
            config, 
            train_state, 
            eval_loader, 
            eval_metadata, 
            checkpoint=True,
            use_subset_of_data=False
        )

    if RANK == 0:
        print("Training Complete.")
    wandb.finish()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    launch()