#!/bin/bash
#SBATCH --job-name=gtrm
#SBATCH --output=out/%A/gtrmarr_%A_%a.out
#SBATCH --error=out/%A/gtrmarr_%A_%a.err
#SBATCH --array=1-7
#SBATCH --time=36:00:00
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --constraint=VRAM_48GB
#SBATCH --cpus-per-task=4

set -euo pipefail

# Define bash arrays for the parameters
q_head_input_detached_list=(False False False False False False)
q_head_input_form_list=("first puzzle emb" "first puzzle emb" "first puzzle emb" "first puzzle emb" "first puzzle emb" "first puzzle emb")
H_deterministic_mode_list=("False" "False" "separate weights" "separate weights" "separate weights" "separate weights")
time_embed_list=(False True False True False True)
actlw_list=(1.5 1.5 1.5 1.5 1.25 1.25)

# Get values for this task
q_head_input_detached_val=${q_head_input_detached_list[$SLURM_ARRAY_TASK_ID]}
q_head_input_form_val=${q_head_input_form_list[$SLURM_ARRAY_TASK_ID]}
H_deterministic_mode_val=${H_deterministic_mode_list[$SLURM_ARRAY_TASK_ID]}
time_embed_val=${time_embed_list[$SLURM_ARRAY_TASK_ID]}
actlw_val=${actlw_list[$SLURM_ARRAY_TASK_ID]}

# Generate abbreviated run_name from parameters
# q_head_input_detached: True -> T, False -> F
# q_head_input_form: "intermediate output" -> io, "first puzzle emb" -> fpe  
# H_deterministic_mode: "separate weights" -> sw, "False" -> F
if [ "$q_head_input_detached_val" = "True" ]; then
    detached_abbrev="T"
else
    detached_abbrev="F"
fi

if [ "$q_head_input_form_val" = "intermediate output" ]; then
    form_abbrev="io"
elif [ "$q_head_input_form_val" = "first puzzle emb" ]; then
    form_abbrev="fpe"
else
    form_abbrev="unknown"
fi

if [ "$H_deterministic_mode_val" = "separate weights" ]; then
    mode_abbrev="sw"
elif [ "$H_deterministic_mode_val" = "False" ]; then
    mode_abbrev="F"
elif [ "$H_deterministic_mode_val" = "skip noise" ]; then
    mode_abbrev="sn"
else
    mode_abbrev="unknown"
fi

if [ "$time_embed_val" = "True" ]; then
    time_embed_abbrev="T"
elif [ "$time_embed_val" = "False" ]; then
    time_embed_abbrev="F"
else
    time_embed_abbrev="unknown"
fi

run_name="${detached_abbrev}_${form_abbrev}_${mode_abbrev}_${time_embed_abbrev}_${actlw_val}"

echo "Running task $SLURM_ARRAY_TASK_ID: q_head_input_detached=$q_head_input_detached_val, q_head_input_form=$q_head_input_form_val, H_deterministic_mode=$H_deterministic_mode_val, time_embed=$time_embed_val, actlw=$actlw_val, run_name=$run_name"

eval "$(conda shell.bash hook)"
conda activate trp

# Run pretrain.py with config overrides
# Use single quotes around value to force string interpretation in Hydra/YAML
python pretrain.py \
  --config-name=cfg_sudoku_mlp_gtrm \
  arch.q_head_input_detached=$q_head_input_detached_val \
  arch.q_head_input_form="$q_head_input_form_val" \
  "arch.H_deterministic_mode='$H_deterministic_mode_val'" \
  arch.time_embeddings=$time_embed_val \
  arch.loss.act_loss_weight=$actlw_val \
  run_name="$run_name"

echo "Completed task $SLURM_ARRAY_TASK_ID: q_head_input_detached=$q_head_input_detached_val, q_head_input_form=$q_head_input_form_val, H_deterministic_mode=$H_deterministic_mode_val, time_embed=$time_embed_val, actlw=$actlw_val"