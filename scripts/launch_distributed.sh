#!/usr/bin/env bash
# Two-node torchrun launcher for train.py --backend peft --distributed (FSDP full-shard).
#
# Run this SAME command on both nodes; only NODE_RANK differs. torchrun's c10d rendezvous
# handles the elastic handshake, so there's no separate "master" launcher script.
#
# Required env vars:
#   MASTER_ADDR   reachable hostname/IP of node 0 (e.g. the first GB10's InfiniBand/NIC address)
#   NODE_RANK     0 on the first node, 1 on the second
# Optional:
#   MASTER_PORT   default 29500
#   NPROC_PER_NODE  GPUs to use per node, default 1 (one GB10 Blackwell GPU per node)
#
# Example (run on node 0):
#   MASTER_ADDR=10.0.0.1 NODE_RANK=0 bash scripts/launch_distributed.sh
# Example (run on node 1):
#   MASTER_ADDR=10.0.0.1 NODE_RANK=1 bash scripts/launch_distributed.sh
set -euo pipefail

: "${MASTER_ADDR:?set MASTER_ADDR to node 0's reachable address}"
: "${NODE_RANK:?set NODE_RANK to 0 on node 0 and 1 on node 1}"
MASTER_PORT="${MASTER_PORT:-29500}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
NNODES="${NNODES:-2}"

export USE_TF=0 USE_FLAX=0
mkdir -p out

torchrun \
  --nnodes="${NNODES}" \
  --nproc-per-node="${NPROC_PER_NODE}" \
  --node-rank="${NODE_RANK}" \
  --rdzv-id=finscale \
  --rdzv-backend=c10d \
  --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
  train.py \
  --backend peft \
  --distributed \
  --input data/train_raw.jsonl \
  --data-dir data/nvidia \
  --model "${MODEL:-Qwen/Qwen2.5-14B-Instruct}" \
  --epochs "${EPOCHS:-3}" \
  --batch-size "${BATCH_SIZE:-1}" \
  --grad-accum-steps "${GRAD_ACCUM_STEPS:-4}" \
  --max-seq-length "${MAX_SEQ_LENGTH:-2048}" \
  --learning-rate "${LEARNING_RATE:-2e-4}" \
  --lora-r "${LORA_R:-16}" \
  --lora-alpha "${LORA_ALPHA:-32}" \
  --eval-steps "${EVAL_STEPS:-25}" \
  --out-dir out 2>&1 | tee "out/train_peft_rank${NODE_RANK}.log"
