# Validating 2-node FSDP on AWS (mechanics-only smoke test)

Goal: prove `train.py --distributed` + `scripts/launch_distributed.sh` actually work across two
real GPU nodes over a real network — torchrun rendezvous, cross-node NCCL, FSDP full-shard,
rank-zero-only adapter save, `scripts/benchmark_scaling.py`. This uses a **tiny model and a
synthetic dataset**, so it says nothing about the hackathon's real throughput or accuracy
numbers — it only proves the distributed code path is correct. For a real-numbers rehearsal
you'd need `p4d`/`p5` instances (A100/H100, 8 GPUs/node) and the actual 14B student + SEC data,
which is a different (10-100x pricier) exercise.

**Estimated cost:** 2x `g5.2xlarge` on-demand in `us-east-1` ≈ $2.40/hr combined. A full run
through this doc (provision → train → benchmark → teardown) should take under 2 hours ⇒ **under
$5 total** if you tear down promptly (last section — don't skip it).

All commands below run from **your own machine** (wherever you have the AWS CLI configured),
not from this repo's environment — I don't have AWS credentials here.

## 0. Prerequisites

- AWS CLI v2 installed and configured (`aws sts get-caller-identity` should return your account).
- **Check your G-instance quota now** — new/lightly-used accounts often default to 0 vCPUs for
  "Running On-Demand G and VT instances" (covers g5). This is the #1 blocker people hit.
  Find the current value either via the Service Quotas console (search "EC2" → "Running
  On-Demand G and VT instances") or by listing it yourself (the exact quota code can shift, so
  look it up rather than trusting a hardcoded one):
  ```bash
  aws service-quotas list-service-quotas --region us-east-1 --service-code ec2 \
    --query "Quotas[?contains(QuotaName, 'G and VT')].{Name:QuotaName,Value:Value,Code:QuotaCode}" \
    --output table
  ```
  You need at least 16 vCPUs available (2x g5.2xlarge = 8 vCPUs each) for the plan below. If
  the value is 0 or too low, request an increase from the same console page — approval is often
  near-instant but can take up to a day; do this before everything else.
- The FinScale repo pushed somewhere you can `git clone` from the instances, or be ready to
  `rsync`/`scp` it up (step 4 covers both).

## 1. Variables

```bash
export AWS_REGION=us-east-1
export KEY_NAME=finscale-validation
export INSTANCE_TYPE=g5.2xlarge   # 1x A10G 24GB, 8 vCPU, 32GB RAM per node
```

## 2. Key pair

```bash
aws ec2 create-key-pair --region $AWS_REGION --key-name $KEY_NAME \
  --query 'KeyMaterial' --output text > ~/.ssh/$KEY_NAME.pem
chmod 400 ~/.ssh/$KEY_NAME.pem
```

## 3. Networking

Use the default VPC/subnet for simplicity, a security group allowing SSH from your IP and all
TCP between the two instances (NCCL and the torchrun c10d rendezvous both need unpredictable
ports — simplest to open everything between just these two instances rather than enumerate
ports), and a cluster placement group for lower inter-node latency.

```bash
VPC_ID=$(aws ec2 describe-vpcs --region $AWS_REGION --filters Name=is-default,Values=true \
  --query 'Vpcs[0].VpcId' --output text)
SUBNET_ID=$(aws ec2 describe-subnets --region $AWS_REGION --filters Name=vpc-id,Values=$VPC_ID \
  --query 'Subnets[0].SubnetId' --output text)

SG_ID=$(aws ec2 create-security-group --region $AWS_REGION \
  --group-name finscale-validation-sg --description "FinScale 2-node validation" \
  --vpc-id $VPC_ID --query 'GroupId' --output text)

MY_IP=$(curl -s https://checkip.amazonaws.com)
aws ec2 authorize-security-group-ingress --region $AWS_REGION --group-id $SG_ID \
  --protocol tcp --port 22 --cidr ${MY_IP}/32

aws ec2 authorize-security-group-ingress --region $AWS_REGION --group-id $SG_ID \
  --protocol tcp --port 0-65535 --source-group $SG_ID

aws ec2 create-placement-group --region $AWS_REGION \
  --group-name finscale-validation-pg --strategy cluster
```

If the placement group later causes an `InsufficientInstanceCapacity` error on launch, drop
`--placement` from step 5 and retry — the test still works without it, just with slightly
higher inter-node latency.

## 4. Find the Deep Learning AMI (CUDA + PyTorch preinstalled)

```bash
AMI_ID=$(aws ec2 describe-images --region $AWS_REGION --owners amazon \
  --filters "Name=name,Values=Deep Learning OSS Nvidia Driver AMI GPU PyTorch*Ubuntu 22.04*" \
  --query 'sort_by(Images, &CreationDate)[-1].ImageId' --output text)
echo $AMI_ID
```

## 5. Launch both nodes

```bash
for RANK in 0 1; do
  aws ec2 run-instances --region $AWS_REGION \
    --image-id $AMI_ID \
    --instance-type $INSTANCE_TYPE \
    --key-name $KEY_NAME \
    --security-group-ids $SG_ID \
    --subnet-id $SUBNET_ID \
    --placement GroupName=finscale-validation-pg \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":150,"VolumeType":"gp3"}}]' \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=finscale-node-$RANK}]" \
    --count 1
done
```

Wait ~2 minutes, then get both IPs:

```bash
aws ec2 describe-instances --region $AWS_REGION \
  --filters "Name=tag:Name,Values=finscale-node-0,finscale-node-1" "Name=instance-state-name,Values=running" \
  --query 'Reservations[].Instances[].{Name:Tags[?Key==`Name`]|[0].Value,Id:InstanceId,Public:PublicIpAddress,Private:PrivateIpAddress}' \
  --output table
```

Note node 0's **private** IP — that's `MASTER_ADDR` in step 7. Use the **public** IPs for SSH.

## 6. Set up each node (repeat on both)

```bash
ssh -i ~/.ssh/$KEY_NAME.pem ubuntu@<node-public-ip>
```

On the instance:

```bash
nvidia-smi                      # confirm the A10G is visible
cat /etc/motd                   # lists the available conda env name(s) — usually "pytorch"
source /opt/conda/bin/activate pytorch

# Get the repo onto the box — pick ONE:
git clone <your-repo-url> FinscaleNvidiaPlatform
# or, from YOUR machine (not the instance), if the repo isn't pushed anywhere:
#   rsync -avz -e "ssh -i ~/.ssh/finscale-validation.pem" \
#     --exclude='.venv' --exclude='__pycache__' --exclude='out' --exclude='data' \
#     ./ ubuntu@<node-public-ip>:~/FinscaleNvidiaPlatform/

cd FinscaleNvidiaPlatform
export USE_TF=0 USE_FLAX=0
export HF_HOME=/home/ubuntu/.cache/huggingface
python -m pip install -r requirements.txt
python -m pip check
python -c 'import torch; assert torch.cuda.is_available(); print("CUDA:", torch.version.cuda, torch.cuda.get_device_name(0))'
```

## 7. Run the 2-node FSDP training

Open **two SSH sessions**, one per node (keep them side by side). Both run the **exact same
command**, differing only in `NODE_RANK`. Launch node 0 first, then node 1 within a minute or
two — torchrun's rendezvous waits for both ranks but don't leave a long gap.

**Node 0** (use its own private IP as `MASTER_ADDR`):
```bash
cd ~/FinscaleNvidiaPlatform
source /opt/conda/bin/activate pytorch
export USE_TF=0 USE_FLAX=0
export HF_HOME=/home/ubuntu/.cache/huggingface

python scripts/make_synthetic_smoke_data.py --n 240   # deterministic — see note below

export MASTER_ADDR=<node0-PRIVATE-ip>
export NODE_RANK=0
export MODEL=Qwen/Qwen2.5-1.5B-Instruct
export EPOCHS=1
export BATCH_SIZE=1
export GRAD_ACCUM_STEPS=1
export MAX_SEQ_LENGTH=512
export LORA_R=8
export LORA_ALPHA=16
bash scripts/launch_distributed.sh
```

**Node 1** (same `MASTER_ADDR`, i.e. node 0's private IP):
```bash
cd ~/FinscaleNvidiaPlatform
source /opt/conda/bin/activate pytorch
export USE_TF=0 USE_FLAX=0
export HF_HOME=/home/ubuntu/.cache/huggingface

python scripts/make_synthetic_smoke_data.py --n 240   # same seed+n => byte-identical file, no shared storage needed

export MASTER_ADDR=<node0-PRIVATE-ip>
export NODE_RANK=1
export MODEL=Qwen/Qwen2.5-1.5B-Instruct
export EPOCHS=1
export BATCH_SIZE=1
export GRAD_ACCUM_STEPS=1
export MAX_SEQ_LENGTH=512
export LORA_R=8
export LORA_ALPHA=16
bash scripts/launch_distributed.sh
```

**What success looks like:** both logs (`out/train_peft_rank0.log`, `out/train_peft_rank1.log`)
show the FSDP wrap message and training steps progressing; only node 0 ends up with
`out/adapters/adapter_config.json` and `out/metrics.json` (rank-zero-only save). Check:
```bash
python -c "import json; m=json.load(open('out/metrics.json')); print(m['distributed'], m['world_size'])"
# should print: True 2
```

**If torchrun hangs at rendezvous:** almost always the security group — confirm the self-
referencing all-TCP rule from step 3 is in place, and that both nodes used the exact same
`MASTER_ADDR`/`MASTER_PORT`. Add `export NCCL_DEBUG=INFO` before `bash scripts/launch_distributed.sh`
on both nodes for verbose cross-node connection logs.

## 8. Capture GPU utilization during the run

In a **third and fourth** SSH session (one per node), started just before step 7's training
commands:

```bash
# node 0
python scripts/gpu_monitor.py --out out/gpu_node0.csv --node-id 0
# node 1
python scripts/gpu_monitor.py --out out/gpu_node1.csv --node-id 1
```
Ctrl+C each once training finishes.

## 9. Run a 1-node baseline for comparison (on node 0 only)

```bash
mkdir -p out/1node
python scripts/gpu_monitor.py --out out/1node/gpu_monitor.csv --node-id 0 &
MONITOR_PID=$!
python -u train.py --backend peft \
  --input data/train_raw.jsonl --data-dir data/nvidia_1node \
  --model Qwen/Qwen2.5-1.5B-Instruct --epochs 1 --batch-size 1 --grad-accum-steps 1 \
  --max-seq-length 512 --lora-r 8 --lora-alpha 16 --eval-steps 25 --out-dir out/1node
kill $MONITOR_PID
```

## 10. Compute scaling efficiency (on node 0)

```bash
python scripts/benchmark_scaling.py \
  --single-node out/1node/metrics.json \
  --multi-node out/metrics.json \
  --single-node-gpu-csv out/1node/gpu_monitor.csv \
  --multi-node-gpu-csv out/gpu_node0.csv out/gpu_node1.csv \
  --world-size 2 --out out/scaling_report.json

python -m pip install -r requirements-optional.txt   # matplotlib, for the chart
python scripts/plot_scaling.py --results out/scaling_report.json
```

**Caveat:** with a 1.5B model and a 240-row synthetic dataset, per-step communication overhead
dominates actual compute, so don't expect — or report — a flattering scaling-efficiency number
here. This step only proves `benchmark_scaling.py`/`plot_scaling.py` correctly parse real
`metrics.json`/GPU-CSV files from a real distributed run; a believable efficiency number needs
the full-scale rehearsal (real model size, real data, likely p4d/p5 instances) called out at the
top of this doc.

Pull the results back to your machine:
```bash
scp -i ~/.ssh/finscale-validation.pem ubuntu@<node0-public-ip>:~/FinscaleNvidiaPlatform/out/scaling_report.json .
scp -i ~/.ssh/finscale-validation.pem ubuntu@<node0-public-ip>:~/FinscaleNvidiaPlatform/out/charts/finscale_scaling.png .
```

## 11. Tear down (do this — don't let it run overnight)

```bash
IDS=$(aws ec2 describe-instances --region $AWS_REGION \
  --filters "Name=tag:Name,Values=finscale-node-0,finscale-node-1" "Name=instance-state-name,Values=running,pending,stopped" \
  --query 'Reservations[].Instances[].InstanceId' --output text)
aws ec2 terminate-instances --region $AWS_REGION --instance-ids $IDS
aws ec2 wait instance-terminated --region $AWS_REGION --instance-ids $IDS

aws ec2 delete-security-group --region $AWS_REGION --group-id $SG_ID
aws ec2 delete-placement-group --region $AWS_REGION --group-name finscale-validation-pg
aws ec2 delete-key-pair --region $AWS_REGION --key-name $KEY_NAME   # only if you won't reuse it
```

Double-check in the EC2 console that both instances show `terminated` and no stray EBS volumes
are left over (terminating an instance normally deletes its root volume too, but confirm).

## What this does and doesn't prove

Proves: `train.py --distributed`'s FSDP wiring, `scripts/launch_distributed.sh`'s torchrun
invocation, cross-node NCCL connectivity, rank-zero-only checkpoint saving, and
`scripts/benchmark_scaling.py`/`plot_scaling.py` all work against two real, physically separate
GPUs — this was previously untested code.

Doesn't prove: real throughput, real scaling efficiency, or real classification accuracy at
hackathon scale (14B student, 72B teacher, real SEC data, GB10/Grace-Blackwell hardware). Those
need the full-scale rehearsal path noted at the top, on different instance types, at
meaningfully higher cost.
