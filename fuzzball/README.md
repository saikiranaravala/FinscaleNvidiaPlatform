# Fuzzball workflow templates

These are **best-effort templates**, not validated against a live Fuzzball cluster or a specific
CLI version — this repo was prepared without access to a Fuzzball install. Before the event:

1. Run `fuzzball version` and `fuzzball job validate <file>.yaml` against each file here and fix
   any schema drift against your installed CLI.
2. Replace `<REGISTRY>/finscale-aarch64:latest` with the real path to the image built from
   `containers/finscale-aarch64.def` (push it to whatever registry the cluster can pull from).
3. Replace the `volumes` mount paths with your actual Fuzzball volume/filesystem names.
4. Confirm the GPU resource request syntax (`resources.gpus`) matches your Fuzzball version —
   some releases key GPU requests by vendor/model string instead of a bare count.

Files:
- `label_workflow.yaml` — single-node job: label excerpts with the 72B teacher via vLLM.
- `train_workflow.yaml` — two-node job: FSDP LoRA fine-tune via `scripts/launch_distributed.sh`.
- `eval_workflow.yaml` — single-node job: base-vs-fine-tuned evaluation via `evaluate.py`.

Having these on a slide — even unvalidated — demonstrates you planned for Fuzzball-native
orchestration rather than ad hoc SSH + tmux, which is part of what's being judged.
