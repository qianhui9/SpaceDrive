# Local compute ledger

### env: SpaceDrive@f3c7a6f7
how: existing /home/qh/miniconda3/envs/SpaceDrive; warm reuse of previously validated installation
tier: {cpus: 8, mem_gib: 64, gpus: 1, gpu: RTX3090-24GiB}
spec: docs/SpaceDrive-env-spec.json; canonical sorted compact JSON SHA256 prefix f3c7a6f7
weights: /data/qh/phdCode/work4/models/SpaceDrive/SpaceDrive-Qwen2.5-VL-7B/iter_21096.pth (18270580002 bytes)
data: /data/qh/nuscenes/v1.0-trainval (real samples, validation infos)
validated: parent real one-frame inference succeeded; first fresh pass: frame 1 succeeded, frame 2 allocator fragmentation OOM; cache release added; second fresh pass also OOM; depth teacher CPU offload added; third fresh pass SUCCESS (2026-11-09); 3/3 real frames, WITNESS + SUCCESS, mean L2 2.3546884893m, peak21.2519846GiB; fresh agent checks exit/artifacts
constraints: every executable shell command must begin taskset -c 0-3,6-31; CPUs 4/5 faulty; do not stop other users' GPU tasks.
gotcha: checkpoint includes all learned weights. Meta construction + strict assign avoids duplicate 7B pretrained download/allocation. Derived legacy buffers are audited. Six UniDepth views microbatched to fit 24GB.

## Agent-follows-doc invocation

Execute EXACTLY the bash command in docs/INFERENCE_LOCAL.md. Do not modify scripts, docs, dependencies, or arguments to fix failures. Request sandbox escalation if needed to write to the authorized SpaceDrive project; it lies outside the IDE OmniDrive writable root. Check group3.log, exit status, results/metrics, and report every divergence. GPU should be idle before allocation; if occupied, wait without stopping processes.

gotcha: successive UniDepth/VLM stages can leave >5GiB free but fragmented. Release unused CUDA caching allocator blocks after depth and before each frame; original weights/resolution preserved. First failure retained in group3_attempt01_oom.log.

gotcha: empty_cache alone was insufficient (second pass frame 1 OOM). Teacher is now moved CPU between depth/VLM stages, preserving original FP32 tensors; old metrics/pictures cleared at launch to prevent stale success artifacts.

validated outputs: outputs/qwen_inference_smoke_20261109_run01/group3/{results.json,metrics.json,trajectories.png,cameras.png}; mean inference4.84267s/frame; all 1485 current keys strictly matched after audited derived-buffer compatibility. No environment reinstall.

## Full validation run (2026-11-09)

Same environment/spec hash f3c7a6f7, warm reuse. No dependency changes.
Full dataset: 6019 frames / 150 scenes; 5119 complete futures for published default scoring.
Input preflight: all36114 camera images exist. Published default CPU evaluator preflight passed on existing real predictions, all9 metrics plus independent ADE agreement.
Invocation and contract: docs/INFERENCE_FULL_DATASET.md (launch command). Fresh agent must execute that command VERBATIM, then verify worker persists, CPU affinity excludes4/5, START total=6019, data preflight, and at least10 successful frames with no fatal errors. Do not stop the worker or wait8hours; report full-run launch validation, not dataset completion. A busy GPU must only be waited on.
Validation: pending fresh full-run launch check. Existing process detection makes the launch idempotent, resume checks config/data/weights.
