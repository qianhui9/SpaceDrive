"""Run real, small nuScenes planning inference from a complete SpaceDrive checkpoint.

Build architecture on meta, then assign ALL learned tensors from the checkpoint.
Legacy derived buffers are audited separately from learned tensors.
"""
import argparse
import copy
import gc
import inspect
import json
import os
from pathlib import Path
import pickle
import random
import re
import subprocess
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--assets', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--num-samples', type=int, default=3)
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    if args.num_samples < 1:
        parser.error('--num-samples must be positive')
    os.chdir(ROOT)
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if not args.audit_only:
        (out / 'results.json').write_text('[]\n')
        for filename in ['metrics.json', 'trajectories.png', 'cameras.png']:
            (out / filename).unlink(missing_ok=True)
    import numpy as np
    import torch
    from accelerate import init_empty_weights
    from transformers import AutoTokenizer, AutoProcessor, Qwen2_5_VLConfig, GenerationConfig
    from mmcv import Config
    from mmcv.parallel import collate, scatter
    import projects.mmdet3d_plugin  # registry imports
    from projects.mmdet3d_plugin.models.vlm.spacedrive import SpaceDrive
    from projects.mmdet3d_plugin.models.vlm_utils.custom_qwen import CustomQwen2_5_VLForConditionalGeneration
    from unidepth.models import UniDepthV2
    from mmdet3d.datasets import build_dataset
    from tools.command_generation import command_generation
    random.seed(888)
    np.random.seed(888)
    torch.manual_seed(888)
    torch.set_num_threads(8)
    torch.autograd.set_detect_anomaly(False)
    torch.backends.cudnn.benchmark = False
    assets = Path(args.assets).resolve()
    qwen = assets / 'Qwen2.5-VL-7B-Instruct-with-new-special-tokens'
    tokenizer = AutoTokenizer.from_pretrained(qwen)
    existing = tokenizer.additional_special_tokens
    tokenizer.add_special_tokens({'additional_special_tokens': list(dict.fromkeys(existing + ['<POS_INDICATOR>', '<POS_EMBEDDING>']))})
    assert [tokenizer.convert_tokens_to_ids(t) for t in ['<POS_INDICATOR>', '<POS_EMBEDDING>']] == [151665, 151666]
    processor = AutoProcessor.from_pretrained(qwen)
    processor.tokenizer = tokenizer
    processor.save_pretrained(qwen)
    tokenizer.save_pretrained(qwen)
    qconfig = Qwen2_5_VLConfig.from_pretrained(qwen)
    qconfig.vocab_size = 151667
    qconfig.text_config.vocab_size = 151667
    qconfig.hidden_size = qconfig.text_config.hidden_size
    qconfig._attn_implementation = 'sdpa'
    qconfig.text_config._attn_implementation = 'sdpa'
    qconfig.vision_config._attn_implementation = 'sdpa'
    qconfig.save_pretrained(qwen)
    depth_config = json.loads((assets / 'unidepth-v2-vitl14/config.json').read_text())
    depth_config['model']['pixel_encoder']['pretrained'] = None
    print('Loading checkpoint metadata and mmap tensors', flush=True)
    ckpt = torch.load(args.checkpoint, map_location='cpu', mmap=True)
    (out / 'checkpoint_config.py').write_text(ckpt['meta']['config'])
    config_ns = {'device': torch.device}
    exec(compile(ckpt['meta']['config'], 'checkpoint_config', 'exec'), config_ns)
    cfg = Config({k: v for k, v in config_ns.items() if not k.startswith('__') and k != 'device'})
    kwargs = dict(cfg.model)
    kwargs['pe_decode_method'] = kwargs.pop('loss_pos_method')
    signature = inspect.signature(SpaceDrive.__init__).parameters
    ignored_config = {k: v for k, v in kwargs.items() if k not in signature}
    kwargs = {k: v for k, v in kwargs.items() if k in signature}
    kwargs.update(lm_head=str(qwen), tokenizer=str(qwen), processor=str(qwen), train_cfg=None, save_path=str(out / 'raw') + '/')
    def make_vlm(cls, *unused, **unused_kwargs):
        vlm = cls(copy.deepcopy(qconfig))
        vlm.generation_config = GenerationConfig.from_pretrained(qwen)
        return vlm
    def make_depth(cls, *unused, **unused_kwargs):
        return cls(config=copy.deepcopy(depth_config))
    print('Constructing architecture without duplicate pretrained downloads', flush=True)
    with patch.object(CustomQwen2_5_VLForConditionalGeneration, 'from_pretrained', classmethod(make_vlm)), patch.object(UniDepthV2, 'from_pretrained', classmethod(make_depth)), init_empty_weights():
        model = SpaceDrive(**kwargs)
    state = ckpt['state_dict']
    expected = model.state_dict()
    buffers = dict(model.named_buffers())
    legacy = []
    buffer_checks = {}
    for k, v in state.items():
        if k in expected:
            continue
        if k == 'position_range' or k.endswith('.cached_penc'):
            legacy.append(k)  # unused range / recomputed per-input PE cache
        elif k.endswith('.inv_freq'):
            current = buffers.get(k)
            if current is None and '.layers.' in k:
                current = buffers.get('lm_head.base_model.model.model.language_model.rotary_emb.inv_freq')
            assert current is not None and torch.equal(v, current), f'Rotary frequencies differ: {k}'
            buffer_checks[k] = 'exactly equal to recomputed nonpersistent buffer'
            legacy.append(k)
    filtered = {k: v for k, v in state.items() if k not in legacy}
    rope_key = 'lm_head.base_model.model.model.rope_3d.inv_freq'
    if rope_key not in filtered:
        assert not model.use_rope
        filtered[rope_key] = expected[rope_key]  # deterministic, unused when use_rope=False
    audit = dict(checkpoint=args.checkpoint, checkpoint_iter=ckpt['meta'].get('iter'), expected_keys=len(expected), checkpoint_keys=len(state), ignored_legacy_buffers=legacy, ignored_legacy_config=ignored_config,
                 missing=sorted(set(expected)-set(filtered)), unexpected=sorted(set(filtered)-set(expected)),
                 shape_mismatches={k: [list(expected[k].shape), list(v.shape)] for k, v in filtered.items() if k in expected and expected[k].shape != v.shape})
    audit['recomputed_buffer_checks'] = buffer_checks
    audit['unused_new_buffer'] = rope_key
    (out / 'checkpoint_audit.json').write_text(json.dumps(audit, indent=2, default=str))
    assert not audit['missing'] and not audit['unexpected'] and not audit['shape_mismatches'], audit
    model.load_state_dict(filtered, strict=True, assign=True)
    assert not any(p.is_meta for p in model.parameters())
    print('All learned tensors loaded with strict key/shape matching', flush=True)
    del state, filtered, expected, ckpt
    gc.collect()
    if args.audit_only:
        return
    data_root = Path(args.data_root).resolve()
    anno = pickle.load(open(data_root / 'nuscenes2d_ego_temporal_infos_val.pkl', 'rb'))
    infos = sorted(anno['infos'], key=lambda x: x['timestamp'])
    selected = None
    for start in range(len(infos)-args.num_samples+1):
        group = infos[start:start+args.num_samples]
        if len({x['scene_token'] for x in group}) != 1:
            continue
        if all(np.asarray(x['gt_planning_mask']).all() and np.linalg.norm(np.asarray(x['gt_planning'])[0,-1,:2]) > 5 for x in group):
            selected = copy.deepcopy(group)
            break
    assert selected, 'No contiguous moving frames with complete 3s GT'
    subset = out / 'subset.pkl'
    pickle.dump(dict(infos=selected, metadata=anno['metadata']), open(subset, 'wb'))
    command_generation(str(subset), str(out / 'subset_with_command_desc.pkl'))
    del anno, infos
    gc.collect()
    # The original info paths and pipeline resolve images relative to data/nuscenes.
    link = ROOT / 'data/nuscenes'
    link.parent.mkdir(exist_ok=True)
    if not link.exists():
        link.symlink_to(data_root, target_is_directory=True)
    assert link.resolve() == data_root, 'Existing dataset link points elsewhere'
    pipeline = copy.deepcopy(cfg.test_pipeline)
    for step in pipeline:
        if step['type'] == 'LoadAnnoatationVQATest':
            step.update(processor=str(qwen), tokenizer=str(qwen))
            assert step['load_ego_command_in_question'] is False
            # This argument only triggers an unused GT dictionary load in test __init__.
            # Planning-only preprocessing never reads key_infos or num_commands.
            step['num_commands'] = 3
    dcfg = copy.deepcopy(cfg.data.test)
    dcfg.update(data_root=str(data_root), ann_file=str(out / 'subset_with_command_desc.pkl'), pipeline=pipeline, lane_path=str(data_root / 'data_dict_sample.pkl'))
    dataset = build_dataset(dcfg)
    manifest = [{'token': x['token'], 'scene_token': x['scene_token'], 'timestamp': x['timestamp'], 'gt_xy': np.asarray(x['gt_planning'])[0,:,:2].tolist(), 'camera_files': {k: v['data_path'] for k,v in x['cams'].items()}} for x in selected]
    (out / 'samples.json').write_text(json.dumps(manifest, indent=2))
    usage = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
    assert int(usage.strip().splitlines()[0]) < 500, 'GPU occupied; wait for other tasks, never stop them'
    torch.cuda.manual_seed_all(888)
    witness = torch.randn(16, 16, device='cuda')
    assert torch.isfinite(witness @ witness.T).all()
    print('WITNESS', list((witness @ witness.T).shape), torch.cuda.get_device_name(), flush=True)
    print('Moving original checkpoint dtypes to idle GPU', flush=True)
    model = model.eval().cuda()
    original_infer = model.depth_model.infer
    def infer_per_view(rgb, *args, **kwargs):
        # The depth teacher and VLM run sequentially. Keep the teacher in CPU
        # memory during VLM attention, preserving its exact FP32 weights.
        model.depth_model.cuda()
        rgb = rgb.cuda()
        depths = []
        for view in range(rgb.shape[0]):
            print(f'Depth camera {view+1}/{rgb.shape[0]}', flush=True)
            depths.append(original_infer(rgb[view:view+1], *args, **kwargs)['depth'])
        result = {'depth': torch.cat(depths, dim=0)}
        model.depth_model.cpu()
        # Vision SDPA needs a large contiguous allocation. Depth temporaries from
        # successive frames otherwise fragment the 24GB allocator's free blocks.
        torch.cuda.empty_cache()
        return result
    model.depth_model.infer = infer_per_view
    results = []
    for i in range(len(dataset)):
        gc.collect()
        torch.cuda.empty_cache()
        print(f'INFERENCE {i+1}/{len(dataset)} {manifest[i]["token"]}', flush=True)
        batch = scatter(collate([dataset[i]], samples_per_gpu=1), [0])[0]
        print('Input shapes', {k: str(getattr(v, 'shape', type(v))) for k,v in batch.items()}, flush=True)
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            response = model(return_loss=False, rescale=True, **batch)
        torch.cuda.synchronize()
        text = response[0]['A']
        if isinstance(text, list):
            text = text[0]
        coords = re.findall(r'\((-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\)', text)
        assert len(coords) == 6, f'Expected 6 trajectory points, got {coords}: {text}'
        pred = np.asarray(coords, dtype=float)
        gt = np.asarray(manifest[i]['gt_xy'])
        errors = np.linalg.norm(pred-gt, axis=-1)
        result = dict(token=manifest[i]['token'], answer=text, predicted_xy=pred.tolist(), gt_xy=gt.tolist(), l2_m=errors.tolist(), mean_l2_m=float(errors.mean()), seconds=time.time()-t0)
        results.append(result)
        (out / 'results.json').write_text(json.dumps(results, indent=2))
        print(json.dumps(result), flush=True)
    metrics = dict(samples=len(results), horizons_s=[0.5,1,1.5,2,2.5,3], mean_l2_by_horizon_m=np.mean([x['l2_m'] for x in results], axis=0).tolist(), mean_l2_m=float(np.mean([x['mean_l2_m'] for x in results])), peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30, seed=888, quantization=None, depth_microbatch=1, depth_cpu_offload_between_stages=True, image_resize=[640,640], load_ego_command_in_question=False, generation_config=model.lm_head.generation_config.to_dict(), note='Small smoke test only; not a benchmark. Coordinates parsed from original two-decimal model output.')
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(results), figsize=(5*len(results), 5), squeeze=False)
    for ax, r in zip(axes[0], results):
        p, g = np.asarray(r['predicted_xy']), np.asarray(r['gt_xy'])
        ax.plot(np.r_[0,g[:,1]], np.r_[0,g[:,0]], 'o-', label='Ground truth')
        ax.plot(np.r_[0,p[:,1]], np.r_[0,p[:,0]], 's--', label='SpaceDrive')
        ax.set(xlabel='Lateral (m)', ylabel='Forward (m)', title=f'{r["token"][:8]} | mean L2 {r["mean_l2_m"]:.2f} m')
        ax.axis('equal')
        ax.grid(True, alpha=.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / 'trajectories.png', dpi=160)
    from PIL import Image
    fig, axes = plt.subplots(2, 3, figsize=(15, 6))
    for ax, (name, filename) in zip(axes.flat, manifest[0]['camera_files'].items()):
        with Image.open(filename) as image:
            ax.imshow(image)
        ax.set_title(name)
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(out / 'cameras.png', dpi=120)
    print('SUCCESS ' + json.dumps(metrics), flush=True)


if __name__ == '__main__':
    main()
