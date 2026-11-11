"""Run every nuScenes validation frame, resume safely, then evaluate planning.

Build architecture on meta, then assign ALL learned tensors from the checkpoint.
Legacy derived buffers are audited separately from learned tensors.
"""
import argparse
import copy
import gc
import fcntl
import hashlib
import traceback
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


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def wait_for_gpu(out):
    while True:
        used = int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).splitlines()[0])
        if used < 500:
            return
        print(f'WAITING_GPU used={used}MiB; leaving existing tasks running', flush=True)
        atomic_json(out / 'status.json', dict(stage='waiting_for_gpu', memory_used_mib=used, pid=os.getpid(), updated_at=time.time()))
        time.sleep(30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--assets', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock_file = (out / '.worker.lock').open('a')
    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    atomic_json(out / 'status.json', dict(stage='preparing', pid=os.getpid(), updated_at=time.time()))
    wait_for_gpu(out)
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
    checkpoint_config_hash = hashlib.sha256(ckpt['meta']['config'].encode()).hexdigest()
    assert not any(p.is_meta for p in model.parameters())
    print('All learned tensors loaded with strict key/shape matching', flush=True)
    del state, filtered, expected, ckpt
    gc.collect()
    if args.audit_only:
        return
    data_root = Path(args.data_root).resolve()
    anno_path = data_root / 'nuscenes2d_ego_temporal_infos_val.pkl'
    anno = pickle.load(anno_path.open('rb'))
    infos = sorted(anno['infos'], key=lambda x: x['timestamp'])
    tokens = [x['token'] for x in infos]
    assert len(set(tokens)) == len(tokens)
    # Every frame is retained, including stationary frames and scene endings.
    # This field is required by the dataset but is disabled in the prompt.
    for info in infos:
        info.setdefault('gt_planning_command_desc', '')
    manifest = [dict(index=i, token=x['token'], scene_token=x['scene_token'], timestamp=x['timestamp'], gt_xy=np.asarray(x['gt_planning'])[0,:,:2].tolist(), gt_mask=np.asarray(x['gt_planning_mask'])[0].all(axis=-1).tolist()) for i,x in enumerate(infos)]
    run_config = dict(checkpoint=str(Path(args.checkpoint).resolve()), checkpoint_bytes=Path(args.checkpoint).stat().st_size, checkpoint_mtime_ns=Path(args.checkpoint).stat().st_mtime_ns, checkpoint_config_sha256=checkpoint_config_hash,
                      annotations=str(anno_path), annotations_bytes=anno_path.stat().st_size, annotations_mtime_ns=anno_path.stat().st_mtime_ns,
                      tokens_sha256=hashlib.sha256('\n'.join(tokens).encode()).hexdigest(), total_frames=len(infos), scenes=len({x['scene_token'] for x in infos}), complete_future_frames=sum(all(x['gt_mask']) for x in manifest),
                      selection='ALL validation infos; no motion or validity filter', seed='888 + chronological frame index', load_ego_command_in_question=False, image_resize=[640,640], depth_microbatch=1, depth_cpu_offload_between_stages=True, quantization=None,
                      generation_config=model.lm_head.generation_config.to_dict())
    config_path = out / 'run_config.json'
    if config_path.exists():
        if not args.resume:
            raise RuntimeError('Output already contains a run; pass --resume or use a new output directory')
        assert json.loads(config_path.read_text()) == run_config, 'Resume checkpoint/data/config mismatch'
    atomic_json(config_path, run_config)
    atomic_json(out / 'samples.json', manifest)
    records_dir = out / 'records'
    records_dir.mkdir(exist_ok=True)
    anno['infos'] = infos
    prepared_anno = out / 'annotations.pkl'
    if not prepared_anno.exists():
        with prepared_anno.with_suffix('.tmp').open('wb') as stream:
            pickle.dump(anno, stream)
        prepared_anno.with_suffix('.tmp').replace(prepared_anno)
    del anno, infos
    gc.collect()
    link = ROOT / 'data/nuscenes'
    link.parent.mkdir(exist_ok=True)
    if not link.exists():
        link.symlink_to(data_root, target_is_directory=True)
    assert link.resolve() == data_root
    pipeline = copy.deepcopy(cfg.test_pipeline)
    for step in pipeline:
        if step['type'] == 'LoadAnnoatationVQATest':
            step.update(processor=str(qwen), tokenizer=str(qwen))
            assert step['load_ego_command_in_question'] is False
            step['num_commands'] = 3  # skips an unused GT dictionary; same planning prompt
    assert model.ego_status == '', 'Resume without historical ego features is only valid for this checkpoint'
    dcfg = copy.deepcopy(cfg.data.test)
    dcfg.update(data_root=str(data_root), ann_file=str(prepared_anno), pipeline=pipeline, lane_path=str(data_root / 'data_dict_sample.pkl'))
    dataset = build_dataset(dcfg)
    assert [x['token'] for x in dataset.data_infos] == tokens, 'Dataset/manifest ordering differs'
    missing_images = []
    for info in dataset.data_infos:
        for name, camera in info['cams'].items():
            if not Path(camera['data_path']).is_file():
                missing_images.append(dict(token=info['token'], camera=name, path=camera['data_path']))
    atomic_json(out / 'data_preflight.json', dict(frames=len(dataset), camera_images=len(dataset)*6, missing_images=missing_images))
    assert not missing_images, f'Missing input images: {missing_images[:5]}'
    print('FULL_DATASET ' + json.dumps(run_config), flush=True)
    results = {}
    if args.resume:
        for sample in manifest:
            record_path = records_dir / (sample['token'] + '.json')
            if record_path.exists():
                record = json.loads(record_path.read_text())
                if record.get('status') == 'success':
                    assert record['token'] == sample['token']
                    assert np.asarray(record['predicted_xy']).shape == (6,2)
                    assert np.isfinite(record['predicted_xy']).all()
                    assert (out / 'raw' / sample['token']).is_file()
                    results[sample['token']] = record
    failures = {}
    attempt_times = []
    start_time = time.time()
    def save_progress(stage, current_token=None):
        elapsed = time.time()-start_time
        remaining = len(manifest)-len(results)-len(failures)
        average = float(np.mean(attempt_times)) if attempt_times else None
        status = dict(stage=stage, pid=os.getpid(), total=len(manifest), successful=len(results), failed=len(failures), remaining=remaining,
                      current_token=current_token, elapsed_this_process_seconds=elapsed, mean_attempt_seconds=average,
                      eta_seconds=remaining*average if average is not None else None, updated_at=time.time())
        atomic_json(out / 'status.json', status)
        complete = [r for r in results.values() if all(r['gt_mask'])]
        metrics = dict(inference_complete=len(results)==len(manifest), total_frames=len(manifest), successful_frames=len(results), failed_frames=len(failures),
                       expected_complete_future_frames=run_config['complete_future_frames'], scored_complete_future_frames=len(complete),
                       excluded_from_scoring_reason='Frames without all six future ground-truth positions are inferred but excluded, matching project evaluation.',
                       horizons_s=[0.5,1,1.5,2,2.5,3], peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                       source_coordinates='Published model output rounded to two decimals', metrics_status='partial' if len(results)!=len(manifest) else 'L2 complete; geometry evaluation pending')
        if complete:
            point_l2 = np.asarray([r['l2_m'] for r in complete])
            metrics.update(mean_l2_by_horizon_m=point_l2.mean(axis=0).tolist(),
                           mean_l2_m=float(point_l2.mean()),
                           project_ade_m={f'{s}s':float(point_l2[:,:2*s].mean()) for s in [1,2,3]},
                           endpoint_l2_m={f'{s}s':float(point_l2[:,2*s-1].mean()) for s in [1,2,3]})
        atomic_json(out / 'metrics.json', metrics)
        return status
    wait_for_gpu(out)
    torch.cuda.manual_seed_all(888)
    witness = torch.randn(16,16,device='cuda')
    assert torch.isfinite(witness @ witness.T).all()
    print('WITNESS', list(witness.shape), torch.cuda.get_device_name(), flush=True)
    model = model.eval().cuda()
    original_infer = model.depth_model.infer
    def infer_per_view(rgb, *unused_args, **unused_kwargs):
        model.depth_model.cuda()
        rgb = rgb.cuda()
        depths = [original_infer(rgb[n:n+1], *unused_args, **unused_kwargs)['depth'] for n in range(rgb.shape[0])]
        result = {'depth':torch.cat(depths,dim=0)}
        model.depth_model.cpu()
        torch.cuda.empty_cache()
        return result
    model.depth_model.infer = infer_per_view
    save_progress('inference')
    print(f'START total={len(manifest)} already_complete={len(results)}', flush=True)
    for i, sample in enumerate(manifest):
        token = sample['token']
        if token in results:
            continue
        gc.collect()
        torch.cuda.empty_cache()
        seed = 888+i  # repeatable at the same frame index after resuming
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        t0 = time.time()
        save_progress('inference', token)
        print(f'INFERENCE {i+1}/{len(dataset)} {token}', flush=True)
        try:
            batch = scatter(collate([dataset[i]], samples_per_gpu=1), [0])[0]
            with torch.no_grad():
                response = model(return_loss=False, rescale=True, **batch)
            torch.cuda.synchronize()
            text = response[0]['A']
            if isinstance(text,list):
                text = text[0]
            coords = re.findall(r'\((-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\)',text)
            if len(coords)!=6:
                raise ValueError(f'Expected six generated trajectory points; got {len(coords)}: {text}')
            prediction = np.asarray(coords,dtype=float)
            assert np.isfinite(prediction).all()
            gt = np.asarray(sample['gt_xy'])
            errors = np.linalg.norm(prediction-gt,axis=-1)
            mask = np.asarray(sample['gt_mask'],dtype=bool)
            record = dict(status='success', index=i, token=token, scene_token=sample['scene_token'], seed=seed,
                          answer=text, predicted_xy=prediction.tolist(), gt_xy=gt.tolist(), gt_mask=mask.tolist(),
                          l2_m=[float(e) if valid else None for e,valid in zip(errors,mask)],
                          mean_l2_m=float(errors[mask].mean()) if mask.any() else None, seconds=time.time()-t0)
            atomic_json(records_dir / (token+'.json'),record)
            results[token]=record
            print('FRAME_SUCCESS ' + json.dumps(dict(index=i,token=token,seconds=record['seconds'],full_future=bool(mask.all()))),flush=True)
        except Exception as exc:
            if not isinstance(exc,(torch.cuda.OutOfMemoryError,ValueError,FileNotFoundError)):
                raise  # unknown failures require diagnosis rather than silent omissions
            record=dict(status='failed',index=i,token=token,error=repr(exc),traceback=traceback.format_exc(),seconds=time.time()-t0)
            atomic_json(records_dir / (token+'.json'),record)
            failures[token]=record
            print('FRAME_FAILED ' + json.dumps(record),flush=True)
            model.depth_model.cpu()
        finally:
            if 'batch' in locals():
                del batch
            gc.collect()
            torch.cuda.empty_cache()
        attempt_times.append(time.time()-t0)
        status=save_progress('inference')
        print('PROGRESS ' + json.dumps(status),flush=True)
    atomic_json(out / 'results.json', [results[s['token']] for s in manifest if s['token'] in results])
    atomic_json(out / 'failures.json', list(failures.values()))
    save_progress('inference_complete' if not failures else 'inference_finished_with_failures')
    del model, original_infer, witness, dataset
    gc.collect()
    torch.cuda.empty_cache()
    if failures:
        raise RuntimeError(f'{len(failures)} frame(s) failed; all other frames attempted. Resume retries these frames; no complete benchmark reported.')
    print('INFERENCE_COMPLETE successful=' + str(len(results)),flush=True)
    # Run the project's default geometry calculations serially to avoid its
    # shared-counter race. Check failures, denominator, and L2 independently.
    save_progress('planning_evaluation')
    eval_log = out / 'evaluation_default.log'
    command=[sys.executable,'-u',str(ROOT/'scripts/evaluation/eval_planning.py'),'--pred_path',str(out/'raw'),'--base_path',str(data_root),'--num_threads','1']
    atomic_json(out / 'evaluation_command.json', command)
    with eval_log.open('w') as stream:
        subprocess.run(command,cwd=out,stdout=stream,stderr=subprocess.STDOUT,check=True)
    log=eval_log.read_text()
    assert 'EVALUATION_ERROR' not in log and 'Traceback (most recent call last)' not in log, 'Geometry evaluation reported errors; inspect evaluation_default.log'
    counts=[int(x) for x in re.findall(r'number of samples:\s*(\d+)',log)]
    assert counts and all(x==run_config['complete_future_frames'] for x in counts), f'Evaluation denominator mismatch: {counts}'
    values={k:float(v) for k,v in re.findall(r'^(plan_obj_box_col_[123]s|plan_boundary_[123]s|l2_[123]s):\s*([-+\deE.]+)',log,re.MULTILINE)}
    assert len(values)==9, f'Missing metrics: {values}'
    metrics=json.loads((out/'metrics.json').read_text())
    for second in [1,2,3]:
        assert abs(values[f'l2_{second}s']-metrics['project_ade_m'][f'{second}s'])<1e-5, 'Evaluator L2 mismatch'
    metrics.update(metrics_status='complete', project_default_evaluation=values, evaluated_frames=counts[0],inference_complete=True)
    atomic_json(out/'metrics.json',metrics)
    status=json.loads((out/'status.json').read_text())
    status.update(stage='completed',updated_at=time.time(),evaluation_complete=True)
    atomic_json(out/'status.json',status)
    print('SUCCESS '+json.dumps(metrics),flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        if '--output-dir' in sys.argv:
            output = Path(sys.argv[sys.argv.index('--output-dir')+1]).resolve()
            status_path=output/'status.json'
            if output.exists():
                try:
                    status=json.loads(status_path.read_text()) if status_path.exists() else {}
                    if not status or status.get('pid') == os.getpid():
                        status.update(stage='failed',error=repr(exc),pid=os.getpid(),updated_at=time.time())
                        atomic_json(status_path,status)
                except Exception:
                    pass
        raise
