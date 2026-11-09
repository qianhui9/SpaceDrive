"""Check installed dependencies and startup paths without downloading model weights."""
import argparse
import contextlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback

ROOT = Path(__file__).resolve().parents[1]


def imports_and_configs():
    import mmcv
    import mmdet3d
    import projects.mmdet3d_plugin
    from mmdet.models import DETECTORS
    from mmdet.datasets import DATASETS, PIPELINES
    from mmdet3d.datasets.builder import PIPELINES as PIPELINES_3D
    from unidepth.models import UniDepthV2
    from unidepth.utils.camera import Pinhole

    assert DETECTORS.get('SpaceDrive') is not None
    assert DATASETS.get('CustomNuScenesDataset') is not None
    for path in sorted((ROOT / 'projects/configs/spacedrive').glob('*.py')):
        cfg = mmcv.Config.fromfile(str(path))
        assert DETECTORS.get(cfg.model.type) is not None
        for pipeline in (cfg.train_pipeline, cfg.test_pipeline):
            for stage in pipeline:
                assert any(reg.get(stage['type']) is not None for reg in (PIPELINES, PIPELINES_3D)), stage['type']
        print('Loaded config and registries:', path.name)
    assert mmdet3d.__version__ == '1.0.0rc6'
    parallel = Path(mmcv.__file__).parent / 'parallel'
    for name in ('_functions.py', 'distributed.py'):
        assert (parallel / name).read_bytes() == (ROOT / 'docker/torch21_mmcv' / name).read_bytes()
    assert Path(mmdet3d.__file__).read_bytes() == (ROOT / 'docker/torch21_mmcv/__init__.py').read_bytes()
    print('Project imports, registries and all three compatibility patches: OK')


def cpu_and_video():
    import cv2
    import decord
    import numba
    import numpy as np
    import open3d as o3d
    import torch
    from mmcv.ops import nms
    assert torch.from_numpy(np.ones(1, dtype=np.float32)).item() == 1.
    keep = nms(torch.tensor([[0., 0., 1., 1.], [0., 0., 1., 1.]]), torch.tensor([0.9, 0.8]), 0.5)[1]
    assert keep.tolist() == [0]
    with tempfile.TemporaryDirectory(prefix='spacedrive-video-') as temp:
        path = Path(temp) / 'smoke.mp4'
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 5, (32, 32))
        assert writer.isOpened()
        for i in range(4):
            writer.write(np.full((32, 32, 3), i * 60, dtype=np.uint8))
        writer.release()
        decord.bridge.set_bridge('native')
        video = decord.VideoReader(str(path), ctx=decord.cpu(0), num_threads=1)
        assert len(video) == 4
        batch = video.get_batch([0, 3]).asnumpy()
        assert batch.shape == (2, 32, 32, 3)
        assert batch[1].mean() > batch[0].mean() + 100
        try:
            decord.bridge.set_bridge('torch')
            assert isinstance(video.get_batch([0]), torch.Tensor)
        finally:
            decord.bridge.set_bridge('native')
    points = o3d.geometry.PointCloud()
    points.points = o3d.utility.Vector3dVector(np.zeros((2, 3)))
    assert len(points.points) == 2
    assert numba.njit(lambda x: x.sum())(np.arange(4, dtype=np.float32)) == 6
    print('CPU NMS, NumPy bridge, MP4 decoding, Open3D and Numba JIT: OK')


def cli_help():
    for mode in ('train', 'test'):
        result = subprocess.run([sys.executable, str(ROOT / f'scripts/{mode}/{mode}.py'), '--help'],
                                cwd=ROOT, capture_output=True, text=True, timeout=120)
        print(result.stdout)
        print(result.stderr)
        assert result.returncode == 0, f'{mode} --help failed'
        assert 'usage:' in result.stdout


def runtime_assets(config):
    from mmcv import Config
    from huggingface_hub import try_to_load_from_cache
    cfg = Config.fromfile(str(ROOT / config))
    paths = {'nuScenes data': cfg.data_root, 'VLM weights/tokenizer directory': cfg.model.lm_head}
    for split in ('train', 'val', 'test'):
        paths[f'{split} annotation'] = cfg.data[split].ann_file
    result = {}
    for name, path in paths.items():
        full = Path(path) if Path(path).is_absolute() else ROOT / path
        result[name] = dict(path=str(full), exists=full.exists())
    depth_repo = 'lpiccinelli/unidepth-v2-vitl14'
    cached = try_to_load_from_cache(depth_repo, 'model.safetensors')
    result['UniDepthV2 weights in Hugging Face cache'] = dict(
        path=cached if isinstance(cached, str) else depth_repo,
        exists=isinstance(cached, str) and Path(cached).is_file())
    result['trained checkpoint for evaluation'] = dict(
        path=str(ROOT / cfg.work_dir / 'iter_21096.pth'),
        exists=(ROOT / cfg.work_dir / 'iter_21096.pth').is_file())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cuda', action='store_true', help='also test an idle GPU, NCCL and tiny multimodal models')
    parser.add_argument('--require-assets', action='store_true', help='also require configured dataset and model files')
    parser.add_argument('--config', default='projects/configs/spacedrive/spacedrive_plus_qwen.py')
    args = parser.parse_args()
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    os.environ['PYTHONPATH'] = str(ROOT) + os.pathsep + os.environ.get('PYTHONPATH', '')
    out = ROOT / 'outputs/environment_setup'
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(out / 'matplotlib'))
    assert not ({4, 5} & os.sched_getaffinity(0)), 'Launch with taskset -c 0-3,6-31'
    report = dict(python=sys.executable, cpu_affinity=sorted(os.sched_getaffinity(0)), checks={})

    def check(name, action):
        log = out / f'{name}.log'
        try:
            with log.open('w') as stream, contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                action()
        except Exception:
            with log.open('a') as stream:
                traceback.print_exc(file=stream)
            report['checks'][name] = False
            print(f'FAIL: {name}; see {log}', flush=True)
        else:
            report['checks'][name] = True
            print(f'PASS: {name}', flush=True)

    def pip_check():
        result = subprocess.run([sys.executable, '-m', 'pip', '--no-cache-dir', 'check'],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print(result.stdout)
        result.check_returncode()

    def tiny_cpu():
        from spacedrive_smoke_models import run
        run('cpu')

    def cuda():
        result = subprocess.run([sys.executable, str(ROOT / 'tools/spacedrive_cuda_smoke.py')],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=180)
        print(result.stdout)
        result.check_returncode()

    check('pip_check', pip_check)
    check('imports_configs_patches', imports_and_configs)
    check('cpu_video', cpu_and_video)
    check('cli_help', cli_help)
    check('tiny_models_cpu', tiny_cpu)
    if args.cuda:
        check('cuda_nccl_models', cuda)
    names = ('torch', 'torchvision', 'transformers', 'peft', 'accelerate', 'mmcv-full', 'mmdet',
             'mmdet3d', 'mmsegmentation', 'unidepth', 'numpy', 'numba', 'networkx', 'opencv-python',
             'decord', 'open3d', 'einops', 'timm', 'wandb')
    report['versions'] = {name: metadata.version(name) for name in names}
    report['assets'] = runtime_assets(args.config)
    report['environment_ready'] = all(report['checks'].values())
    report['runtime_assets_ready'] = all(asset['exists'] for asset in report['assets'].values())
    (out / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    frozen = subprocess.check_output([sys.executable, '-m', 'pip', '--no-cache-dir', 'freeze', '--all'], text=True)
    (out / 'requirements-installed.txt').write_text(frozen)
    print('Environment ready:', report['environment_ready'])
    print('Runtime assets ready:', report['runtime_assets_ready'])
    for name, asset in report['assets'].items():
        if not asset['exists']:
            print(f'MISSING: {name}: {asset["path"]}')
    print('Report:', out / 'validation.json')
    return 0 if report['environment_ready'] and (not args.require_assets or report['runtime_assets_ready']) else 1


if __name__ == '__main__':
    sys.exit(main())
