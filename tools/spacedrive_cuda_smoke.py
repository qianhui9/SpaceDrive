import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

busy = subprocess.check_output([
    'nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader',
], text=True).strip()
if busy:
    raise SystemExit('GPU has active compute processes; retry after it is idle.')

import torch
import torch.distributed as dist
from mmcv.ops import nms
from mmcv.parallel import MMDistributedDataParallel
from mmcv.parallel.scatter_gather import scatter

assert torch.cuda.is_available(), 'CUDA is unavailable'
torch.cuda.set_device(0)
print('CUDA:', torch.version.cuda, 'GPU:', torch.cuda.get_device_name(0), flush=True)
boxes = torch.tensor([[0., 0., 1., 1.], [0., 0., 1., 1.]], device='cuda')
scores = torch.tensor([0.9, 0.8], device='cuda')
assert nms(boxes, scores, 0.5)[1].tolist() == [0]
assert scatter(torch.ones(2, 4), [0])[0].is_cuda
print('PASS: GPU NMS and MMCV CPU-to-GPU scatter', flush=True)


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(4, 2)

    def forward(self, x):
        return self.layer(x)

    def train_step(self, x):
        return dict(loss=self(x).square().mean(), num_samples=len(x), log_vars={})


with TemporaryDirectory(prefix='spacedrive-nccl-') as temp:
    dist.init_process_group('nccl', init_method=(Path(temp) / 'init').as_uri(), rank=0, world_size=1)
    try:
        model = Toy().cuda()
        wrapped = MMDistributedDataParallel(model, device_ids=[0])
        wrapped(torch.ones(2, 4)).sum().backward()
        wrapped.zero_grad()
        wrapped.train_step(torch.ones(2, 4))['loss'].backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        print('PASS: NCCL single-GPU MMDistributedDataParallel forward/train_step/backward', flush=True)
    finally:
        dist.destroy_process_group()

from spacedrive_smoke_models import run
run('cuda')
torch.cuda.synchronize()
