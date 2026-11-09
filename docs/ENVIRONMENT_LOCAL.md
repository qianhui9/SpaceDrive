# 本机 SpaceDrive 环境

安装日期：2026-11-09。项目：`/data/qh/phdCode/work4/SpaceDrive`。
解释器：`/home/qh/miniconda3/envs/SpaceDrive/bin/python`，Conda 环境名大小写为 **SpaceDrive**。

## 启动与自检

```bash
cd /data/qh/phdCode/work4/SpaceDrive
conda activate SpaceDrive

# CPU 环境检查：依赖、插件/四份配置、入口、视频解码、微型 Qwen/LLaVA 前向反向
taskset -c 0-3,6-31 bash tools/run_spacedrive.sh check

# GPU 空闲时：额外验证 CUDA NMS、scatter、单卡 NCCL 和微型模型前向反向
taskset -c 0-3,6-31 bash tools/run_spacedrive.sh check --cuda

# 分布式启动入口的参数说明
taskset -c 0-3,6-31 bash tools/run_spacedrive.sh train --help
taskset -c 0-3,6-31 bash tools/run_spacedrive.sh test --help
```

`run_spacedrive.sh` 固定使用上述解释器、设置项目 PYTHONPATH，并避开故障 CPU 4/5。
训练/测试通过 `torch.distributed.run --standalone` 启动；默认单进程，符合本机一张 RTX 3090。
GPU 被其他任务使用时应等待空闲，不停止其他任务。CUDA 自检发现计算进程时会退出并提示稍后重试。

## 正式运行前需要的文件

当前验证环境与入口，不等于已完成真实数据训练/推理。自检报告分别记录
`environment_ready` 和 `runtime_assets_ready`，后者不会被当作依赖安装失败。

默认配置 `projects/configs/spacedrive/spacedrive_plus_qwen.py` 需要：

- `data/nuscenes/`：nuScenes 图像、点云、地图及标注；按照 `docs/data_model.md` 准备 OmniDrive 扩展数据。
- `data/nuscenes/nuscenes2d_ego_temporal_infos_train_with_command_desc.pkl` 和对应的 `val` 文件。
- 训练管线引用的 `lane_obj_train.pkl`，以及所用任务需要的 `vqa/`、`desc/`、`conv/`、`keywords/` 数据。
- `ckpts/Qwen2.5-VL-7B-Instruct-with-new-special-tokens/`：完整 VLM 权重、配置、tokenizer/processor 文件。
  按 `tools/token_additor.py` 添加位置 token，并核对 `datasets/utils/constants.py` 中 token ID。
- UniDepthV2 ViT-L 权重：代码从 `lpiccinelli/unidepth-v2-vitl14` 加载；需提前准备缓存或允许首次联网下载。
- 评测还需要已训练的 SpaceDrive `.pth` 检查点。原始 `dist_test.sh` 写死 `iter_21096.pth`；下面的入口可直接传入实际路径。

完整目录与缺失项记录在 `outputs/environment_setup/validation.json`。

准备好数据和权重后，单卡启动示例：

```bash
# 训练；配置原本按 8 卡设置，正式实验需按实际卡数重新核对学习率/迭代数。
taskset -c 0-3,6-31 bash tools/run_spacedrive.sh train \
  projects/configs/spacedrive/spacedrive_plus_qwen.py

# 评测；替换最后的检查点为实际文件。
taskset -c 0-3,6-31 bash tools/run_spacedrive.sh test \
  projects/configs/spacedrive/spacedrive_plus_qwen.py \
  workspace/spacedrive_plus_qwen/iter_21096.pth --format-only
```

启动脚本不自动加载或下载大模型，不更改公开配置的训练超参数；未用正式权重验证整模型显存占用。
训练时 WandB 如需离线记录，可在上述命令的 `bash` 前添加 `env WANDB_MODE=offline`。

## 本机兼容修复与版本记录

子模块的本地兼容改动保存在 `docker/torch21_mmcv/mmdetection3d_env.patch`，随主仓库一起提交。
在全新克隆并初始化子模块后、安装 mmdet3d 前，可从项目根目录应用：

```bash
taskset -c 0-3,6-31 git -C mmdetection3d apply ../docker/torch21_mmcv/mmdetection3d_env.patch
```

本机已应用这些改动，无需重复应用补丁。

- 两个子模块已初始化；`mmdetection3d` 位于 `v1.0.0rc6`，UniDepth 使用项目指定的提交。
- `mmdetection3d/requirements/runtime.txt`：Numba 改为 0.56.4（支持 NumPy 1.23），NetworkX 改为 >=2.5，匹配 SpaceDrive 依赖。
- 已按原文档应用 `docker/torch21_mmcv/` 的三份文件：MMCV 的 `_functions.py`、`distributed.py`，以及 mmdet3d 的 `__init__.py`。
  MMCV 原文件备份保存在 `outputs/environment_setup/backups/`。重新安装 MMCV 后需重新应用对应补丁。
- NumPy 固定 1.23.4；OpenCV 和 headless 固定 4.11.0.86，避免新 OpenCV 把 NumPy 升至 2.x。
- 官方 `decord-0.6.0-py3-none-manylinux2010_x86_64.whl` 内部 WHEEL 错标成 cp36，且 `top_level.txt` 的 RECORD 哈希陈旧。
  已先与 PyPI 公布的完整归档 SHA256 对照，再重打包修复 WHEEL/RECORD；所有 Python 代码和 native library 字节保持不变。
  修复 wheel：`outputs/environment_setup/wheels/decord-0.6.0-1-py3-none-manylinux2010_x86_64.whl`；实际 MP4 解码与 PyTorch bridge 已验证。

关键版本约束：`docs/environment_constraints.txt`。
完整已装版本：`outputs/environment_setup/requirements-installed.txt`（包含本机可编辑路径，作为安装记录使用）。
安装前版本：`outputs/environment_setup/requirements-before.txt`。
各项检查日志与 JSON 报告：`outputs/environment_setup/`。

Transformers `auto_docstring` 对自定义模型的文档告警、timm 旧导入路径提示，以及 UniDepth 的
NystromAttention/KNN 提示可能出现在日志中；它们不阻止本项目使用 UniDepthV2 推理。
原始 UniDepthV1 的 NystromAttention 和独立深度指标计算的 KNN 扩展不属于 SpaceDrive 文档要求的运行路径。
本次未安装会改变 PyTorch 版本的 xformers，也未编译这些可选扩展。
