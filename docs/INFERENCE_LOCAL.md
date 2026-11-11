# 本机 nuScenes 小批量推理

使用已有 SpaceDrive Python 环境与用户提供的完整 iter_21096.pth。无需下载基础模型权重。
重复运行会清空当前 results.json，并移除旧 metrics/图片，避免失败后误读上次结果。

小文件 assets 已通过 Hugging Face 官方仓库的 snapshot_download 获取：
Qwen/Qwen2.5-VL-7B-Instruct 的 config、tokenizer、processor、generation_config，
以及 lpiccinelli/unidepth-v2-vitl14 的 config.json。推理为离线运行。

## 直接运行

以下命令可从任意工作目录执行。首次加载约需 1–2 分钟，随后逐帧生成轨迹。
GPU 显存被其他任务占用时，应等待空闲后重新执行，不停止其他进程。

```bash
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/SpaceDrive/bin/python -u /data/qh/phdCode/work4/SpaceDrive/tools/infer_checkpoint_smoke.py --checkpoint /data/qh/phdCode/work4/models/SpaceDrive/SpaceDrive-Qwen2.5-VL-7B/iter_21096.pth --data-root /data/qh/nuscenes/v1.0-trainval --assets /data/qh/phdCode/work4/SpaceDrive/outputs/qwen_inference_smoke_20261109_run01/assets --output-dir /data/qh/phdCode/work4/SpaceDrive/outputs/qwen_inference_smoke_20261109_run01/group3 --num-samples 3 > /data/qh/phdCode/work4/SpaceDrive/outputs/qwen_inference_smoke_20261109_run01/group3.log 2>&1
```

## 输出及范围

- group3.log：完整运行日志，成功时出现 WITNESS 和 SUCCESS。
- group3/results.json：每帧预测轨迹、真实轨迹、逐时刻 L2 距离与耗时。
- group3/metrics.json：3 帧平均误差、显存峰值及生成配置。
- group3/trajectories.png：轨迹对比，横轴为横向位置，纵轴为前向位置，单位米。
- group3/cameras.png：第一帧的六路输入相机图像。
- group3/checkpoint_audit.json：学习参数严格匹配及旧版本缓存的审计记录。
- group3/samples.json：样本 token、场景、时间戳及输入文件路径。

样本选择：验证集按时间排序后的首组连续 3 帧，同一场景，未来 3 秒标注完整，
且每帧最终位移超过 5 米。六路图像按权重训练配置缩放到 640×640。
保持 checkpoint 原始 bf16/fp32 权重，不量化；使用 SDPA，UniDepth 逐相机执行以控制显存，并在视觉语言阶段将深度模型临时移至 CPU，同时释放空闲缓存，为大块注意力计算预留显存。
随机种子为 888，生成配置从原始 Qwen assets 加载。
使用 checkpoint 内保存的 prompt 配置，load_ego_command_in_question=False，
ego_status 为空，不向模型提供真实未来轨迹或由其生成的导航描述。
子集 command_desc 由原项目脚本生成以满足数据格式要求，但未参与该配置下的模型推理。
测试 pipeline 的 num_commands 改为 3，仅跳过初始化时未使用的额外 GT 字典读取。
原始共享数据与权重不修改；data/nuscenes 为指向现有数据的符号链接。

旧模型名 PEADVLM 映射至发布代码中的 SpaceDrive，loss_pos_method 映射至 pe_decode_method。
所有学习参数键名与形状必须严格匹配。旧版 rotary inv_freq 须与当前派生值逐元素相等；
旧版 cached_penc 重新计算，未使用的 position_range 忽略。
新增 rope_3d 常量仅在 use_rope=False 时允许由代码构建，未忽略任何缺失学习参数。

此结果仅用于验证真实数据推理链路；3 个相邻样本不足以代表完整 nuScenes 指标。
