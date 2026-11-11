# 本机完整 nuScenes 规划评测

评测集合采用发布项目 data.test 的完整 val：150 个场景、6019 帧。
全部帧均进行推理，包括静止帧和未来标注不完整的场景末尾；按发布项目规则，
完整未来标注的 5119 帧用于默认轨迹误差、碰撞率和可行驶区域边界指标。
这不等同于官方 v1.0-test；后者无公开目标标注，此次不是官方排行榜提交。

## 启动或断点续跑

从任意工作目录执行以下命令。已有存活任务时返回 ALREADY_RUNNING；
否则启动一个与终端断开的后台进程，自动 --resume。关闭终端不会终止任务。

```bash
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/SpaceDrive/bin/python /data/qh/phdCode/work4/SpaceDrive/tools/launch_full_eval.py --checkpoint /data/qh/phdCode/work4/models/SpaceDrive/SpaceDrive-Qwen2.5-VL-7B/iter_21096.pth --data-root /data/qh/nuscenes/v1.0-trainval --assets /data/qh/phdCode/work4/SpaceDrive/outputs/qwen_inference_smoke_20261109_run01/assets --output-dir /data/qh/phdCode/work4/SpaceDrive/outputs/spacedrive_qwen_full_val_20261109_run01
```

模型加载及数据准备通常需 1–3 分钟。完成准备后每帧约需 5 秒，整套推理
预计 8–10 小时，随后自动进行 CPU 几何评估。程序使用已有 SpaceDrive 环境，
不安装或升级依赖。CPU 亲和性为 0–3,6–31（避开故障 CPU 4/5），
子进程继承；GPU 被其他任务占用时等待，不停止其他进程。

## 输出

运行目录：outputs/spacedrive_qwen_full_val_20261109_run01/

- worker.json：后台进程 PID、真实命令、日志位置。
- inference.log：持续推理日志，出现 START total=6019 表示全量推理已开始。
- status.json：阶段、成功/失败/剩余帧数、当前帧、预计剩余秒数和更新时间。
- run_config.json / samples.json：全量选择规则、全部 token、场景与标注掩码。
- data_preflight.json：36114 张输入图像完整性检查。
- records/<token>.json：每帧预测、有效标注掩码、距离误差、耗时或失败原因；逐帧原子保存。
- raw/<token>：项目原始回答格式，可输入项目默认评估脚本。
- results.json：全部推理结束后汇总，运行期间以 records/ 和 status.json 为准。
- metrics.json：实时、明确标记 partial 的 L2 指标；全部推理和评测通过后标记 complete。
- evaluation_default.log：原项目默认 1/2/3 秒 ADE、碰撞率与越界率的 CPU 评估日志。

只有 status.stage=completed 且 metrics.metrics_status=complete 才代表整个评测完成。
若出现失败帧会继续尝试其余帧，记录 failures.json 并报错；重新运行启动命令将重试失败帧。
未识别的运行错误立即报错，不能把不完整结果误认为完整指标。
恢复时核对权重/标注路径、大小、mtime、token 列表和配置，禁止混用不同实验。
该 checkpoint ego_status 为空，未使用历史特征；随机种子为 888+时间顺序索引，以保证续跑一致。

## 指标含义

project_ade_m：分别平均未来前 2/4/6 个点的 L2，与默认评估脚本一致。
endpoint_l2_m：1/2/3 秒单独时刻的 L2，单独列出，避免与 ADE 混淆。
project_default_evaluation：项目默认几何评测；碰撞与越界值为百分比。
所有指标仅使用真实标注和模型输出，不采用伪造轨迹、GT 提示词或缺失点填充。
完整未来标注不足的 900 帧仍推理，但不进入默认计分分母。

发布评测器的数学计算不改动，使用 --num_threads 1 避免共享计数器竞态；
增加 EVALUATION_ERROR 日志标记，使被原代码捕获的异常可被任务检测到。
结束时检查计分分母为 5119、9 项指标齐全，并与独立 L2 汇总交叉核对。

权重结构、离线 assets、640×640 处理、原始 bf16/fp32 精度、分阶段深度模型 CPU 暂存，
沿用已验证的小批量设置，见 INFERENCE_LOCAL.md。全量程序不按位移或未来有效性筛选输入帧。
