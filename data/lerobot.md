# EgoSteer-RealWorld LeRobot v3 流式读取

本实现依据《EgoSteer 数据格式规范》v1.11（2026-08-30）及《EgoSteer
数据开源 · 工作档案》（2026-08-31），适配**已发布产物的实际 schema**。
格式版本 `v3.0` 与官方 Python 工具包版本 `lerobot 0.6.1` 是两回事。

## 代码组织

```text
src/dataset/
  wds/
    wds_dataset.py       # tar 读取、滑窗、混合管线
    vla_dataset.py       # VLAWdsDataset / VLALowLevelWdsDataset / UnifiedWdsDataset
    vlm_dataset.py       # VLMWdsDataset
  lerobot/
    lerobot_dataset.py   # 字段映射、Parquet / 视频读取、shuffle、worker 与 resume
    vla_dataset.py       # VLALeRobotDataset / VLALowLevelLeRobotDataset / UnifiedLeRobotDataset
    vlm_dataset.py       # VLMLeRobotDataset：独立 Parquet 图文问答源
  unified_dataset.py    # 两种 backend 共用的 VLA/VLM 包装器
  data_transforms.py    # 共用变换和 ViewDropoutConfig
  unified_vla_collator.py
  qwen3_vl_batching.py
  normalizer_utils.py
  sanity_checks.py
```

公开名称对齐为 `VLAWdsDataset` / `VLALeRobotDataset`、
`VLALowLevelWdsDataset` / `VLALowLevelLeRobotDataset`、
`UnifiedWdsDataset` / `UnifiedLeRobotDataset`，以及 `VLMWdsDataset` / `VLMLeRobotDataset`。
原生 VLM 的字段映射、混合配置见 [LeRobot VLM](lerobot_vlm.md)。现有训练配置名称不变，Python import 和 Hydra
`_target_` 已更新到对应子包；共享组件保留在父目录。

## 前一版的问题及修正

| 项目 | 前一版 | 本版 |
|---|---|---|
| 仓库 | 误写到 `C:/Users/rovanji/Desktop/X-lingual-vla/egosteer` | 在 `C:/PsiLab/projects/egosteer`，基于 `55f882e` |
| 状态/动作磁盘字段 | 把早期逻辑清单当成独立 Parquet 列，要求 `arm_state_left_wrist_pose` 等 | 只读取实际 `observation.state` 和 `action` 两个 74D 向量 |
| 训练目标定义 | 误把磁盘指令端 action 当作训练监督，并用它不同于下一 state 的测试证明 | 按用户明确的训练约定，监督 `action[t] = observation.state[t+1]`；磁盘 action 仅保留读取能力 |
| 元数据命名 | 要求 `head_camera_intrinsics`、`calibration_head2left` 等草案名 | 使用实际 `calibration.*` 列，矩阵 float64、行主序展平 |
| 任务 | 要求不存在的 `task_name` 列 | `tasks=[正式任务名]`，与词表及帧 `task_index` 对齐 |
| 指令 | 要求不存在的 `instruction_num` 列 | 读取 episode `instructions`，在模型适配时计算长度 |
| Train/val | 要求不同根目录，明确不读 `info.splits` | 同一根目录，按 `info.json.splits` 的 **episode** 区间选择，并核对 `split` 列 |
| 深度 | 假设 Parquet 数组/PNG，直接拒绝 depth MP4 | 独立 `head_depth/chest_depth` 视频；PyAV 保留 gray12le 码值，按持久化参数反量化 |
| 视频分片 | 未覆盖四路视频独立滚文件的实际情况 | 每个 feature 各用自己的 file/chunk/from/to 时间元数据 |
| 当前代码能力 | 基于旧 `1a600b3`，没有目标仓库新增的 `val_stride` | 新 reader 保留验证抽样和现有训练/评估接入 |
| 验证 | 用自行假定的独立列及数组深度验证自身 | 按发布规范生成 74D、任务索引、split、四路真实编码视频的测试 |

前一版保留的 BC 时间窗口/几何变换思路本身可以复用，但不能证明它能读取发布数据。
本版重新核对了 `C:/PsiLab/projects/egosteer-RL` 的 stream 顺序，仅移植数据流逻辑，
不读取 reward、bootstrap、next_obs、接管标记或策略版本。

## 磁盘结构与划分

```text
<root>/
  meta/info.json
  meta/tasks.parquet
  meta/stats.json
  meta/episodes/chunk-*/file-*.parquet
  data/chunk-*/file-*.parquet
  videos/observation.images.head/chunk-*/file-*.mp4
  videos/observation.images.chest/chunk-*/file-*.mp4
  videos/observation.images.head_depth/chunk-*/file-*.mp4
  videos/observation.images.chest_depth/chunk-*/file-*.mp4
```

发布集 `info.splits` 为 `train: "0:54261"`、`val: "54261:54454"`。
这些是 episode 索引，不是全局帧索引，不存在必须额外生成的 `root/train`、`root/val`。
代码从文件读取实际区间，不写死上述数量。数据/视频路径服从 `info.json` 的模板。

读取全部 `meta/episodes` 分片，只投影必要列，跳过大的 `stats/*`。低维按 episode/frame
索引从候选 Parquet row group 取行，读取前检查 footer；视频采用各 feature 独立的时间偏移。
同一个 data 文件可含多个 episode，每个 episode 的低维行完整位于它声明的文件内，符合该发布集。

## 帧级字段逐项核对

| 实际字段 | 类型/形状 | 读取语义 |
|---|---|---|
| `observation.images.head` | video `[480,640,3]` | H.264 RGB → uint8 HWC，头部即世界系 |
| `observation.images.chest` | video `[480,640,3]` | 同上，独立视频定位；由 `load_chest` 控制读取 |
| `observation.images.head_depth` | video `[480,640,1]` | HEVC gray12le → float32 米；由 `load_depth` 控制读取 |
| `observation.images.chest_depth` | video `[480,640,1]` | 同上；两个加载开关均开启时读取 |
| `observation.state` | float32 `[74]` | 当前参考时刻的实测状态，下面八段布局 |
| `action` | float32 `[74]` | 文档中的磁盘指令端数据，同布局；原始 reader 可读取，**训练不使用该列** |
| `timestamp` | float32 scalar | `frame_index / fps`，已经对齐的 30Hz 网格，不重新插值/对齐 |
| `frame_index` | int64 scalar | episode 内连续帧号，从 0 起 |
| `episode_index` | int64 scalar | 关联 episode 元数据及 split |
| `index` | int64 scalar | 全局帧号，核对 `dataset_from_index + frame_index` |
| `task_index` | int64 scalar | 指向正式任务词表，不是语言指令编号 |

### 74D 状态和动作：两个向量使用同一布局

| 切片 | 草案中的逻辑内容 | 单位/语义 | 现有 48D 模型如何使用 |
|---|---|---|---|
| `[0:7]` | 左臂 7 关节 | rad，arm1_joint_link1…7 | 保留在原始 74D，模型不直接用 |
| `[7:14]` | 右臂 7 关节 | rad，arm2_joint_link1…7 | 同上 |
| `[14:20]` | 左手 6 自由度 | **电机归一化值，不是弧度** | 同上，不误做关节角变换 |
| `[20:26]` | 右手 6 自由度 | 同上；拇指旋转量程 0–0.6，其余 0–1 | 同上 |
| `[26:35]` | 左腕位姿 | 世界系 xyz 米 + rot6d | xyz → wrist18 `[0:3]`，rot6d → `[6:12]` |
| `[35:44]` | 右腕位姿 | 同上 | xyz → wrist18 `[3:6]`，rot6d → `[12:18]` |
| `[44:59]` | 左手 5 指尖 | 世界系，拇/食/中/无名/小指，每指 xyz 米 | hand30 `[0:15]` |
| `[59:74]` | 右手 5 指尖 | 同上 | hand30 `[15:30]` |

因此，不能把 `vector[26:44]` 直接当作 EgoSteer wrist18：源布局是
`Lxyz,Lrot6d,Rxyz,Rrot6d`，模型需要 `Lxyz,Rxyz,Lrot6d,Rrot6d`。
`LeRobotEpisodeReader.read_lowdim` 返回完整 74D，`unpack_motion` 才进行这一步映射。

rot6d 原样保留：`[R00,R10,R20,R01,R11,R21]`，前两列按列优先展平。
发布集的腕/指尖已经做过 FK、手部坐标变换及 cam2base 逆变换，**不能再次做 FK、
手部坐标旋转或手眼标定变换**。现有 `process_state_action` 再把指尖转换到各自腕系，
并按配置生成相对动作、应用模型 normalizer，这些是模型表示步骤。

**用户明确指定训练监督 `action[t] = state[t+1]`，优先于文档的采集侧 action 描述。**
loader 从 `observation.state` 取未来状态构造训练目标，而不是使用同一行的磁盘 `action`。
锚点 `k` 的 H 个动作对应 `state[k+1 + j*action_stride]`（j=0…H-1）。
这个全等关系在坐标变换、相对化、归一化前成立；经过不同的模型表示变换后，输入 state 与
输出 action 张量不要求仍逐值相同。资料中的采样延迟、FK 常数和标定误差不是此次 loader
需要“修复”的内容；它们已经固化在数据产物中。

## Episode 字段逐项核对

| 实际列 | 形状/内容 | 用途 |
|---|---|---|
| `episode_index` / `length` | int64 | split 与窗口边界 |
| `tasks` | `[正式任务名]` | 与 `meta/tasks.parquet` 任务词表一致；输出 `dataset_name` |
| `instructions` | 英文指令列表 | 每 episode 独立；训练随机取一条，验证取第一条 |
| `calibration.head_intrinsics` | float64 `[9]` | reshape K 行主序，取 `[fx,fy,cx,cy]`，resize 后沿用现有内参缩放 |
| `calibration.chest_intrinsics` | float64 `[9]` | 同上 |
| `calibration.head_cam_to_left_base` | float64 `[16]` | `p_left_base = T @ p_head`；保留，不重复作用于已在世界系的 FK |
| `calibration.head_cam_to_right_base` | float64 `[16]` | 同上，右臂 |
| `calibration.chest_cam_to_left_base` | float64 `[16]` | `p_left_base = T @ p_chest`；保留 |
| `calibration.chest_cam_to_right_base` | float64 `[16]` | 同上，右臂 |
| `calibration.head_world2cam` | float64 `[16]` | 恒 I；验证头部相机就是世界系 |
| `calibration.chest_world2cam` | float64 `[16]` | 直接使用发布值，不改成右臂链重新估计 |
| `split` | train/val | 若存在，核对它与 info.json episode 区间一致 |
| `dataset_from_index` / `dataset_to_index` | 半开全局帧区间 | 核对 length、帧级 index |
| `data/chunk_index` / `data/file_index` | 文件索引 | Parquet 定位 |
| `videos/{feature}/chunk_index` / `file_index` | 每 feature 独立 | 不从 RGB 的索引推导 depth 索引 |
| `videos/{feature}/from_timestamp` / `to_timestamp` | 每 feature 独立的文件内秒数 | episode 帧 `k` 对应 `from_timestamp + k/fps` |
| `stats/*`、metadata 自身定位列 | 工具链生成 | loader 无需物化 |

内参、变换矩阵均为行主序存储，齐次点为列向量、左乘变换。
规范给出的胸部外参派生式是
`inv(chest_cam_to_left_base) @ head_cam_to_left_base`；代码读取已发布结果，不重新标定。

发布集没有必需的 `instruction_num`。这是送给现有 EgoSteer adapter 的派生字段，值为
`len(instructions)`。原生 LeRobot 合并标注后可有 12 条英文指令，全部保留；内部 WDS 回归集
最多 5 条并带中文的规则不应搬来截断原生数据。因此同一条有效 motion 的模型表示可以对齐，
但超过五条的 episode 在两种数据产物上的训练语言分布并不完全一样。

## 深度解码与单位

深度编码是 HEVC Main12、gray12le、lossless=1，量化本身是 12-bit 对数映射。
`info.json.features[depth_key].info` 的 `is_depth_map` 和 `video.depth_min`、
`video.depth_max`、`video.shift`、`video.use_log` 决定反量化。发布参数为
0.01–10 米、shift=3.5、use_log=true；实现读取元数据，不写死这些默认值。

读取使用 PyAV 原始 gray12le plane，不能经 OpenCV/RGB/uint8 转换。四路视频各自定位，
按关键帧 seek 后顺序解码；缓存逐帧结果、decoder 位置，重叠窗口复用已解码帧。
缓存有界，每个 DataLoader worker 独立建立，pickle 不携带文件句柄或视频帧。

本版输出 float32 **米**，现有 `process_image` 对浮点深度不会再除以 1000。
规范 §5.2 明确 0 是无效值；因此 code=0 恢复为 0。这里有一项已核对的工具链细节：
官方 lerobot 0.6.1 通用 `dequantize_depth` 会把 code=0 映射到 depth_min；本版对有效码使用
相同反量化公式，另外按发布规范恢复无效零。实际完整产物仍需对该边界做数据侧复核。

`load_depth=False` 不打开任何深度视频。`read_window` 在开启时返回头/胸深度；当前仓库的
`VLAWdsDataset.sample_to_data` 会处理/检查深度，但没有输出 `depth_values` 训练目标，
本版沿用该现状，不在数据格式适配中擅自新增模型头。

## 流式样本规则

1. 在指定 split 内按 episode_index 排序，再 round-robin 分配到全局 rank/worker。
2. 每个 worker 每轮打乱自己的 episode 顺序，episode 内锚点按帧号递增。
3. 训练以 `drop_ratio` 在读取前丢弃锚点；保留样本完整读取、变换后进入 shuffle buffer。
4. shuffle warmup 与 RL 一致：到 `shuffle_initial` 开始产出，未满时每轮取两条，容量满后取一条。
5. 验证顺序、有限、不随机丢弃；沿用当前仓库 `val_stride`，按 worker 的窗口流每 N 条取一条，
   发生在解码前，跨 episode 不重置计数。

有下一状态的帧才是 BC 锚点，即 `0…T-2`；末帧只用于观测/监督目标，不伪造它的下一状态，
单帧 episode 不产生训练样本。这来自上述训练定义，不是引入 RL transition。
状态与图像保留各自的 horizon/stride，动作保留独立 stride；历史和尾部按现有
repeat/truncate 逻辑处理。repeat 模式重复最后可用状态，truncate 模式丢弃越界目标。
动作先对有效部分变换、归一化，再补零并生成 mask。可用已有 RGB/静态相机外参生成
world-model future-frame 监督，不增加磁盘字段，不改变原实验的 world_model 开关。

训练按 step 运行，数据无限循环。shuffle 容量与初始预热均为 **4096/worker**。
第一次产出前 materialize 4096 条；产出后队列有 4095 条，下次先补到 4096 再取一条。
没有原先从 64 逐渐增长到 256 的阶段。它仍然存完整解码样本：默认双视角、6 帧历史加
1 帧 future、384×384 uint8，仅图像约 23.6 GiB/worker（4 worker 约 94.5 GiB），还不含
视频缓存、Parquet、processor 和预取 batch。这个容量按用户要求设置。

## Checkpoint / resume

现有 DCP checkpoint 现在同时保存模型、optimizer、scheduler、training_state 和数据流状态。
数据流使用独立的 `app.data_stream.rank_<rank>` 键，不让多个 rank 的 bytes 叶子被当作同一份
副本去重。WDS 路径仍使用原来的 checkpoint 结构，不凭空声明它能恢复数据位置。

一次保存包含：

| 状态 | 含义 |
|---|---|
| `consumed_batches` | 主训练循环已经处理的 microbatch 数；包含梯度异常而跳过更新的 batch |
| worker 逻辑编号 | 用来恢复 DataLoader 的轮转交付顺序 |
| source cursor | round、episode 在该轮的位置、下一候选帧、该轮 attempted/usable 数 |
| drop RNG | 下一次 Bernoulli 丢弃从哪里继续 |
| shuffle RNG | 下一次随机选择哪个 resident 槽位 |
| resident queue | 每条为 `(episode位置, anchor, augmentation_seed)`，不是图像 |
| delivered | 每个 worker 已交付的样本计数 |
| signature | 数据 metadata、normalizer、窗口/变换配置、collator 配置、seed、容量、batch size、worker/rank 数等 |

**以训练消费边界为准，不以 worker 实时进度为准：** sample 的内部属性携带 worker 状态；
collator 在一个 batch 组装完后将状态序列化为不可变 bytes；训练主进程在该 batch 经过
`train_step` 后才记录它。预取领先的样本/队列不会覆盖 checkpoint 中的已消费位置。
该内部状态在模型前向前从 batch 中移除，不是模型输入字段。

恢复流程：

1. 加载前检查 checkpoint 是否含当前 rank 的数据流状态。
2. DCP 加载权重与训练状态，再校验数据流 signature。
3. 新 DataLoader 的物理 worker 0 映射到 `consumed_batches % num_workers` 对应的逻辑 worker，
   其他 worker 同样轮转；未曾交付过 batch 的 worker 从初始状态开始。
4. 从保存的 source cursor 和 RNG 直接继续，不从头重放全部历史索引。
5. 恢复 resident 描述符。旧 resident 按需解码，不重新随机预热、丢掉原队列，也不把数万张
   图像存进 checkpoint。新 source 样本仍按顺序 materialize 后入队，保留流式解码局部性。
6. 每条 sample 的 Python、NumPy、Torch CPU 和 Albumentations 随机源都由描述符 seed
   控制，因此恢复队列时的指令选择、颜色增强、depth 增强、view dropout 可复现。
7. 训练循环保留跨逻辑 epoch 的同一个数据 iterator；根据已消费 microbatch 数恢复 epoch
   和 batch offset，并在边界之前限流，避免多取、丢弃一个 batch。

使用方式仍是原训练参数：

```bash
torchrun --standalone --nproc_per_node=1 train.py \
  --config-name=experiment/egosteer_lerobot \
  training.resume_checkpoint_path=/path/to/checkpoint
```

要求相同 batch size、worker 数、world size、梯度累积、epoch 长度、采样/处理配置和
normalizer。metadata 或 signature 不符会明确报错。大体积视频和 Parquet 内容不做全量哈希，
需要保持原始文件不可变；精确样本恢复也要求相同代码和依赖版本。profiling 时间和日志计数
不属于复现目标。本功能恢复数据输入，不额外承诺 CUDA 随机训练轨迹逐 bit 一致。

旧 checkpoint 没有 source/queue 状态，无法无损恢复：`resume_checkpoint_path` 会明确失败，
不会假装跳过若干 global_step 就恢复了。若只取旧权重开始新数据流，使用现有
`training.finetune_checkpoint_path`。当前精确恢复使用普通 in-order DataLoader；支持纯 VLA
和两条原生 LeRobot 流的固定比例 VLA/VLM 混合。混合 checkpoint 同时保存两条流的队列、
游标和消费计数，不支持 WebLoader 跨 worker 重排，也不能混入不可恢复的 WDS VLM 流后
继续声称精确恢复。

## 与当前 WDS 管线对比

| 逻辑 | LeRobot stream | WDS train stream |
|---|---|---|
| 源读取 | Parquet + 每路独立 MP4，索引窗口 | tar 内连续帧组成滑窗 |
| 源循环 | episode 固定分配到 worker，每轮各访问一次 | shard 有放回独立重采样，各 worker/rank 可重复抽中同一 shard |
| 窗口锚点 | `0…T-2`，要求下一 state | 所有帧可作锚点，再按 high_quality 过滤 |
| 动作来源 | 从 state 的 `k+1+j*stride` 构造 | 读取 lowdim 中的 wrist_action/hand_action，数值关系由 producer 保证 |
| 丢弃 | 先丢锚点索引，再读取/解码窗口 | 已读取 meta/lowdim、组装窗口引用后 keep_ratio 过滤 |
| shuffle | 解码、变换、增强后入队 | 压缩字节/帧引用入队，出队后解码、变换、增强 |
| 容量/预热 | 4096 / 4096，第一批之后不再增长 | 16384 / 4096，预热后继续增长 |
| 多源 | VLA 一个 released root，可固定比例混合独立 LeRobot VLM root | 支持按权重 RandomMix 多个 subset，也可接 VLM |
| DAgger | 不读取 high_quality/is_intervention | 可丢低质量锚点并截断 future/action |
| 历史、相对动作、归一化、图像变换、collator | 复用原有模型样本处理 | 原处理逻辑 |
| 深度 | HEVC gray12le + 反量化为米 | npy，uint16 毫米时再转米 |
| 验证抽稀 | worker 窗口流的 val_stride，读取前执行 | worker/subset 窗口流的 val_stride，窗口组成后执行 |
| resume | 消费边界的 source/queue/RNG + 样本 seed；DCP 按 rank 保存 | 保存模型训练状态，不保存 WDS 数据流位置/队列 |
| 随机增强 | 每次 source 出现有独立确定 seed，同一 resident 恢复可复现 | 依赖 worker 随机状态与出队时机，当前无数据流恢复保证 |

相同锚点和相同原始监督下，几何/归一化/padding 的模型表示可一致；但源采样分布、随机化
时机、内存、末帧规则、生产端 action 定义和 resume 语义并不等价。原生指令全保留与内部
回归 WDS 的五条上限是数据产物差异，不是 WDS loader 本身强制的上限。

## 配置与运行

依赖增加 `pyarrow>=16` 和 `av>=15,<17`。选择
`src/config/experiment/egosteer_lerobot.yaml`，根目录配置在
`src/config/dataset_paths/vla_lerobot.yaml`。原 WDS 配置保持可用。

`meta/stats.json` 是原始特征统计，**不是模型空间 normalizer**。用 train split 的低维有限扫描
拟合，不读视频、不读 val、不计入截断补齐行：

```bash
python -m src.workspace.compute_lerobot_norm_stats \
  --output_dir outputs/normalizer/lerobot \
  lerobot_root=/share_data/yifan/EgoSteer-RealWorld
```

使用现有 Linux/FSDP 训练入口：

```bash
torchrun --standalone --nproc_per_node=1 train.py \
  --config-name=experiment/egosteer_lerobot \
  lerobot_root=/share_data/yifan/EgoSteer-RealWorld
```

实现已验证 H.264/HEVC 视频、独立深度分片偏移、74D 重排、指令数量、split 隔离、
WDS 模型表示对照、val_stride、worker 序列化及无视频 normalizer。验证脚本不随仓库发布。
当前环境没有完整 2.9TB 产物，以上验证不代替全量数据验收或 GPU 训练。
