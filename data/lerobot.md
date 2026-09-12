# EgoSteer-RealWorld LeRobot v3 流式读取

本实现依据《EgoSteer 数据格式规范》v1.11（2026-08-30）及《EgoSteer
数据开源 · 工作档案》（2026-08-31），适配**已发布产物的实际 schema**。
格式版本 `v3.0` 与官方 Python 工具包版本 `lerobot 0.6.1` 是两回事。

## 代码组织

LeRobot 不导入或继承 src.dataset.wds 中的实现。窗口配置、媒体解码、VLA/VLM
样本构建及 normalizer 扫描均在 lerobot 子目录内实现。Reader 在初始化时一次读取 schema
和 row-group 索引，VLM 复用这些 schema 信息，不再为查询列名打开文件。
本地 LeRobotDataset 负责配置、worker 分配、验证集和 collator；__iter__ 只调用 build_pipeline。
build_lerobot_pipeline 显式串联读取/组窗、shuffle、媒体解码和 preprocess，结构与 WDS 管线一致。
视频缓存直接归入 episode reader；source 游标、预热与 shuffle 在同一个管线函数内维护，
不再单独设置 VideoFrameCache、EpisodeSource 或流包装类。
encode_frame 直接返回 JPEG/NPY 字节，组窗直接生成最终媒体字段。VLA 训练与 normalizer
扫描共用本地 motion 检查/变换方法。父目录的 data_transforms、
sanity_checks、unified_dataset、collator 和模型 normalizer 是存储格式无关的公共组件。

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

## 校验职责

Reader 按发布格式直接读取必需字段，缺字段、损坏 Parquet/视频和无法 reshape 的数据由
PyArrow、PyAV、NumPy 或直接字段访问报错，管线统一附带 episode/frame 定位。
不再逐层重复审计字段类型、指令文本、像素 dtype 或自身生成的窗口索引范围。
样本的有限值、指令、图像质量、几何和归一化检查统一由出队后的 DataChecker 执行；
非有限 motion 不再在 Reader 中提前终止，而是按样本质量规则跳过。
仍保留 74D 状态布局、头部世界系、episode/帧/时间对应、视频尺寸/FPS/PTS、
gray12le 深度及反量化参数，以及 checkpoint 的配置、worker 顺序和消费计数约束。
collator 只冻结状态，消费端统一校验数据流名称、worker 和交付计数。

## 流式样本规则

1. 在指定 split 内按 episode_index 排序，再 round-robin 分配到全局 rank/worker。
2. 每个 worker 每轮打乱自己的 episode 顺序，episode 内锚点按帧号递增。
3. 训练以 `drop_ratio` 在读取前丢弃锚点；保留窗口读取原始低维字段，视频顺序解码后先 resize 到
   `target_image_size`，再以 JPEG quality 80 编码 RGB、以 NPY 无损保存 float32 米单位的 depth 入队。每个窗口独立保存媒体字节，出队后解压，再调用 LeRobot 自身的
   `sample_to_data` 执行数据质量检查、随机增强、指令选择、变换及归一化。
   出队后检查失败的窗口直接跳过，不计入 delivered；source 游标和 shuffle RNG 的推进保留。
4. shuffle warmup 与 RL 一致：到 `shuffle_initial` 开始产出，未满时每轮取两条，容量满后取一条。
5. 验证顺序、有限、不随机丢弃；沿用当前仓库 `val_stride`，按 worker 的窗口流每 N 条取一条，
   发生在解码前，跨 episode 不重置计数。

有下一状态的帧才是 BC 锚点，即 `0…T-2`；末帧只用于观测/监督目标，不伪造它的下一状态，
单帧 episode 不产生训练样本。这来自上述训练定义，不是引入 RL transition。
状态与图像保留各自的 horizon/stride，动作保留独立 stride；历史和尾部按现有
repeat/truncate 逻辑处理。repeat 模式重复最后可用状态，truncate 模式丢弃越界目标。
动作先对有效部分变换、归一化，再补零并生成 mask。可用已有 RGB/静态相机外参生成
world-model future-frame 监督，不增加磁盘字段，不改变原实验的 world_model 开关。

训练按 step 运行，数据无限循环。shuffle 容量为 **16384/worker**，初始预热为 **4096/worker**。
第一次抽样前准备 4096 个压缩窗口；未满时每次补两条、取一条，逐步增长到容量。
满容量后每次补一条、取一条，正常产出后队列有 16383 条。
若出队样本被质量检查过滤，继续补入和抽样，直至得到有效样本。入队前只做确定性的 resize 和内参缩放，不做随机增强或 state/action 模型输入变换。
使用存储格式无关的公共 data_transforms.resize_frames；RGB 使用 INTER_LINEAR，depth 使用 INTER_NEAREST，内参按原始尺寸到目标尺寸缩放一次；
出队后共用预处理发现尺寸已匹配，不会再次 resize 或缩放内参。target_image_size=null 时保留原尺寸。
队列保存原始低维字段、文本及压缩媒体，不保存完整解码图像或最终模型张量。
VLA 队列采用 WDS 的 image_frame_refs / future_frame_refs，成员键为 image.jpg、
chest_image.jpg、depth.npy、chest_depth.npy。VLM 队列使用 image_N.jpg 字节字段。
出队调用 LeRobot 自身的 decode_sample_media，以及本地
sample_to_data；模型 collator 使用父目录公共组件；验证也走相同 JPEG/NPY 媒体路径，但不随机增强、不建立 shuffle 队列。
JPEG 有损，RGB 不再保证与编码前逐像素一致；depth、state/action 不引入 JPEG 误差。
不做跨窗口媒体共享。底层每路视频保留至多 `frame_cache_size` 个已 resize 的解码帧，最多保留
`video_reader_cache_size` 路 decoder/缓存上下文；它们不属于 shuffle 队列。
默认 RGB 训练每路 256×384×384×3=108 MiB，双路约 216 MiB，四路上限约 432 MiB/worker。
启用 depth 时每张 float32 depth 为 576 KiB；原尺寸诊断读取则按原尺寸占用。
VLM 视频走同一路径；VLM 独立图片也 resize 后编码为 JPEG quality 80。验证与 normalizer 扫描不使用此队列。

### 主机内存估算（16384 容量）

默认 RGB 双视角各 6 帧历史 + 1 帧未来、384×384、depth 关闭，每个窗口保存 14 张压缩帧。
每张未压缩 RGB 为 432 KiB。令 S 为平均压缩帧字节数，
满队列图像载荷约为 16384×14×S，不计跨窗口去重收益。每 GPU 有 4 个训练 worker，与 WDS 一致。
训练与验证 loader 各配置 4 workers，验证期间可同时存在；验证不建立训练 shuffle 队列。

下表仅估算每 GPU 对应的 4-worker **CPU 图像队列载荷**，不代表总 RSS 或 GPU 显存。
下表压缩帧大小是假设，需要真实数据测量。两张用户提供的图像 resize 后实测 JPEG quality 80
分别为 30.73 / 31.69 KiB，平均 31.21 KiB，据此队列图像约 6.83 GiB/worker、
27.31 GiB/GPU（4 workers）；这两张图的平均值不能代替完整数据集统计。

| 平均压缩帧大小 | 每 GPU 的 4-worker 图像队列 |
|---|---:|
| 50 KiB | 43.75 GiB |
| 100 KiB | 87.5 GiB |
| 200 KiB | 175 GiB |

此外，满窗口时原始 state/action、内参/外参数组载荷约 118.5 MiB/worker，
还需加 Python 对象、episode 元数据、Parquet 缓存、视频 decoder、已解码帧缓存、processor、
预取/pinned batch 等。
VLA/VLM 混合时，两条流各有 16384 容量，不按 6:2 比例缩减容量；
若 VLM 每样本一张独立图片且平均压缩后 100 KiB，其队列另加约 6.25 GiB/GPU。
相同帧数下，480×640 提前 resize 为 384×384 使未压缩像素载荷减少 52%；
实际压缩后节省比例仍取决于内容。精确总内存需在真实数据和最终 processor 配置上测量。

## Checkpoint / resume

现有 DCP checkpoint 现在同时保存模型、optimizer、scheduler、training_state 和数据流状态。
单流和固定配比 VLA/VLM 混合流共用一个 StreamCheckpoint；消费计数、worker 校验与
保存/恢复只实现一套。磁盘上仍兼容原有的单流及 vla_vlm_v1 混合包装格式。
数据流使用独立的 `app.data_stream.rank_<rank>` 键，不让多个 rank 的 bytes 叶子被当作同一份
副本去重。WDS 路径仍使用原来的 checkpoint 结构，不凭空声明它能恢复数据位置。

一次保存包含：

| 状态 | 含义 |
|---|---|
| `consumed_batches` | 主训练循环已经处理的 microbatch 数；包含梯度异常而跳过更新的 batch |
| worker 逻辑编号 | 用来恢复 DataLoader 的轮转交付顺序 |
| source cursor | round、episode 在该轮的位置、下一候选帧、该轮 attempted/usable 数（尝试/成功读取的窗口数） |
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
5. 按 episode/anchor 排序读取 resident 描述符，通过与新 source 相同的解码→resize→JPEG80/NPY 编码路径重建独立压缩窗口，
   放回原队列槽位。排序只影响读取顺序，不改变出队顺序或 source/RNG；不重新随机预热，
   也不把图像或压缩媒体载荷存进 checkpoint。队列重建完成后才继续推进 source。
6. 每条 sample 的 Python、NumPy、Torch CPU 和 Albumentations 随机源都由描述符 seed
   控制，全部在出队后执行，因此指令选择、颜色增强、depth 增强、view dropout 可复现。
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

媒体队列使用 stream spec version 3，并记录 JPEG quality 80 / depth NPY 编码配置；
version 1 在入队前做质量过滤和增强，version 2 使用 zlib，两者均不接受精确数据恢复，
以免混用不同取样或像素语义。旧 checkpoint 没有 source/queue 状态，同样无法无损恢复：`resume_checkpoint_path` 会明确失败，
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
| shuffle | 视频解码、resize 后 JPEG80/NPY 字节窗口入队；出队后解压、检查、变换、增强 | 压缩字节/帧引用入队，出队后解码、检查、变换、增强 |
| 容量/预热 | 16384 / 4096，预热后继续增长 | 16384 / 4096，预热后继续增长 |
| 多源 | VLA 一个 released root，可固定比例混合独立 LeRobot VLM root | 支持按权重 RandomMix 多个 subset，也可接 VLM |
| DAgger | 不读取 high_quality/is_intervention | 可丢低质量锚点并截断 future/action |
| 历史、相对动作、归一化、图像变换、collator | 本地样本处理；共用公共几何/归一化/图像函数及 collator | 原处理逻辑 |
| 深度 | HEVC gray12le + 反量化为米 | npy，uint16 毫米时再转米 |
| 验证抽稀 | worker 窗口流的 val_stride，读取前执行 | worker/subset 窗口流的 val_stride，窗口组成后执行 |
| resume | 消费边界的 source/queue/RNG + 样本 seed；DCP 按 rank 保存 | 保存模型训练状态，不保存 WDS 数据流位置/队列 |
| 随机增强 | 每次 source 出现有独立确定 seed，同一 resident 恢复可复现 | 依赖 worker 随机状态与出队时机，当前无数据流恢复保证 |

相同锚点和相同原始监督下，几何/归一化/padding 的模型表示可一致；但源采样分布、随机源、
内存、末帧规则、生产端 action 定义和 resume 语义并不等价。原生指令全保留与内部
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

## 读取性能与复杂度

1. VLA 历史 state 与未来监督 state 合并为一次查询，再按原始顺序拆开；重复 padding 索引保留。
2. 初始化以 row-group 的 episode 范围反查 episode 列表，从每文件 O(E×G) 改为
   O(G log E + M)，M 是实际 episode/group 交集数量；缺少 footer 统计时仍需覆盖全部候选。
3. row-group 首次进入有界缓存时建立 (episode_index, frame_index) 排序索引，成本 O(R log R)，int64 索引额外约 24R 字节。
   后续查询用 searchsorted 定位，约 O(K log R)，不再每次扫描该组全部 R 行。
   按 episode/frame 定位后再检查全局 index 和时间戳，避免漏掉 index 写错的重复帧。
   返回 Arrow Table，motion 通过 NumPy row views 取值，避免逐浮点数 to_pylist 再转回 NumPy。
   VLM 仅将选中的 QA 行转成 Python 字典。索引/时间一致性检查对 K 行向量化执行。
4. 视频每路缓存已 resize 的帧，顺序读取时保留中间帧，避免两路相机互相淘汰历史窗口。
   编码直接使用这些帧，省去压缩前完整窗口 stack 和重复 resize；只在单窗口内复用重复索引的
   编码结果，不设置跨样本 JPEG 共享池。出队后的 JPEG 解码、检查和增强顺序不变。
5. 读取阶段只做确定性处理，不再反复设置增强随机种子；每条样本的 seed 仅在出队预处理使用。

缓存位于 shuffle 前：worker 在 episode 内顺序读帧，shuffle 随机的是已经准备好的 JPEG 窗口。
全库有几千万行不等于 row-group 缓存无命中；实际命中率取决于 row-group 大小、episode 分配和
缓存容量，不能只根据总行数推断。

本机局部基准（非整训吞吐，绝对耗时受系统负载影响）：

| 场景 | 修改前 | 修改后 |
|---|---:|---:|
| 20 万行/group，38 帧低维窗口，缓存命中（组合键修正后复测） | 10.88 ms | 1.07 ms |
| 上述首次遍历的 row-group miss | 30 | 30 |
| 强制每次查询清空缓存，100 个 anchor：row-group 读取次数 | 200 | 100 |
| 双路 32×32 H.264，41 个 anchor、6 帧历史+1 帧未来：seek | 82 | 2 |
| 同上：实际解码帧数 | 15842 | 762 |

首次遍历新实现命中率为 99.5%，没有重复任何 anchor；强制冷读时旧版每 anchor 两次读取，
新版一次。基准使用合成 Parquet/视频，只说明读取机制的变化；实际数据上的耗时仍应单独测量。
回归对照覆盖 JPEG/NPY 字节、raw 窗口、正常样本、验证/normalizer、旧 checkpoint 续读、
多 worker 及 VLA/VLM 混合；另检验跨 row-group、乱序物理行、重复请求索引和重复数据行拒绝。

## 官方 LeRobot 库是否能复用

已核查 [PyPI lerobot 0.6.1](https://pypi.org/project/lerobot/0.6.1/) 发布包源码，未修改训练环境依赖。

| 官方能力 | 可复用部分 | 当前流程需要额外适配的部分 |
|---|---|---|
| LeRobotDataset / DatasetReader | 官方 v3 路径、按索引和 delta_timestamps 取窗口 | 本项目 next-state action、JPEG80 队列、独立 QA 源与消费边界恢复 |
| LeRobotDatasetMetadata | tasks、episodes、路径模板及格式兼容 | 默认加载 stats.json，episodes 经 HF 加载后再去掉 stats/*；本实现是在 Parquet 读取前投影列 |
| StreamingLeRobotDataset | HF Parquet 流式读取、lookback/lookahead、decoder cache | make_frame 先解码/变换再进 buffer；单次迭代耗尽后结束；未提供覆盖内部 buffer 的 state_dict/load_state_dict |
| VideoDecoderCache / decode_video_frames | TorchCodec decoder 缓存、PyAV/TorchCodec 解码 | 缓存后 resize/JPEG 编码和本项目帧序仍需适配；不能保证更换 decoder 后 JPEG 像素逐值不变 |
| dequantize_depth | 官方 12-bit 深度反量化 | 默认输出毫米，需显式选择米；还须按本数据规范将 code=0 还原为无效零 |

因此官方库可以作为后续替换底层 reader/decoder 的候选，但 StreamingLeRobotDataset 不是
当前无限 JPEG 队列与精确数据 resume 的直接替代品。本轮未引入新的 lerobot 依赖。
0.6.1 要求 Python>=3.12、Torch>=2.7、NumPy>=2.0，dataset extra 要求 PyArrow>=21、
PyAV>=15,<16；本机验证环境为 Torch 2.6、NumPy 1.26、PyArrow 16、PyAV 16.1，直接安装会
改变多项依赖。仓库训练环境的版本需另外核对，不能把本机验证环境视为正式训练环境。

源码入口：[streaming_dataset.py](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/datasets/streaming_dataset.py)、
[dataset_reader.py](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/datasets/dataset_reader.py)、
[dataset_metadata.py](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/datasets/dataset_metadata.py)、
[video_utils.py](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/datasets/video_utils.py)。

### 优化等价性复核

以优化前的工作区快照为基准，对照编码窗口、训练/验证/normalizer 数值、source/queue/RNG
checkpoint 字节以及 VLA/VLM/混合流多 worker 恢复。另做 800 组行查询对照，覆盖物理乱序、
重复 frame 但错误全局 index、坏时间戳、缺行及重复请求；接受/拒绝结果与旧实现一致。
二分索引使用 episode/frame 组合键，不以可能错误的 global index 代替旧版选行条件。
视频缓存保留原始尺寸信息，仅在请求该帧时验证，避免对未请求的中间帧新增失败条件。
缓存策略、占用和执行耗时属于有意的性能变化；它们不是样本数值等价性的比较对象。
