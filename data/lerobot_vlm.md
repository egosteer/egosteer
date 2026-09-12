# LeRobot / Parquet VLM 数据

`src.dataset.lerobot.vlm_dataset.VLMLeRobotDataset` 读取独立的图文问答数据源。
问答内容、候选评分和图像处理沿用 `VLMWdsDataset`，改变的是磁盘读取和可恢复采样。
不会从机器人任务指令自动生成问答。

## 字段对应

一条原 WDS sample 对应一条 Parquet frame row，多个图片仍属于同一条样本。

| WDS 字段 | LeRobot / Parquet 中的位置 |
|---|---|
| `meta.json.texts` | 帧列 `texts`：`[{"user": question, "assistant": answer}, ...]` |
| `meta.json.formatting_ratings` | 帧列 `formatting_ratings`：每个候选一项，null 当作 0 |
| `meta.json.visual_dependency_ratings` | 帧列 `visual_dependency_ratings`：同上 |
| `meta.json.relevance_ratings` | 帧列 `relevance_ratings`：同上 |
| `image_0.jpg`、`image_1.jpg`… | `info.json.features` 声明的 image/video feature，例如 `image_0`、`image_1` |
| `meta.json.source/dataset_name/sample_idx` | 可选同名帧列，用于调试定位 |

四个 QA 字段支持 Arrow 原生列表/struct，也支持 JSON 字符串。例如，用官方 feature 类型
存储变长候选列表时，可以声明 `dtype: string, shape: [1]`，将列表 JSON 编码进该列。
若转换器保留整个 `meta.json` 为一个 struct/JSON 列，可设 `metadata_key: meta.json`；
该列内仍须包含上面四个同名字段。

候选数量与三组评分长度必须一致。多候选时沿用 WDS：

```text
score = formatting * weights[0]
      + visual_dependency * weights[1]
      + relevance * weights[2]
```

默认 weights 为 `[0.5, 0.5, 0.5]`，取最高分；并列取列表中第一项。
这里不是随机抽一个回答，也不是把所有候选拼接到同一段文本。

## LeRobot 布局

```text
<vlm_root>/
  meta/info.json
  meta/episodes/chunk-*/file-*.parquet
  data/chunk-*/file-*.parquet
  images/...                         # image feature 使用路径时
  videos/<feature>/chunk-*/file-*.mp4 # video feature 时
```

按 `info.json.splits` 的 episode 区间选择 train/val。metadata 需要标准的 episode_index、
length、dataset_from/to_index、data 文件定位；video feature 还需要各自的
chunk/file/from_timestamp/to_timestamp。帧行需要 episode_index、frame_index、index、timestamp。
tasks.parquet 等标准产物可以保留，reader 不用任务词表替代 QA 文本。

VLM 不要求 74D state/action、相机标定、instruction_num、reward 或接管字段。
每一帧都是完整 QA 样本，包括 episode 的最后一帧；单帧 episode 也有效。
episode 只是存储和 worker 分配单位，不会生成历史窗口、next state 或 future target。

图片 feature 使用 LeRobot/HF 的 `{"bytes": ..., "path": ...}` 形式，或直接的图片路径/bytes。
路径相对 root，绝对本地路径也可读取。可选图片为空时跳过该视图，至少需要一张有效图片。
video feature 则按 episode 的独立文件偏移读取当前帧，仍交给模型作为 image，保持 WDS VLM
语义。默认根据 features 中的 RGB image/video 键排序发现图片；可用 `image_keys` 明确指定顺序。
同一样本的多张图片需保持原 WDS adapter 可接受的统一尺寸。

## 混合训练

保留原 VLA-only 配置；新增可选实验 `experiment/egosteer_lerobot_vlm`：

```bash
torchrun --standalone --nproc_per_node=1 train.py \
  --config-name=experiment/egosteer_lerobot_vlm \
  lerobot_root=/data/EgoSteer-RealWorld \
  vlm_lerobot_root=/data/VLM-LeRobot
```

默认 `batch_size=8, vla_ratio=0.75`：每个 worker 依次输出 6 条 VLA、2 条 VLM，组成一个 batch。
计算规则与 WDS 相同：`ceil(batch_size * vla_ratio)` 条 VLA，其余为 VLM。
两条流分别顺序遍历、shuffle，各自的预热和容量均为 4096；VLM 默认 `drop_ratio=0`。
这是 VLA/VLM 的固定配额混合，不是多个 VLA 数据源之间的 RandomMix。

VLM sample 的 actions/states 补零，n_states/n_actions=0，actions_valid_mask 全 false，
view_mask 全 false，n_future_frames=0。collator 只给 VLM assistant 的有效回答 token 设置
文本 labels；VLA 文本 labels 保持 -100。动作与 future-frame 监督不会应用到 VLM 行。
normalizer 仍仅由 VLA train 数据拟合；VLM 不使用另一套机器人 normalizer。

验证先读 VLA val，再读 VLM val，有限且无训练增强。若 VLM 没有验证 split，设
`vlm_lerobot_val_split=null`，只验证 VLA。两个训练数据源都需要有足够 episode 分配给各 rank/worker。

## Resume

使用相同的 `training.resume_checkpoint_path`。混合状态包含：

- VLA、VLM 各自的 source cursor、drop/shuffle RNG、resident queue 和 sample seed。
- 两条流各自的已交付样本数，以及共同的已消费 batch 边界。
- 固定配额、数据/评分/图像配置指纹及 DataLoader 拓扑。

一个混合 batch 完成后才共同推进两条流的消费状态。预取领先不进入 checkpoint；恢复时
两条流使用相同的逻辑 worker 轮转，从下一组 VLA/VLM 配额继续。改变混合比例或来源配置会
明确拒绝恢复，不会继续使用不匹配的队列。已有纯 VLA checkpoint 的格式保留，但不能当作
已包含 VLM 进度的混合 checkpoint 使用；新增 VLM 阶段可以通过 finetune_checkpoint_path 起步。

验证覆盖原生 image/视频 frame、嵌套/JSON QA、WDS 评分与像素对照、混合 labels/mask，
以及包含预取和 worker 轮转的两条流恢复。模型 tokenizer/完整 GPU 训练仍需真实数据环境验证。
