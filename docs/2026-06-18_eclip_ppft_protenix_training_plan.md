# eCLIP 弱监督 PPFT 训练 Protenix 实现计划

## 目标

用 eCLIP 的 protein/RNA 序列和 RNA 位点级 binding signal 微调 Protenix，使模型 rollout 出的 protein-RNA 结构在几何接触上解释实验信号。训练数据没有真实结构，因此训练阶段必须跳过 Protenix 现有的 diffusion、distogram、bond、PAE/pLDDT 等依赖真实坐标的 loss；但原模型中不依赖真实坐标、只依赖序列/掩码/预测结构自洽性的 loss 或正则项要保留。

核心监督链路：

```text
protein sequence + RNA sequence + cell_line/eCLIP signal
  -> Protenix featurizer
  -> Pairformer trunk
  -> differentiable diffusion rollout
  -> predicted atom coordinates
  -> RNA residue heavy atoms vs protein heavy atoms distance
  -> differentiable p_bind per RNA residue
  -> compare with eCLIP signal target
```

## 参考 BioEmu PPFT 的关键点

BioEmu 的 PPFT 不是用真实结构监督单步 denoising，而是对模型采样结果计算 property loss。参考实现位于：

- `/inspire/ssd/project/sais-bio/public/xiangwenkai/GITHUB/bioemu/src/bioemu/training/loss.py`
- `/inspire/ssd/project/sais-bio/public/xiangwenkai/GITHUB/bioemu/notebooks/ppft_example.ipynb`
- `/inspire/ssd/project/sais-bio/public/xiangwenkai/GITHUB/bioemu/notebooks/rollout.yaml`

需要借鉴的设计：

1. `calc_ppft_loss` 先 rollout 得到样本结构，再计算 property；本项目只保留这个“采样结构 -> property loss”的思想。
2. rollout 使用少步数近似采样，`rollout.yaml` 中是 `mid_t=0.786`、`N_rollout=7`、`record_grad_steps=[3,4,5]`，用少量带梯度步骤控制显存。
3. 最终 property loss 不依赖真实结构；BioEmu 示例是 foldedness，这里替换为 eCLIP binding profile。
4. BioEmu notebook 中的多副本均值估计和 cross-product estimator 不用于本实现，因为这里明确要求每条 eCLIP 样本只 rollout 一个结构，不做多个结构的平均。

## 现有 Protenix 可复用接口

Protenix 训练入口是 `runner/train.py` 的 `AF3Trainer`。现有训练流：

```text
get_dataloaders()
  -> model_forward()
  -> Protenix.main_train_loop()
  -> ProtenixLoss()
```

这个路径强依赖 `label_dict["coordinate"]` 和 `coordinate_mask`，不适合 eCLIP-only 数据直接复用。

推荐新增独立训练入口，而不是硬塞进 `AF3Trainer`：

```text
runner/train_eclip_ppft.py
protenix/data/eclip_ppft_dataset.py
protenix/model/eclip_binding_loss.py
configs/configs_eclip_ppft.py
```

可复用的 Protenix 组件：

- `SampleDictToFeatures`：从 inference JSON 风格的 protein/RNA 序列构造 atom/token feature。
- `make_dummy_feature`：无 MSA/template 时填充 dummy feature。
- `Protenix.get_pairformer_output`：得到 `s_inputs, s, z`。
- `Protenix.sample_diffusion` / `sample_diffusion`：从 trunk 表征 rollout 坐标。
- `input_feature_dict["atom_to_token_idx"]`、`is_protein`、`is_rna`、`ref_element`：用于从 atom 坐标映射到 protein/RNA residue。

parnet 已有一个 frozen Protenix eCLIP head 框架，可复用数据读取和序列映射逻辑：

- `/inspire/ssd/project/sais-bio/public/xiangwenkai/GITHUB/parnet/parnet/eclip_head/dataset.py`
- `/inspire/ssd/project/sais-bio/public/xiangwenkai/GITHUB/parnet/parnet/eclip_head/protenix_adapter.py`
- protein symbol 到序列映射：`/inspire/ssd/project/sais-bio/public/xiangwenkai/GITHUB/parnet/parnet/assets/ENCODE.protein_symbol2sequence.uniprot.tsv`

## eCLIP 数据格式

数据目录：

```text
/inspire/ssd/project/sais-bio/public/xiangwenkai/GITHUB/parnet/data
```

抽样确认 parquet schema：

```text
rna_seq: string
protein_symbol: string
cell_line: string
signal_vector: list<double>
```

`signal_vector` 长度与 `rna_seq` 一致，当前样本常见长度为 600。`selected_experiments.tsv` 包含 223 个 `(protein_symbol, cell_line)` 任务。训练样本需要通过 `ENCODE.protein_symbol2sequence.uniprot.tsv` 补齐 protein sequence。

第一版建议先过滤：

- protein sequence 缺失的样本。
- RNA 长度和 signal 长度不一致且无法修正的样本。
- protein 长度超过显存预算的样本，例如先设 `max_protein_length=1200`。
- 全零 signal 样本按比例下采样，否则 loss 会被 negative 主导。

## 模型 forward 方案

新增 `EclipPPFTTrainer`，不要直接调用完整 `ProtenixLoss`。训练时用一个 eCLIP 专用 loss aggregator：

```text
total_loss =
  eclip_binding_loss
  + kept_coordinate_free_original_losses
  + optional_prediction_geometry_regularizers
```

其中 `kept_coordinate_free_original_losses` 是从原 Protenix loss/正则项里筛出来的、不需要真实坐标的部分；凡是需要 `label_dict["coordinate"]` 或真实 atom-atom distance 的项必须跳过。

每个 batch 推荐从 `batch_size=1` 起步，因为 Protenix 原训练也默认复杂体系单样本 batch。每条样本只生成一个 rollout 结构，`N_sample` 固定为 1。

流程：

```python
features = featurize_eclip_sample(sample)
features = model.relative_position_encoding.generate_relp(features)
features = update_input_feature_dict(features)

s_inputs, s, z = model.get_pairformer_output(
    input_feature_dict=features,
    N_cycle=config.eclip_ppft.n_cycle,
    inplace_safe=False,
    chunk_size=config.eclip_ppft.chunk_size,
)

coords = differentiable_rollout(
    model=model,
    input_feature_dict=features,
    s_inputs=s_inputs,
    s=s,
    z=z,
    N_sample=1,
    N_step=config.eclip_ppft.n_rollout_steps,
    grad_steps=config.eclip_ppft.record_grad_steps,
)

p_bind = compute_rna_binding_probability(
    coords=coords,
    input_feature_dict=features,
    cutoff=5.0,
    temperature=0.5,
    softmin_beta=4.0,
)

loss = eclip_binding_loss(p_bind, target_signal)
```

### Differentiable rollout

Protenix 当前 `sample_diffusion` 没有 BioEmu 的 `record_grad_steps` 参数。实现分两步：

1. 第一版直接用短 rollout 全程带梯度，`N_step=4-8`、`N_sample=1`，验证 loss/梯度链路。
2. 第二版 fork `protenix/model/generator.py::sample_diffusion` 为 `sample_diffusion_ppft`，增加：

```python
record_grad_steps: set[int]
detach_unrecorded_steps: bool = True
```

每个 denoising step 外层按 step index 判断：

```python
ctx = nullcontext() if step_idx in record_grad_steps else torch.no_grad()
with ctx:
    x_denoised = denoise_net(...)
if step_idx not in record_grad_steps:
    x_l = x_l.detach()
```

最后一个 clean coordinate 计算必须带梯度，避免 property loss 到模型参数断开。这个设计对应 BioEmu `_rollout()` 的“少步 rollout + 中间少数 step 记录梯度”思想。

## Binding score 的连续化

定义 RNA residue `r` 的 heavy atom 集合 `A_r`，protein heavy atom 集合 `P`。先排除 hydrogen；Protenix reference features 里通常只建 heavy atoms，但仍建议显式根据 element mask 排除 H。

对单个 rollout 结构：

```text
d_r = softmin_{a in A_r, p in P} ||x_a - x_p||
p_bind_r = sigmoid((cutoff - d_r) / temperature)
```

softmin 可实现为：

```python
softmin(d, beta) = -logsumexp(-beta * d) / beta
```

为了数值稳定和显存可控：

- protein atom 数很多时按 protein atom chunk 计算 `logsumexp`。
- RNA residue 以 token 为单位聚合；标准 RNA nucleotide 在 Protenix 中是单 token，modified RNA 可能 token 化成多 atom token，第一版先用 `is_rna` atom mask 和 `atom_to_token_idx` 聚合。
- `p_bind` 输出形状为 `[N_rna_token]`。实现内部可以保留 `[1, N_rna_token]` 的维度以复用 Protenix diffusion 输出，但 loss 不能对多个结构求平均。

可选替代形式：

```text
p_bind_r = 1 - prod_{a,p}(1 - sigmoid((cutoff - d_ap) / temperature))
```

这个更接近“任意 atom pair 接触即 binding”，但在 atom pair 多时容易饱和。第一版建议 softmin。

## Target 处理与 loss

eCLIP signal 不是严格二分类标签，建议同时支持三种 target 模式。

### 模式 A：profile regression

适合信号 shape 学习：

```text
target_profile = normalize(log1p(signal_vector))
pred_profile = normalize(p_bind)
loss_profile = KL(target_profile || pred_profile)
```

对全零 signal 样本，跳过 profile KL，只使用 negative/background loss。

### 模式 B：soft binding MSE

适合保留绝对强度：

```text
target_bind = clamp(log1p(signal) / per-protein q95, 0, 1)
loss_mse = mean((p_bind - target_bind)^2)
```

`q95` 需要按 `protein_symbol` 或 `(protein_symbol, cell_line)` 预先统计，避免不同 RBP 的 eCLIP scale 不可比。

### 模式 C：log-odds / BCE

适合二值接触监督：

```text
target_binary = signal >= protein_specific_threshold
logit_bind = logit(clamp(p_bind, eps, 1-eps))
loss_bce = BCEWithLogits(logit_bind, target_binary, pos_weight=...)
```

第一版推荐：

```text
loss = profile_weight * profile_BCE + point_weight * SmoothL1(p_bind, normalize_per_sample(log1p(signal)))
```

其中 profile target 使用 `signal > 0` 的二值标签，阳性位置默认权重为 5。

## 单 rollout 约束

本任务不做多结构采样平均。每条 eCLIP 样本的训练路径固定为：

```text
one protein/RNA sample
  -> one differentiable rollout structure
  -> one p_bind vector
  -> one eCLIP binding loss
```

因此代码要强约束：

- rollout sample 数固定为 1，不作为训练配置暴露。
- 不实现 BioEmu 的 `n_replications` 数据复制。
- 不实现 cross-product estimator。
- validation 也按单结构预测计算 profile Pearson/top-k overlap，不能用多 seed ensemble 提升指标。

## 原模型 loss 保留规则

不要给 eCLIP 样本构造假的 `label_dict["coordinate"]` 来走完整 `ProtenixLoss`，否则 MSE、distogram、bond、PAE/PDE/pLDDT 会学习 reference conformer 或零坐标，方向是错的。

实现上要把原 loss 按是否需要真实坐标拆开：

```text
requires_true_coordinate = true:
  diffusion MSE
  smooth LDDT
  distogram CE
  bond loss using true distances
  pLDDT / PDE / PAE confidence supervision

requires_true_coordinate = false:
  experimentally resolved loss, if a meaningful coordinate_mask target is available
  prediction-only geometry regularizers, if they are already present or added without changing model parameters
  any future original Protenix loss whose inputs are only sequence features, masks, logits, or predicted coordinates
```

当前 Protenix 主分支里多数训练 loss 都需要真实坐标。对 eCLIP-only batch 的默认策略应是：

```text
enabled:
  eclip_binding_loss

disabled:
  alpha_diffusion
  smooth_lddt
  bond loss that compares to label coordinates
  alpha_distogram
  plddt/pde/pae supervised confidence losses
```

如果后续确认某个原模型 loss 在当前代码分支中完全不需要真实坐标，再单独加入 trainer，并加单测确认无 `label_dict["coordinate"]` 输入也能运行。当前实现只保留 eCLIP signal loss，避免混入不明确的正则项。

混合结构监督时按 batch type 分流：

```text
if batch["supervision_type"] == "structure":
    use existing AF3Trainer loss
elif batch["supervision_type"] == "eclip_ppft":
    use eclip_binding_loss + coordinate_free_original_losses
```

混合训练是推荐的长期方案，因为 BioEmu PPFT notebook 也提醒：只用 property fine-tuning 容易损伤模型整体采样质量，应穿插标准 denoising score-matching/结构训练。

## Checkpoint 兼容性要求

训练后的 Protenix 主模型必须能被原来的 Protenix 模型类加载，不能和原模型冲突。这里允许新增 eCLIP signal loss 相关模块，类似 BioEmu PPFT 中的 property/loss/potential 组件；但这些组件属于训练期外部模块，不应改变 Protenix 主模型结构。

具体约束：

1. 可以新增 `EclipSignalLoss` / `EclipBindingScorer` / target normalization / rollout wrapper。
2. 这些新增模块不挂到 `Protenix` 主模型下面，不写入 `checkpoint["model"]`。
3. `checkpoint["model"]` 只包含 Protenix 原有参数；key 集合必须和原 checkpoint 一致。
4. eCLIP 训练可以更新 Protenix 原有参数的值，也可以选择保存 signal loss 模块自己的 sidecar state。
5. 如果 signal loss 模块有可训练参数，例如可学习温度、scale、bias，它们只能保存在独立 key 下，例如 `checkpoint["eclip_signal_loss"]`，不能混进 `checkpoint["model"]`。
6. 原 Protenix inference / fine-tune 代码加载时只读取 `checkpoint["model"]`，不需要知道 eCLIP signal loss 模块存在。
7. checkpoint 保存格式兼容 `runner/train.py`，并允许额外 sidecar key：

```python
torch.save(
    {
        "model": model.state_dict(),
        "eclip_signal_loss": signal_loss.state_dict(),  # optional sidecar
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "step": step,
    },
    path,
)
```

8. DDP 训练保存时可以带 `module.` 前缀，但加载逻辑要和 `runner/train.py::try_load_checkpoint()` 一致；单卡原模型加载时能自动去掉 `module.`。
9. 新增配置只影响 eCLIP 训练脚本，不要求 Protenix inference 侧新增字段。

验收测试：

```python
base_model = Protenix(base_configs)
ckpt = torch.load(eclip_finetuned_ckpt, map_location="cpu", weights_only=False)
missing, unexpected = base_model.load_state_dict(ckpt["model"], strict=True)
assert not missing
assert not unexpected
```

如果未来确实需要新增 eCLIP 专用模块，只能另存 sidecar checkpoint，不能写入 `"model"` state_dict；否则不满足“原来的模型能够加载 eCLIP 训练好的模型”。

## 建议训练的参数模块

默认不要全量微调。eCLIP 只有 binding signal，没有真实结构，梯度只来自 rollout 后的接触几何；最直接、风险最低的参数是 diffusion 采样分支，其次才是 Pairformer 的最后几层 protein-RNA 表征。

### 推荐默认训练集合

第一版实际训练建议解冻：

```text
Protenix 主模型:
  diffusion_module.diffusion_conditioning
  diffusion_module.diffusion_transformer.blocks.{last_4_to_8}
  diffusion_module.atom_attention_decoder
  diffusion_module.layernorm_s
  diffusion_module.linear_no_bias_s
  diffusion_module.layernorm_a

训练期 sidecar:
  eclip_signal_loss / eclip_binding_scorer 中少量可学习标定参数（可选）
```

理由：

- `diffusion_module` 直接决定 rollout 坐标，binding loss 的梯度路径最短。
- `diffusion_conditioning` 控制 trunk `s/z`、noise level 和 diffusion 表征的融合，适合用弱监督调整采样方向。
- `diffusion_transformer` 是结构生成主干，建议先只开最后 4 到 8 个 block，避免弱监督破坏全局结构能力。
- `atom_attention_decoder` 直接输出 atom 坐标更新，对接触距离 loss 最敏感。
- signal loss 模块若有温度、scale、bias 等可学习参数，只作为 sidecar 保存，不进入 `checkpoint["model"]`。

### 第二阶段可选解冻

如果 Stage 0/1 能稳定下降，但 protein-RNA 接口位置仍学不动，再逐步解冻：

```text
pairformer_stack.blocks.{last_4_to_8}
linear_no_bias_z_cycle
linear_no_bias_s
```

理由：

- Pairformer 的 `s/z` 决定 protein-RNA pair 表征，后几层对界面定位有帮助。
- 只开最后几层可以减少灾难性遗忘。
- recycling 的 `linear_no_bias_z_cycle` / `linear_no_bias_s` 是主模型已有参数，允许更新，但学习率应低于 diffusion。

### 不建议在 eCLIP-only 阶段训练

```text
input_embedder
relative_position_encoding
template_embedder
msa_module
constraint_embedder
distogram_head
confidence_head
pairformer_stack 全部 block
```

理由：

- input/MSA/template/relative encoding 是基础表示，eCLIP 弱监督不足以安全重写。
- `distogram_head`、`confidence_head` 的原监督依赖真实结构或真实误差标签；eCLIP-only 不应训练这些 head。
- 全量 Pairformer 微调容易让模型为了贴合 signal 牺牲通用 folding 能力。

### 建议 freeze pattern

若用 substring 方式筛选参数，推荐从以下 pattern 开始：

```python
trainable_patterns_stage1 = [
    "diffusion_module.diffusion_conditioning",
    "diffusion_module.diffusion_transformer.blocks.20",
    "diffusion_module.diffusion_transformer.blocks.21",
    "diffusion_module.diffusion_transformer.blocks.22",
    "diffusion_module.diffusion_transformer.blocks.23",
    "diffusion_module.atom_attention_decoder",
    "diffusion_module.layernorm_s",
    "diffusion_module.linear_no_bias_s",
    "diffusion_module.layernorm_a",
]

trainable_patterns_stage2_extra = [
    "pairformer_stack.blocks.44",
    "pairformer_stack.blocks.45",
    "pairformer_stack.blocks.46",
    "pairformer_stack.blocks.47",
    "linear_no_bias_z_cycle",
    "linear_no_bias_s",
]
```

这里假设默认配置为 diffusion transformer 24 blocks、Pairformer 48 blocks。实现时不要硬编码 block index，应从 `len(module.blocks)` 动态取最后 `k` 层。

### 学习率建议

```text
eclip signal loss sidecar params: 1e-4
diffusion decoder / last transformer blocks: 1e-5
diffusion conditioning: 5e-6 to 1e-5
last Pairformer blocks: 1e-6 to 3e-6
recycling linear layers: 1e-6
```

如果只做第一轮 smoke / overfit，建议先固定 signal loss 的温度和 scale，不训练 sidecar 参数，避免 loss 标定参数吸收梯度、Protenix 本体学不到东西。

## 参数冻结与优化策略

建议分阶段：

### Stage 0：只验证 loss 梯度

- 加载 `protenix_base_default_v1.0.0`。
- 冻结 Pairformer/trunk，只训练 diffusion module 的少量参数，或只开最后几个 diffusion blocks。
- `N_step=2-4`、`N_sample=1`、`N_cycle=1`。
- 用 8-32 个样本做 overfit，确认 loss 能下降、`p_bind` 与 signal peak 对齐。

### Stage 1：轻量 PPFT

- 解冻 diffusion module + selected Pairformer blocks。
- `N_step=4-8`、`N_sample=1`。
- 使用 `record_grad_steps` 限制显存。
- 每隔固定 step 用 full inference `N_step=20/50` 评估。

### Stage 2：混合训练

- eCLIP batch 和结构 batch 按比例混合，例如 `eclip:structure = 1:3`。
- 结构 batch 继续走原 Protenix loss。
- eCLIP batch 走 binding loss + coordinate-free original losses。
- 目标是保留结构物理合理性，同时增强 RNA binding signal 一致性。

## 数据与 featurization 实现细节

新增 `EclipPPFTDataset`：

- 流式读取 parquet，使用 `pq.ParquetFile.iter_batches()`，不要一次读完整 shard。
- 读取列：`rna_seq, protein_symbol, cell_line, signal_vector`。
- 用 `ENCODE.protein_symbol2sequence.uniprot.tsv` 补 protein sequence。
- 构造 Protenix inference JSON：

```json
{
  "name": "UPF1_HepG2_xxx",
  "sequences": [
    {"proteinChain": {"sequence": "...", "count": 1}},
    {"rnaSequence": {"sequence": "...", "count": 1}}
  ]
}
```

注意 eCLIP 数据中 RNA 是 DNA alphabet `T`，构造 Protenix `rnaSequence` 时要转成 `U`。

缓存策略：

- featurization 比较重，可选 `feature_cache_dir`，key 包含 model_name、protein sequence、rna_seq、MSA path。
- 不要缓存 rollout 坐标，因为要反向传播且模型会更新。

## 指标与验证

训练日志至少记录：

- `loss/profile_kl`
- `loss/point_loss`
- `p_bind_mean`
- `target_signal_total`
- `pearson(signal_pred, target_signal)`
- `topk_overlap`: target top-k RNA 位点和 predicted top-k 位点重叠率
- `contact_count`: `p_bind > 0.5` 的 RNA token 数
- 采样结构健康度：protein/RNA atom clash 粗略统计、RNA/protein radius of gyration

验证分两类：

1. eCLIP validation parquet：看 profile Pearson、top-k overlap、loss。
2. 少量已知 protein-RNA PDB：不用训练结构 loss，只用作 sanity check，比较真实 contact map 与模型 `p_bind`。

## 文件改动清单

建议按以下顺序实现：

1. `protenix/model/eclip_binding.py`
   - `compute_soft_binding_score`
   - `eclip_binding_loss`
   - chunked pairwise distance utilities

2. `protenix/data/eclip_ppft_dataset.py`
   - parquet streaming dataset
   - protein sequence mapping
   - target normalization/statistics

3. `protenix/model/generator_ppft.py` 或扩展 `generator.py`
   - `sample_diffusion_ppft`
   - `record_grad_steps`
   - short rollout config

4. `runner/train_eclip_ppft.py`
   - 独立 trainer
   - checkpoint / wandb / eval
   - 冻结和解冻参数配置

5. `configs/configs_eclip_ppft.py`
   - data path、loss weights、rollout 参数、freeze patterns

6. `tests/test_eclip_binding_loss.py`
   - 构造小坐标，验证 5A 内 `p_bind` 高、远距离低、梯度非零。

7. `tests/test_eclip_ppft_dataset.py`
   - 用 `_smoke_export` 或小 parquet 验证 sample schema 和 RNA/signal 对齐。

8. `tests/test_eclip_checkpoint_compat.py`
   - eCLIP fine-tuned checkpoint 的 `"model"` state_dict 可被原始 `Protenix` `strict=True` 加载。

## 初始配置建议

```yaml
eclip_ppft:
  data_dir: /inspire/ssd/project/sais-bio/public/xiangwenkai/GITHUB/parnet/data
  protein_sequence_tsv: /inspire/ssd/project/sais-bio/public/xiangwenkai/GITHUB/parnet/parnet/assets/ENCODE.protein_symbol2sequence.uniprot.tsv
  max_rna_length: 600
  max_protein_length: 1200
  batch_size: 1
  n_cycle: 1
  n_rollout_steps: 4
  record_grad_steps: [2, 3, 4]
  cutoff_angstrom: 5.0
  contact_temperature: 0.5
  softmin_beta: 4.0
  profile_weight: 1.0
  positive_weight: 5.0
  point_weight: 2.0
  lr: 1.0e-5
  dtype: bf16
```

## 风险与缓解

1. eCLIP signal 是体内/转录组窗口信号，不一定对应单个稳定复合物构象。
   - 先用 profile/top-k 指标，不要期待 absolute signal 完全拟合。

2. 仅用 binding loss 可能诱导塌缩：protein 整体贴到 RNA 上。
   - 先监控 `p_bind_mean`、top-k overlap 和 profile AUPRC。
   - 长期混合结构监督或加入明确验证过、不依赖真实坐标的原模型 loss。

3. Protenix short rollout 梯度显存很大。
   - 从 `N_step=2-4, N_sample=1` 开始。
   - 加 `record_grad_steps`。
   - 冻结大部分 trunk。

4. protein_symbol 可能对应 isoform，不一定是实验 RBP 实际构建。
   - 先使用 parnet 的 UniProt 映射；后续按 ENCODE metadata 校正 isoform。

5. 全零/弱信号样本过多。
   - 使用 positive-enriched 采样，或按 `signal_total` 分层采样。

## 里程碑

1. M0：完成 binding score 单元测试，确认坐标到 `p_bind` 可反传。
2. M1：完成 eCLIP dataset smoke run，单 batch 能 featurize protein/RNA 并 rollout。
3. M2：32 个样本 overfit，loss 和 top-k overlap 明显改善。
4. M3：validation 跑通，保存 checkpoint 和指标。
5. M4：加入单结构 rollout 的 `record_grad_steps`，不加入多结构平均。
6. M5：混合结构监督训练，评估结构质量和 eCLIP 指标的 tradeoff。
7. M6：严格加载兼容性测试通过，确认原 Protenix 类能 `strict=True` 加载 eCLIP fine-tuned checkpoint。
