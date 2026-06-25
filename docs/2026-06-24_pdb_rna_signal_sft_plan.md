# PDB protein-RNA binding signal SFT 实现评估

## 结论

这两个任务都可以实现，而且建议分两层做：

1. 先把 protein-RNA 结构中抽取出的 RNA binding signal 写入现有 Protenix bioassembly pkl。
2. 再把当前 eCLIP SFT 中的 signal loss 抽成通用模块，让它既能用于 eCLIP parquet 数据，也能用于 PDB 结构数据。

PDB 数据训练和 eCLIP 数据训练的关键差异是：eCLIP 只有序列和 signal，因此只能用 signal loss；PDB 数据同时有真实结构和结构派生的 signal，因此应保留原 Protenix 结构监督 loss，再额外加 RNA binding signal loss。

## 当前代码状态

`configs/configs_data.py` 中的 `train_rna_before202606` 和 `test_rna_before202606` 已经走原 Protenix 数据管线：

- `bioassembly_dict_dir`: `/inspire/ssd/project/sais-bio/public/xiangwenkai/GITHUB/Protenix_v1/data/train` 或 `test`
- `indices_fpath`: `protein_rna_train_recent.csv` 或 `protein_rna_test.csv`
- 每个样本 pkl 是 `bioassembly_dict`，包含 `atom_array`、`token_array`、MSA/template features 和结构 metadata。

这些 pkl 当前不包含 RNA binding signal。抽样检查过，pkl 顶层 key 只有：

```text
assembly_id, atom_array, entity_poly_type, msa_features, num_assembly_polymer_chains,
num_prot_chains, num_tokens, pdb_id, release_date, resolution, sequences,
template_features, token_array
```

`/inspire/ssd/project/sais-bio/public/xiangwenkai/GITHUB/parnet/parnet/data/structure_rna_signal.py` 已经实现了从 PDB/mmCIF 结构抽取 RNA 位点级 contact signal 的逻辑，包括：

- protein/RNA chain 自动识别
- RNA residue 到 full polymer sequence 的映射
- heavy atom contact
- base/sugar/phosphate 分组权重
- `signal_vector`
- `profile_label`
- `signal_total`、`signal_peak` 等统计量

但这个脚本输出的是 eCLIP-like parquet/tsv/jsonl 行，不是直接写 Protenix bioassembly pkl。因此需要一个 Protenix 侧的适配脚本。

## 任务 1：把结构 binding signal 整合进 PDB pkl

### 可行性

可行。推荐不要只生成外部 CSV，也不要只按 RNA sequence 存 signal。原因是 Protenix 训练时会 crop，crop 后 token 顺序和长度会变化，signal 必须和 crop 后的 RNA token 严格对齐。

最稳的方式是把 signal 写成 `token_array` 的 token-level annotation。`TokenArray` 支持 `set_annotation()`，crop 时 `Token` 会被 deepcopy，annotation 会自然保留。这样 crop 后可以直接从 `cropped_token_array.get_annotation("rna_binding_signal")` 取出与当前 crop 对齐的 signal。

### 推荐写入格式

在每个 bioassembly pkl 中增加两类信息。

第一类是 token annotation：

```python
token_array.set_annotation("rna_binding_signal", token_signal.tolist())
token_array.set_annotation("rna_binding_signal_mask", token_signal_mask.tolist())
token_array.set_annotation("rna_binding_resolved_mask", token_resolved_mask.tolist())
```

其中：

- `token_signal`: shape `[N_token]`，非 RNA token 为 0；RNA token 为结构 contact signal。
- `token_signal_mask`: shape `[N_token]`，RNA token 且 signal 可定义的位置为 1，其余为 0。
- `token_resolved_mask`: shape `[N_token]`，RNA residue 在结构中有解析坐标则为 1。

第二类是 pkl 顶层 metadata：

```python
bioassembly_dict["rna_binding_signal_meta"] = {
    "source": "structure_contact",
    "contact_radius": 6.0,
    "contact_midpoint": 4.0,
    "contact_temperature": 1.0,
    "base_weight": 1.0,
    "sugar_weight": 0.7,
    "phosphate_weight": 0.4,
    "version": 1,
}
```

### 推荐实现脚本

新增脚本：

```text
scripts/add_structure_rna_signal_to_protenix_pkl.py
```

输入参数建议。这里按 `bioassembly-dir` 下的 `*.pkl.gz` 文件名逐个处理，不依赖 indices CSV：

```bash
python scripts/add_structure_rna_signal_to_protenix_pkl.py \
  --bioassembly-dir /inspire/ssd/project/sais-bio/public/xiangwenkai/GITHUB/Protenix_v1/data/train \
  --output-dir /inspire/ssd/project/sais-bio/public/xiangwenkai/GITHUB/Protenix_v1/data/train_signal \
  --contact-radius 8.0 \
  --num-workers 28
```

对 test 集同理：

```bash
python scripts/add_structure_rna_signal_to_protenix_pkl.py \
  --bioassembly-dir /inspire/ssd/project/sais-bio/public/xiangwenkai/GITHUB/Protenix_v1/data/test \
  --output-dir /inspire/ssd/project/sais-bio/public/xiangwenkai/GITHUB/Protenix_v1/data/test_signal \
  --contact-radius 8.0 \
  --num-workers 28
```

默认建议写到新目录，不要原地覆盖。验证通过后再把 `configs_data.py` 的 `bioassembly_dict_dir` 指向新目录，或者用参数覆盖。

脚本默认行为：

1. 扫描 `--bioassembly-dir/*.pkl.gz`。
2. 逐个读取 Protenix bioassembly pkl。
3. 根据 pkl 内部 `atom_array` / `token_array` 计算 RNA token 的 protein contact signal。
4. 将 signal annotation 写回 pkl。
5. 输出到 `--output-dir/{pdb_id}.pkl.gz`，保持原文件名。
6. 额外写一个 summary，例如 `structure_rna_signal_pkl_summary.json`，记录处理数量、无 protein/RNA 的样本数、全零 signal 样本数、失败样本等。

`--contact-radius` 保留为显式参数，用于定义 protein heavy atom 与 RNA heavy atom 的最大 contact 距离。默认可以设为 `6.0`，如果希望更接近二值 contact 标签，也可以后续实验比较 `5.0` 和 `6.0`。

`--indices-csv` 不作为必需参数。最多作为可选参数使用，例如只统计当前 train/test indices 覆盖了哪些 PDB，或者生成过滤后的训练 CSV；但“给 train/test 目录下所有 pkl 补 signal”这个任务本身不需要它。

### 信号计算逻辑

脚本可以复用 `structure_rna_signal.py` 中的核心规则，但不建议直接复用其 parquet 输出作为训练输入。更合适的是在 Protenix pkl 的 `atom_array` / `token_array` 上直接计算：

1. 从 `atom_array.is_protein` 选 protein heavy atoms。
2. 从 `atom_array.is_rna` 选 RNA atoms。
3. 用 `token_array.get_annotation("atom_indices")` 建立 token 到 atoms 的关系。
4. 对每个 RNA token，取其 RNA heavy atoms，与所有 protein heavy atoms 计算 contact。
5. 得到每个 RNA token 的 `signal`。
6. 对非 RNA token 写 0，mask 写 0。

这样可以避免 PDB/mmCIF 解析链 ID、residue 编号、polymer scheme 与 Protenix pkl token 顺序不一致的问题。

### 训练 indices 需要注意

数据加工阶段会给目录下所有 pkl 补 signal，不需要根据 indices 过滤。

但训练阶段仍然需要注意：当前 `protein_rna_train_recent.csv` 不一定每一行都是 protein-RNA interface，里面可能有 `intra_rna`、`intra_prot` 等行。对于 signal 训练，建议至少提供一个只包含 RNA-protein interface 的 indices，例如：

```text
eval_type == "rna_prot"
mol_type_group == "nuc_prot"
sub_mol_2_type == "rna" 或 sub_mol_1_type == "rna"
```

否则 crop 可能采到纯 protein 或纯 RNA 区域，signal loss 会因为没有有效 RNA/protein token 被跳过，或者产生大量全零 target。

## 任务 2：PDB 数据上同时使用 signal 监督和真实结构监督

### 可行性

可行。不要把 PDB 结构训练硬塞进当前 eCLIP iterable parquet loader。PDB 数据已经有成熟的原 Protenix loader 和原结构 loss，应该复用：

- `protenix.data.pipeline.dataset.get_datasets`
- `BaseSingleDataset`
- `WeightedMultiDataset`
- `runner/train.py` 中的原 `model_forward`
- `ProtenixLoss`

当前 eCLIP SFT runner 的 signal loss 和 distogram contact 计算可以复用，但数据入口和主 loss 计算方式需要区分。

### 推荐训练目标

PDB signal SFT 的总 loss：

```text
total_loss =
    original_protenix_structure_loss
  + signal_loss_weight * structure_signal_loss
```

其中 `original_protenix_structure_loss` 保留原 Protenix loss，包括 diffusion、distogram、bond、confidence 等由配置权重控制的结构监督项。

`structure_signal_loss` 使用当前 eCLIP v2 的逻辑：

```text
distogram_logits = pred_dict["distogram"]
contact_probs = compute_contact_prob(distogram_logits, threshold)
p_bind_r = max_p p_contact[r, p]
signal_loss = profile BCE(p_bind, target_binary) + point loss(p_bind, normalized_log1p(signal))
```

这里 target 不是 eCLIP parquet 的 `signal_vector`，而是 crop 后从 PDB pkl token annotation 取出的 `rna_binding_signal`。

### 数据管线改动

在 `BaseSingleDataset.get_feature_and_label()` 中，在 crop 后、生成 `features_dict` / `labels_dict` 时加入：

```python
if "rna_binding_signal" in cropped_token_array token annotations:
    labels_dict["rna_binding_signal"] = torch.tensor(
        cropped_token_array.get_annotation("rna_binding_signal"),
        dtype=torch.float32,
    )
    labels_dict["rna_binding_signal_mask"] = torch.tensor(
        cropped_token_array.get_annotation("rna_binding_signal_mask"),
        dtype=torch.bool,
    )
```

注意不要把 signal 存成只含 RNA 长度的向量。PDB 结构训练中更稳的是 token-length 向量 `[N_token]`，之后用 RNA token indices 取子集：

```python
p_bind, rna_token_indices = compute_distogram_binding_score(contact_probs, feat_dict)
target = label_dict["rna_binding_signal"][rna_token_indices]
target_mask = label_dict["rna_binding_signal_mask"][rna_token_indices]
```

这样 crop、chain shuffle、token order 都不会破坏对齐。

### Trainer 改动方案

推荐新增一个 runner，而不是大幅改坏现有两个路径：

```text
runner/train_rna_signal_sft.py
```

它应复用原 `AF3Trainer` 的主体逻辑，但重写 `init_loss()` 和 `get_loss()`：

1. `self.loss = ProtenixLoss(self.configs)` 保留。
2. `self.signal_loss = EclipSignalLoss(...)` 复用当前 signal loss。
3. `model_forward()` 仍然走原 Protenix forward，得到 `pred_dict` 和 `label_dict`。
4. `get_loss()` 先算原结构 loss，再从 `pred_dict["distogram"]` 算 `p_bind`，最后加 signal loss。

伪代码：

```python
structure_loss, structure_metrics = self.loss(
    feat_dict=batch["input_feature_dict"],
    pred_dict=batch["pred_dict"],
    label_dict=batch["label_dict"],
    mode=mode,
)

signal_loss, signal_metrics = self.get_structure_signal_loss(
    feat_dict=batch["input_feature_dict"],
    pred_dict=batch["pred_dict"],
    label_dict=batch["label_dict"],
)

loss = structure_loss + configs.rna_signal_sft.signal_loss_weight * signal_loss
```

如果样本没有 `rna_binding_signal`，训练直接报错；这条 PDB SFT 路径默认要求先把 train/test pkl 全部补好 signal。只有 crop 后没有有效 RNA signal mask 时，signal loss 返回 0，不影响结构训练。

### eCLIP 兼容性

当前 `runner/train_eclip_ppft.py` 可以保持 eCLIP 专用：

- 继续读取 eCLIP parquet。
- 继续只使用 signal loss 和可选 quality loss。
- 不要求真实结构 label。

PDB SFT 新 runner 复用同一个 `EclipSignalLoss` 和 distogram contact helper，但走原 PDB dataset 与 ProtenixLoss。这样 eCLIP 和 PDB 代码路径清晰，不会互相污染。

不把 eCLIP 和 PDB 强行合到一个 runner。两类 batch schema 差异太大，分开实现更接近原始 Protenix 训练路径，也避免增加无关命令参数。

### 参数建议

新增配置文件可命名：

```text
configs/configs_rna_signal_sft.py
```

建议默认：

```python
rna_signal_sft = {
    "signal_profile_weight": 1.0,
    "signal_point_weight": 0.2,
    "signal_loss_weight": 1.0,
}
```

如果 PDB 结构监督权重本身很强，`signal_loss_weight` 可以先设为 `0.2` 或 `0.5` 做 smoke test，避免 signal 直接主导原结构学习。

## 需要修改的文件

第一阶段，数据加工：

```text
scripts/add_structure_rna_signal_to_protenix_pkl.py
tests/test_structure_rna_signal_pkl.py
```

第二阶段，PDB 训练兼容：

```text
protenix/data/pipeline/dataset.py
protenix/model/eclip_binding.py
configs/configs_rna_signal_sft.py
runner/train_rna_signal_sft.py
tests/test_pdb_signal_crop_alignment.py
tests/test_rna_signal_sft_loss.py
```

可选地更新：

```text
configs/configs_data.py
.vscode/launch.json
scripts/run_rna_signal_sft_ddp4.sh
```

## 验证计划

### 数据加工验证

1. 随机取 10 个 train pkl，写入 signal 后重新 load。
2. 检查 `len(rna_binding_signal) == len(token_array)`。
3. 检查 RNA token 上 mask 有效，protein token 上 mask 为 0。
4. 检查至少一部分 RNA-protein interface 样本 `signal.sum() > 0`。
5. 对一个已知小结构人工验证 6A contact 与 signal peak 一致。

### crop 对齐验证

构造一个小 pkl 或 mock token array：

1. 写入 token-level signal。
2. 走 `BaseSingleDataset.process_one()`。
3. 检查 crop 后 `label_dict["rna_binding_signal"]` 与 `input_feature_dict["is_rna"]` 的 token 顺序一致。
4. 检查 `compute_distogram_binding_score()` 返回的 `rna_token_indices` 能正确切出 target。

### 训练验证

1. 单卡 2-4 个样本 overfit。
2. 确认日志中同时有原 Protenix structure loss 和 signal loss。
3. 确认 `profile_auprc`、`topk_overlap` 可计算。
4. 确认没有 signal 的 crop 不会报错，只跳过 signal loss。
5. 确认保存的 checkpoint 仍然只在 `checkpoint["model"]` 中保存 Protenix 参数，不把 signal loss 模块混进 Protenix 主模型结构。

## 主要风险

1. **indices 中存在非 RNA-protein interface 行**  
   需要过滤或新建 signal 训练 indices，否则 signal loss 有效样本比例会很低。

2. **随机 crop 可能裁掉 RNA 或 protein**  
   对 signal 训练应优先使用 SpatialInterfaceCropping，或新增 signal-aware crop，优先保留 RNA-protein contact 区域。

3. **结构 contact signal 与 eCLIP signal 分布不同**  
   PDB signal 是结构接触 proxy，通常更稀疏、更接近 0/1；eCLIP 是实验富集信号。两者可以复用 loss 形式，但指标解释不能完全等同。

4. **直接覆盖 pkl 有数据损坏风险**  
   第一版必须写新目录，完成统计和 smoke test 后再替换配置。

5. **训练目标可能冲突**  
   原结构 loss 要求真实结构还原，signal loss 要求 distogram contact 能解释 RNA binding。结构派生 signal 来自同一个真实结构，理论上不冲突；但如果 signal 权重过大，会让 pairformer 更偏向 contact 分类而不是整体结构质量。

## 推荐执行顺序

1. 写 `add_structure_rna_signal_to_protenix_pkl.py`，先处理 20 个 pkl 做 smoke test。
2. 在 `BaseSingleDataset` 中透传 crop 后的 token-level signal 到 `label_dict`。
3. 写 PDB signal loss 单测，验证 target 对齐。
4. 新增 `runner/train_rna_signal_sft.py`，在原 Protenix loss 外加 signal loss。
5. 单卡 overfit 小样本。
6. 4/8 卡 DDP smoke test。
7. 全量加工 train/test pkl。
8. 正式训练。
