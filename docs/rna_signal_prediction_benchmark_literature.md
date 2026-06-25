# RNA 残基级 binding signal 预测任务的 benchmark 文献调研

本文档面向当前 Protenix/eCLIP signal SFT 任务：输入 protein 和 RNA 序列，模型输出每个 RNA 残基的 binding score，并与实验或结构派生的 RNA residue-level binding signal 对齐。调研时优先选择输出为 **每个 RNA 碱基/残基的 binding profile、crosslink count distribution、binding affinity track 或 binding site score** 的工作；只做整段 RNA 是否结合的分类模型不作为主要 benchmark，只在其被文献改造成伪单碱基 baseline 时提及。

## 结论摘要

最接近当前任务的公开工作是：

1. **RBPNet**：从 RNA 序列预测 CLIP/eCLIP crosslink count distribution，输出长度与输入一致的单碱基 profile。它的 loss、指标和 profile 评测方式最值得参考。
2. **iDeepB**：基于 ENCODE eCLIP 与 RNA-seq 构建 expression-aware benchmark，预测 101 bp RNA 上的 base-resolution RBP binding profile。它提供了较完整的 signal AP、peak AP、Pearson、Spearman 对标数值。
3. **Reformer**：单个 transformer 模型整合 RBP/cell-line token，预测 511 nt 区域的 base-level binding affinity，标签是 log2(1 + normalized fold-change coverage)。它与 eCLIP signal regression 很接近，但训练区域偏向 strong binding peaks。
4. **TransRBP**：预印本，利用 RNA sequence 与 m6A signal 预测 m6A-related RBP 的 base-resolution binding profile。它说明多模态表观转录组信号能显著提高 RBP binding profile 预测，但任务依赖 m6A 输入，不是通用 protein-RNA 序列模型。
5. **ProtScan / GraphProt / DeepBind 类方法**：可作为历史参考。ProtScan 输出 transcriptome-wide nucleotide-level interaction profile，但标签主要来自 peak/binding-site 距离，而不是连续 eCLIP signal；GraphProt/DeepBind 本质是区域分类模型，只有通过滑窗或 profile 后处理才能近似单碱基输出。

对你当前模型的直接启发：

- 评价时不要只看 BCE loss；应该主看 **AUPRC/AP、AUROC、top-k precision/overlap、per-sample 或 per-transcript Spearman/Pearson**。
- AUPRC 必须报告 **random baseline = positive rate**。如果你的 target positive rate 约 0.07，AUPRC 0.09 只比随机高约 1.3 倍，说明 ranking 能力很弱。
- 文献中常用两种二值标签：`crosslink count >= 2` 的 signal-positive site，以及 ENCODE/PureCLIP peak site。建议你的 benchmark 同时保留 `signal > 0` 和更严格的 `signal >= 2` 或 top-quantile positive。
- RBPNet/iDeepB/Reformer 多数只输入 RNA 序列、RBP ID 或 cell-line 信息，不输入 protein 序列，也不 rollout 结构。因此它们适合作为 **signal profile upper/reference baseline**，但不是完全同任务的公平横向比较。

## 候选 benchmark 汇总表

| 工作 | 输出粒度 | 输入 | 标签/监督 | 数据集 | 训练/切分 | Loss | 主要指标 | 报告结果 | 与当前任务相似度 |
|---|---:|---|---|---|---|---|---|---|---|
| RBPNet, Genome Biology 2023 | 单碱基 profile | RNA sequence；可使用 SMInput control | eCLIP/iCLIP/miCLIP read-start crosslink count distribution | ENCODE 103 个 HepG2 eCLIP RBP 模型；另含 iCLIP/miCLIP | 300 nt windows；chr2/9/16 验证，chr1/8/15 测试 | multinomial NLL；total + control tracks | PCC、auROC、AP | mean PCC 0.328；PureCLIP site auROC 0.89，AP 0.086 | 很高：直接 sequence-to-signal profile |
| iDeepB, NAR 2025 | 单碱基 profile | RNA sequence + cell-specific expression context；可带 control | eCLIP crosslink count profile | ENCODE 225 paired-end eCLIP，150 RBPs，K562/HepG2/adrenal gland | 101 bp windows；chr2/9/16 验证，chr1/8/15 测试 | Poisson NLL | PCC、Spearman、signal AUC/AP、peak AUC/AP | full-transcript：PCC 0.31，Spearman 0.24，signal AP 0.20，peak AP 0.11；优于 RBPNet | 很高：直接 base-resolution eCLIP signal |
| Reformer, Patterns 2025 | 单碱基 binding affinity | cDNA 511 nt + RBP/cell-line token | log2(1 + normalized fold-change coverage) | 225 eCLIP experiments，155 RBPs，3 cell lines | SR train/val/test：872,618/23,633/94,713 sequences | MSE；另有 BCE 的 Reformer-BC | Spearman；BC AUC | base-level Spearman 0.63；per-sequence mean Spearman 0.65；peak-level Spearman 0.76/0.65 | 高：base-level signal regression，但偏 peak 区域 |
| TransRBP, bioRxiv 2024 | 单碱基 profile | RNA sequence + m6A signal | eCLIP-derived binding profiles | 32 个 m6A-related RBPs | 文中按 RBP/cell context 评测 | transformer regression/classification 组合，细节需看原文 | PCC/SCC/accuracy 等 | median accuracy 0.59；部分例子 PCC 明显高于 RBPNet | 中高：profile 任务相似，但依赖 m6A signal |
| ProtScan, arXiv 2024 | transcriptome-wide nucleotide profile/peaks | RNA sequence k-mer/string-kernel features | binding site 距离标签；从 CLIP peak/binding regions 派生 | 96 RBPs，K562/HepG2，两 replicate；benchmark 用 11 RBPs、34 comparisons | 约 1% human genome genes 测试 | ridge regression squared loss + consensus voting + smoothing | AUROC、auPRC | average AUROC 0.8；average auPRC 0.008，positive ratio 约 1:2500 | 中：单碱基 profile，但不是连续 eCLIP signal |
| DeepRiPe / DeepBind / PrismNet / GraphProt | 主要是区域/窗口分类 | RNA sequence，部分加入结构/annotation | bound/unbound region labels | ENCODE/eCLIP/CLIP/RNAcompete 等 | 依模型而异 | BCE/SVM/classification loss | AUC/AUPRC | RBPNet 文献中 DeepRiPe 伪单碱基 baseline：auROC 0.74，AP 0.012 | 低到中：不是天然 residue-level profile |

## 当前使用的 parnet/ENCODE eCLIP 数据说明

你当前用于 Protenix eCLIP signal SFT 的数据来自 `parnet` 发布的 ENCODE eCLIP 多任务数据集，而不是 iDeepB 或 RBPNet 原始 benchmark 的逐模型数据文件。它与 iDeepB 在数据来源上非常接近：都来自 ENCODE eCLIP，覆盖约 150 个 RBP，并主要包含 K562 和 HepG2 两个细胞系；但样本组织方式不同。

`parnet` README 中说明其主训练集是 HuggingFace Dataset format，包含 223 条 ENCODE eCLIP tracks。本地数据位于：

```text
/inspire/ssd/project/sais-bio/public/xiangwenkai/GITHUB/parnet/encode.filtered.hfds
```

原始 HFDS 的一个样本不是一个 `(protein_sequence, RNA_sequence, label)` 蛋白-RNA 配对，而是一个 RNA genomic window 及其多任务信号矩阵：

```text
inputs.sequence    sparse one-hot RNA sequence，通常长度 600
outputs.eCLIP      sparse [223, 600] eCLIP signal matrix
outputs.control    sparse [223, 600] SMInput/control signal matrix
meta.name          genomic interval，例如 chr14:100374289-100374889:-
```

这里的 `223` 是 task 数。每个 task 是一个固定 ENCODE 实验轨道，由 `protein_symbol + cell_line` 定义，而不是动态输入的蛋白序列。本地 task map 为：

```text
/inspire/ssd/project/sais-bio/public/xiangwenkai/GITHUB/parnet/parnet/assets/ENCODE.idx2symbol-cell.tsv
```

本地统计结果：

| 项目 | 数量 |
|---|---:|
| eCLIP task tracks | 223 |
| unique protein symbols | 150 |
| K562 tracks | 120 |
| HepG2 tracks | 103 |
| train RNA windows | 512,946 |
| validation RNA windows | 116,542 |
| test RNA windows | 70,626 |

因此，`parnet` 原始训练范式是 **RNA sequence -> 223 个 task 的 per-base eCLIP/control profile**。protein identity 在原模型中是固定输出 head 的 task index，不是输入 protein sequence。

为了给 Protenix 使用，本地用以下脚本把 HFDS 展开成 task-level parquet rows：

```text
/inspire/ssd/project/sais-bio/public/xiangwenkai/GITHUB/parnet/parnet/bin/export_hfds.py
/inspire/ssd/project/sais-bio/public/xiangwenkai/GITHUB/parnet/export_all_hfds.sh
```

展开后的每一行对应一个 `(protein_symbol, cell_line, RNA window)`：

```text
rna_seq
protein_symbol
cell_line
signal_vector
```

也就是说，一个 600 nt RNA window 会按 223 个 task 展开，最多变成 223 条 task-level 样本：

```text
512,946 train RNA windows x 223 tasks = 114,386,958 train task-level rows
```

这解释了为什么你看到的数据可以表述为“约 150 个蛋白、2 个细胞系、50 多万 RNA 序列、总计约一亿多条 protein/RNA signal rows”。这里的一亿多条不是独立测序得到的一亿多条 RNA，而是 RNA windows 乘以 ENCODE task tracks 后的展开结果。

当前主要训练路径使用的是进一步过滤后的高质量正样本：

```text
/inspire/ssd/project/sais-bio/public/xiangwenkai/GITHUB/parnet/data_process/high_quality_positive
```

该目录由 `parnet/data/eclip_quality_filter.py` 从全部 task-level rows 过滤得到。当前 summary 显示：

| Split | 输入 task-level rows | 保留 high-quality positive rows |
|---|---:|---:|
| train | 114,386,958 | 969,307 |
| validation | 25,988,866 | 221,572 |
| test | 15,749,598 | 113,936 |

过滤逻辑按 `protein_symbol` 分组校准阈值，筛选有足够总信号、足够 peak 信号、且信号在局部窗口内较集中的 profile。当前 `high_quality_positive` 是 positive-only subset，`include_negatives = false`；它不是 `parnet` 原始完整训练分布，也不是全量 eCLIP 负样本集。

### signal 数值来源与含义

`signal_vector` 中的数值是每个 RNA 位置上的 eCLIP count-like signal。它不是模型预测概率，也不是 `log1p` 后的归一化标签。

从本地代码能确认的处理链条是：

1. `parnet/data/bioIO.py` 中的设计从 stranded bigWig 读取每个 genomic interval 上的逐碱基值。
2. 这些逐碱基值被堆叠成 `outputs.eCLIP` 和 `outputs.control`，形状为 `[num_tasks, sequence_length]`。
3. `parnet/data/datasets.py` 读取 HFDS 时只是把 sparse tensor 转 dense，不做 log transform、softmax 或概率归一化。
4. `parnet/bin/export_hfds.py` 展开 HFDS 时只是把某个 task 的 dense eCLIP vector 写成 `signal_vector`，同样不做归一化。
5. `eclip_quality_filter.py` 额外生成的 `profile_label` 才是从 `signal_vector` 派生的训练标签：先对 signal 做 `log1p`，再在单条 RNA 内归一化。

因此，你在 parquet 里看到 `signal_vector` 多数是整数，是因为上游 eCLIP bigWig 轨道本身保存的是每个碱基位置的 read/crosslink count 或 count-like coverage。HFDS 虽然把这些值存成 `float32`，parquet 中也可能显示为 double/list float，但数值语义仍然是观测到的 eCLIP 计数强度。简单说：

```text
signal_vector[i] = 这个 ENCODE eCLIP task 在该 RNA window 第 i 个碱基上的观测 read/crosslink count-like signal
```

它的几个派生量含义如下：

| 字段 | 计算方式 | 含义 |
|---|---|---|
| `signal_total` | `sum(signal_vector)` | 该 RNA window 对该 RBP/cell task 的总 eCLIP 信号 |
| `signal_peak` | `max(signal_vector)` | 单碱基最高信号 |
| `signal_nonzero_fraction` | `mean(signal_vector > 0)` | 有观测信号的位置比例 |
| `signal_topk_ratio` | `sum(top-k signal) / sum(signal)` | 信号是否集中在少数位置 |
| `signal_window_var_norm` | 先按 10 nt window 求和，再计算归一化方差 | 信号是否局部集中 |
| `profile_label` | `log1p(signal_vector) / sum(log1p(signal_vector))` | 单条 RNA 内的归一化 profile 标签 |

这也意味着当前 Protenix 训练里的二值标签：

```text
target_binary = signal_vector > 0
```

本质上是在判断“该位置是否有非零 eCLIP 观测信号”；连续 point loss 使用的 per-sample normalized `log1p(signal)` 则是在学习“非零信号在 RNA window 内的相对强弱分布”。如果要和 iDeepB 的指标更接近，可以额外报告 `signal >= 2` 的 strict label，因为 iDeepB 的 signal AUC/AP 使用了 crosslink count `>= 2` 作为 positive base。

需要注意一个限制：当前本地 `export_hfds.py` 导出的字段没有包含 `control_signal`，虽然 HFDS 中存在 `outputs.control`。因此当前 `high_quality_positive` 过滤 summary 中 `control_column = null`，没有做 control-adjusted signal，例如 `log1p(eCLIP) - log1p(control)`。如果后续要更严格对齐 iDeepB/RBPNet 的 bias-correction 思路，应考虑重新导出 control track 或在过滤/训练时引入 control-adjusted label。

#### signal来源解释
  signal_vector 整数值来源解释：它是从 eCLIP bigWig 轨道读取的 per-base read/crosslink count-like signal，不是概率，也不是 log/归一化后的标签。
  signal = 0      没有观测到 eCLIP 支持
  signal = 1      极弱/低置信信号
  signal >= 2     更像 iDeepB 里常用的 positive base 阈值
  signal peak 高 + 局部集中 + total 高   更可信的 binding profile
  所以训练/评估时建议保留两套 label：
  loose positive:  signal > 0
  strict positive: signal >= 2
#### parnet训练预测以及计算指标时，有进行这种强弱过滤吗？计算AP指标，是否是把信号转为0，1变量后计算的？
  1. parnet 训练时有强弱过滤，但不是逐碱基过滤
  本地 parnet 的训练 loss 是 MultinomialNLLLoss，配置里：
  parnet.losses.MultinomialNLLLoss.min_height = 3
  代码位置：/inspire/ssd/project/sais-bio/public/xiangwenkai/GITHUB/parnet/parnet/losses.py:21
  核心逻辑是：
  min_height_map = counts.max(dim=-1).values >= self.min_height
  也就是说，对每个 sample x task 的 600 nt signal profile，如果这个 profile 里 最大 signal < 3，这个 task/profile 的 loss 会被 mask 掉，不参与 multinomial NLL。
  但它不是说：
  signal = 1 的位置被删掉
  而是：
  如果一个 profile 里至少有某个位置 signal >= 3，
  那么整个 count profile 参与训练，里面 signal=1 的位置仍然保留。
  所以 parnet 训练不是逐碱基把弱信号过滤掉，而是按整条 profile 判断这个 task/window 是否有足够强的 signal。

  2. parnet 训练 loss 不把 signal 转成 0/1
  parnet 的 loss 不是 BCE。它用的是 multinomial NLL：
  torch.distributions.Multinomial(logits=logits).log_prob(counts)
  这里的 counts 是原始 count profile，例如：
  [0, 0, 1, 0, 3, 5, 0, ...]
  模型预测的是这个 window 内 count mass 的位置分布。训练时不会先变成：
  [0, 0, 1, 0, 1, 1, 0, ...]
  所以 parnet 训练更像是在学：
  信号主要集中在哪些碱基，以及相对强弱如何分布
  而不是单纯学每个位置是否 positive。

  3. parnet 本地训练指标里没有 AP
  本地配置里 metric 是：
  train.metrics = {
      'pcc': PearsonCorrCoeff,
      'filtered_pcc': FilteredPearsonCorrCoeff
  }
  代码位置：/inspire/ssd/project/sais-bio/public/xiangwenkai/GITHUB/parnet/configs/RBPNet-11M.config.gin:9
  也就是说，当前 parnet 训练代码主要看 Pearson/filtered Pearson，没有看到内置 AP/AUPRC 训练指标。

  4. 如果计算 AP，一定需要把真实 signal 转成 0/1

  AP/AUPRC 是二分类 ranking 指标，所以 target 必须是 binary label。常见做法是：
  loose:  positive = signal > 0
  strict: positive = signal >= 2
  peak:   positive = ENCODE/PureCLIP peak site

  预测值不需要二值化，仍然用连续 score，例如：
  score = predicted binding probability / predicted profile probability / p_bind
  然后计算：
  average_precision_score(binary_target, score)
  所以回答你的问题：
  parnet 训练 loss：不转 0/1，用原始 count profile，并按 max signal >= 3 过滤整条 profile。
  parnet 本地 metric：主要是 Pearson，不是 AP。
  文献里的 AP/iDeepB 的 signal AP：是把真实 signal 转成 0/1 后算的，常用 threshold 是 count >= 2。
  对你当前任务，我建议同时报：
  auprc_loose:  target = signal > 0
  auprc_strict: target = signal >= 2
  signal >= 2 更能排除很多只有 1 条 read 支持的弱噪声位置。

  signal_binary_threshold设成 2 的原因是对齐 iDeepB/RBPNet 文献里常见的 base-level signal positive 定义：positive base = crosslink count >= 2

#### 极值应对
   单个signal截断为100，总的signal做total count cap：
      total = signal.sum()
      if total > max_total:
            signal_for_loss = signal / total * max_total
      else:
            signal_for_loss = signal

## 1. RBPNet：最直接的 sequence-to-crosslink-profile baseline

**论文**：Horlacher et al., *Towards in silico CLIP-seq: predicting protein-RNA interaction profiles from sequence*, Genome Biology, 2023.  
**代码**：`https://github.com/mhorlacher/rbpnet`

### 任务定义

RBPNet 输入 300 nt RNA sequence，输出同长度的 probability vector，用来参数化该窗口内 crosslink read-start counts 的 multinomial distribution。它不是预测一个窗口是否 bound，而是预测每个位置的 crosslink count distribution，因此和当前 RNA 残基级 signal 预测最接近。

RBPNet 对 eCLIP 的一个重要处理是 bias correction：同时建模 total/eCLIP signal、SMInput/control signal 和 unobserved target/protein-specific signal。实际评估 crosslink count shape 时常用 total track；评估 PureCLIP crosslink sites 时使用 target track。

### 数据集与预处理

- ENCODE eCLIP：103 个 HepG2 RBP eCLIP experiments。
- 输入窗口：先在 GENCODE genes 上用 100 nt 滑窗，stride = 1。
- 候选窗口筛选：
  - Poisson test `p < 0.01`
  - window total count `N >= 8`
  - positional maximum count `H >= 2`
  - 记录候选后滑窗前进 50 nt，减少冗余
  - 最终 100 nt window 对称扩展到 300 nt。
- 平均每个 eCLIP dataset 302,752 个候选位点；最少 LARP7 7,937，最多 HNRNPC 1,105,807。
- 切分：chromosome-wise split，validation = chr2/chr9/chr16，hold-out test = chr1/chr8/chr15，train = 其他 autosomes。

### Loss

每个 track 的输出 `p_pred` 是一个 multinomial probability vector。给定真实 counts `c_obs` 和总 counts `n_obs`，loss 是 observed count vector 在该 multinomial distribution 下的 negative log-likelihood。总 loss 是各 task-specific losses 求和；带 control 时同时约束 total 和 control track。

这比普通 BCE 更贴近 CLIP/eCLIP count profile，因为模型被迫学习 **窗口内部 count mass 如何在位置上分布**，而不是只学习每个位置独立二分类。

### 指标与结果

1. **Profile shape correlation**
   - 对 hold-out 300 nt sequences，计算预测 profile 与 merged replicate eCLIP counts 的 Pearson correlation coefficient。
   - RBPNet mean PCC = 0.328；不同 RBP 范围约 0.200 到 0.587。
   - eCLIP replicate-replicate 平均 PCC = 0.149，文中认为 RBPNet 接近或超过 replicate-level shape accuracy。

2. **PureCLIP crosslink site discrimination**
   - 在 hold-out chromosomes 的 transcripts 上调用 PureCLIP crosslink sites。
   - 对每条 transcript 内部计算 auROC 和 average precision，再对 transcripts 平均。
   - RBPNet：mean auROC = 0.89，mean AP = 0.086。
   - DeepRiPe 伪单碱基 baseline：mean auROC = 0.74，mean AP = 0.012。
   - AP 看起来低，是因为 positive site 极稀疏；文中给出的平均 PureCLIP positive fraction 约 0.0014，因此 AP 0.086 已远高于随机基线。

### 对当前任务的参考价值

RBPNet 的评估方式非常适合当前任务：

- 对连续 signal：计算 per-sample / per-RNA / per-protein 的 Pearson 或 Spearman。
- 对二值 signal：计算 AUPRC/AP 和 AUROC。
- AUPRC 必须与 positive rate 一起报告。
- 对不同 RNA/transcript 长度，建议在样本内部或 transcript 内部计算 AP/AUROC，再 macro-average，避免长 RNA 或高 signal 样本主导 pooled metric。

当前模型若使用 `p_bind_r = max_p p_contact[r,p]`，本质上输出的是单调 binding score，而不是 count distribution；因此更适合对标 RBPNet 的 PureCLIP site AP/AUROC，而不是 multinomial count likelihood。

## 2. iDeepB：eCLIP expression-aware base-resolution benchmark

**论文**：Liu et al., *Base-resolution binding profile prediction of proteins on RNAs*, Nucleic Acids Research, 2025.  
**网页/服务**：`http://www.csbio.sjtu.edu.cn/bioinf/iDeepB/`

### 任务定义

iDeepB 预测给定 101 bp RNA sequence 上每个碱基的 RBP binding profile。它显式指出已有很多方法只做 bound/unbound region classification，而 iDeepB/RBPNet 属于少数 base-resolution profile prediction 方法。

### 数据集

- ENCODE paired-end eCLIP：225 个 datasets。
- 覆盖 150 RBPs。
- Cell/tissue：K562 120 个，HepG2 103 个，adrenal gland 2 个。
- 整合 cell-specific RNA-seq expression profile，用 expressed genes 构建 benchmark。

### 数据构建

- 下载 ENCODE candidate narrow peaks，合并 replicates。
- narrow peak 过滤：
  - `log2(signal fold change) > 0`
  - `P-value < 0.01`
- peaks 注释到 expressed genes。
- 对长度 101 bp 到 36,000 bp 的 expressed genes，以 101 bp window、step = 101 切分并保留全部 subsequences。
- 对长度超过 36,000 bp 的 expressed genes，只保留最大 crosslink signal `>= 2` 且 treatment 最大 signal 高于 control 最大 signal 的 subsequences。
- 平均每个 eCLIP dataset 508,092 个 candidate sites；最少 NSUN2/K562 6,856，最多 HNRNPC/HepG2 1,445,270。

### 训练设置

- 切分：train = 其他 chromosomes，validation = chr2/chr9/chr16，test = chr1/chr8/chr15。
- 模型：one-hot RNA sequence -> CNN blocks -> BiLSTM -> multi-head attention -> MLP。
- Loss：Poisson negative log likelihood，用于建模 eCLIP crosslink count target；带 control 时 experimental profile loss 与 control profile loss 相加。
- Early stopping：validation AP 8 epochs 无提升停止；validation loss 5 epochs 无提升则 learning rate 衰减。

### 指标

iDeepB 使用六个指标，值得直接借鉴：

1. Pearson correlation：预测 profile vs observed crosslink counts。
2. Spearman correlation：同上，关注 rank。
3. Signal AUC：positive base 定义为 true crosslink count `>= 2`。
4. Signal AP：positive base 定义同上。
5. Peak AUC：positive base 定义为 ENCODE peak binding site。
6. Peak AP：positive base 定义同上。

### 报告结果

full-length transcript evaluation 中，iDeepB 与 RBPNet 的平均结果：

| 方法 | Pearson | Spearman | Signal AUC | Signal AP | Peak AUC | Peak AP |
|---|---:|---:|---:|---:|---:|---:|
| iDeepB | 0.31 | 0.24 | 0.82 | 0.20 | 0.68 | 0.11 |
| RBPNet retrained | 0.26 | 0.22 | 0.79 | 0.16 | 0.62 | 0.07 |

Cross-cell prediction 结果：

| 训练 -> 测试 | Pearson | Spearman | Signal AUC | Signal AP | Peak AUC | Peak AP |
|---|---:|---:|---:|---:|---:|---:|
| K562 -> HepG2, shared 72 RBPs | 0.272 | 0.22 | 0.797 | 0.179 | 0.685 | 0.108 |
| HepG2 -> K562, shared 72 RBPs | 0.254 | 0.209 | 0.783 | 0.166 | 0.65 | 0.09 |

### 对当前任务的参考价值

iDeepB 的指标体系比单一 BCE 更适合你的任务。建议你在 eCLIP validation 上至少记录：

- `signal_ap` / `auprc`：positive = `signal > 0`，以及 positive = `signal >= 2` 两套。
- `peak_ap`：如果有 peak 或 high-confidence binary label。
- `signal_auroc`：作为辅助，不能替代 AUPRC。
- `spearman_log_signal`：`p_bind` vs per-sample normalized `log1p(signal)`。
- `positive_rate`：每次报告 AUPRC 时同时报告。

如果你的 eCLIP target positive rate 约 0.07，而 AUPRC 约 0.09，那么只比随机基线高 0.02，说明模型排序能力明显不足；可参考 iDeepB 的 signal AP 0.16-0.20 作为更合理的目标区间，但注意 iDeepB 是专门为 eCLIP profile 训练的序列模型，不涉及 protein-RNA 结构 rollout。

## 3. Reformer：统一 transformer 的 base-level binding affinity regression

**论文**：Shen et al., *A deep learning model for characterizing protein-RNA interactions from sequences at single-base resolution*, Patterns, 2025.  
**DOI**：`10.1016/j.patter.2024.101150`

### 任务定义

Reformer 输入 cDNA sequence，并把 RBP/cell-line name 作为 token 加入模型，输出每个碱基的 binding affinity。它采用两阶段设计：

1. Reformer-BC：先判断一个 511 nt region 是否为 binding region。
2. Reformer：对 positive/high-affinity regions 做 single-base resolution binding affinity regression。

### 数据集与标签

- 225 个 ENCODE eCLIP experiments。
- 155 RBPs，3 个 cell lines。
- 标签是 fold enrichment coverage 处理后的连续值：
  - 先将 absolute fold change coverage 除以总 coverage sum，再乘以 `1e6`。
  - normalized coverage capped at 2500。
  - binding affinity = `log2(1 + normalized fold change coverage)`。
- Peak 定义：binding affinity 至少 3，且 `-log10(P value) >= 5`，相对 size-matched control enriched。
- 所有 binding sites 标准化为 511 nt，短区域从 midpoint 对称扩展，长区域从 midpoint 对称截断。
- SR train/validation/test：872,618 / 23,633 / 94,713 sequences。
- BC train/validation/test：1,745,538 / 42,022 / 180,040 sequences，负样本从不 overlap binding sites 的 transcriptome regions 随机抽取，数量匹配正样本。

### 模型与训练

- 12 层 transformer，每层 12 attention heads，hidden size 768。
- 输入 token 包含 `[CLS]`、`RBP&cell-line`、3-mer cDNA tokens、`[SEP]`。
- 对 511 nt input，输出右侧 trimming 后的 509 bp base-resolution prediction。
- Reformer loss：MSE，预测每个 base 的 binding affinity。
- Reformer-BC loss：BCE。
- 训练：SR task 30 epochs，learning rate `2e-5`，weight decay `1e-4`，8x A100 40GB。

### 指标与结果

- SR-test base-level Spearman：0.63。
- individual sequence mean Spearman：0.65。
- peak-level binding affinity Spearman：
  - aggregated across all eCLIP experiments：0.76。
  - per individual experiment：0.65。
- predicted vs observed peak affinity difference 与 biological replicate difference 接近：0.61 vs 0.60。
- Reformer-BC 在 BC-test 上 AUC 高于 HDRNet、DeepCLIP、PrismNet、DeepBind。

### 对当前任务的参考价值

Reformer 的标签定义和你现在的 `per-sample normalized log1p(signal)` 思路很接近，都是把 eCLIP signal 转换到更稳定的连续 affinity scale，再做 regression。

但 Reformer 的训练样本偏向 peak/high-affinity regions，且模型知道 RBP/cell-line token；你的模型输入是 protein sequence + RNA sequence，并通过 structure/contact probability 推出 binding。因此可比指标主要是：

- `Spearman(p_bind, normalized_log_signal)`。
- `MSE/point loss(p_bind, normalized_log_signal)` 仅作训练 loss，不建议作为主要 benchmark 指标。
- `AUPRC(p_bind, binary_signal)` 用来衡量 site ranking。

## 4. TransRBP：结合 m6A signal 的 RBP profile prediction

**论文**：Zhou et al., *In-silico modeling of RNA binding protein profiles using RNA sequence and m6A signal*, bioRxiv, 2024.

### 任务定义

TransRBP 针对 m6A-related RBP，利用 RNA sequence 与 m6A signal 预测 RBP binding profile。它强调 m6A modification 与 RBP binding 的关系，因此是多模态 profile prediction。

### 关键信息

- 任务是 base-resolution binding profile prediction，不只是整段分类。
- 输入包含 RNA sequence 和 m6A signal。
- 研究对象偏向 m6A-related RBPs，文摘中提到 32 个 RBPs。
- 报告 median accuracy 约 0.59，相比 state-of-the-art 提升约 28%。
- 文摘示例中，TransRBP 在若干 eCLIP datasets 上的 PCC 明显高于 RBPNet，例如某些 YTHDC1/YTHDF2 examples 中 TransRBP PCC 约 0.79/0.83，而 RBPNet 约 0.38/0.51。

### 对当前任务的参考价值

TransRBP 的重要启发不是具体数值，而是：**单靠 RNA sequence 可能不足以解释某些 RBP 的 binding profile，额外实验或表观转录组信号能显著提高 profile quality**。对当前 Protenix 模型而言，如果 eCLIP signal 很强地受 cell state、RNA abundance、RNA modification、accessibility 影响，仅从 protein/RNA sequence 和 rollout contact 中学习，AUPRC 和 Spearman 可能存在天然上限。

因此建议把 TransRBP 作为“带额外生物信号的上界参考”，而不是直接 baseline。

## 5. ProtScan：nucleotide-level binding site profile 的历史参考

**论文**：Corrado et al., *ProtScan: Modeling and Prediction of RNA-Protein Interactions*, arXiv, 2024.

### 任务定义

ProtScan 将长 RNA 上 binding site localization 视为 regression task。它不是直接预测 eCLIP count，而是对短窗口预测其到最近 binding site 的距离，再通过 consensus voting 和 smoothing 得到 transcriptome-wide single-nucleotide interaction profile，并进一步做 peak extraction。

### 数据集

- 数据来自 CLIP-seq binding regions。
- full dataset 包括 96 RBPs。
- cell lines：K562 和 HepG2。
- 38 个 RBP 同时有两个 cell lines，40 个只在 K562，18 个只在 HepG2。
- 每个 experiment 有两个 replicates。
- binding sites 由 fold-change threshold 定义。
- 与 GraphProt/DeepBind 的 benchmark 使用 11 RBPs，共 34 comparisons；测试约 1% human genome，约 600 protein-coding/non-coding genes。

### Loss 与指标

- Loss：ridge regression squared loss + L2 regularization，使用 SGD。
- 回归标签：窗口中心到最近 binding site 的距离转换值。
- 指标：
  - AUROC：在所有 target transcripts 和所有 positions 上评价 interaction score ranking。
  - auPRC：用于极度类别不平衡的 position-level task。
- 数据极不平衡，文中报告 interacting vs non-interacting nucleotide 平均比例约 1:2500。

### 报告结果

- ProtScan average AUROC 约 0.8。
- 相比 GraphProt relative AUROC improvement 约 35%，相比 DeepBind 约 43%。
- ProtScan 在 34/34 comparisons 中优于 GraphProt；在 32/34 中优于 DeepBind，另 2 个持平。
- average/median auPRC：
  - ProtScan：0.008 / 0.006
  - DeepBind：0.003 / 0.001
  - GraphProt：0.002 / 0.001

### 对当前任务的参考价值

ProtScan 强调了一个关键事实：当评价单碱基 binding site 时，positive rate 可能极低，AUPRC 绝对值会很小。因此报告 AUPRC 时必须同时报告 positive rate 和 fold-over-random。

不过 ProtScan 的标签是 binding site/peak 距离，不是 eCLIP count signal；它更适合参考 **peak localization**，不适合作为连续 signal regression 的直接 benchmark。

## 6. 不建议作为主 benchmark 的模型

以下模型常见于 RBP-RNA binding 文献，但与当前“每个 RNA 残基 signal”任务不完全一致：

- **DeepBind**：主要预测 sequence/region 的 binding affinity 或 bound/unbound，不天然输出每个碱基 signal。
- **DeepCLIP**：主要从 CLIP data 学习 binding sequence motifs 和 region-level binding。
- **DeepRiPe**：使用 eCLIP/sequence/annotation 做 protein-RNA interaction classification；RBPNet 文献通过滑窗把它改造成 pseudo single-nucleotide score，但这不是原生 profile 模型。
- **PrismNet**：利用 in vivo RNA structure profile 预测 RBP binding，通常是 region/window-level binding prediction，不是直接输出 eCLIP count distribution。
- **GraphProt**：SVM/graph kernel 方法，主要用于 binding region classification 和 motif 学习；可以通过窗口扫描生成 profile，但不是原生 signal profile regression。
- **RNAProt/RBPsuite 类工具**：常用于 RBP binding site/region prediction 或 motif discovery，除非明确输出 per-base profile，否则只适合作为弱参考。

如果要 benchmark 当前模型，不建议拿这些模型的 region-level AUC 直接对比你的 residue-level AUPRC，因为任务难度和样本单位不同。

## 7. 建议当前 Protenix signal SFT 的 benchmark 方案

### 7.1 eCLIP validation/test 指标

建议每次 eval 输出以下指标：

1. `positive_rate`
   - `mean(target_binary)`。
   - AUPRC 的随机基线就是这个值。

2. `auprc`
   - 主指标。
   - 建议至少两套 binary label：
     - loose：`signal > 0`
     - strict：`signal >= 2` 或每条 RNA top q% signal。

3. `auroc`
   - 辅助指标。
   - 类别极不平衡时 AUROC 可能看起来较高但实际 top binding site retrieval 很差。

4. `precision_at_k` / `topk_overlap`
   - `k = num_positive` 或固定比例，例如 top 5%、top 10%。
   - 对实际 binding-site retrieval 更直观。

5. `spearman_log_signal`
   - 对 `p_bind` 和 per-sample normalized `log1p(signal)` 计算 Spearman。
   - 建议作为连续 signal ranking 指标。

6. `pearson_log_signal`
   - 可选。
   - 对 scale/calibration 更敏感。如果模型输出压在 0/1 附近，Pearson 可能异常低；此时 Spearman 更稳。

7. `brier` 或 calibration bins
   - 如果把 `p_bind` 当概率解释，建议做 calibration。
   - 当前 `p_contact` 或 `max contact prob` 未必是 calibrated binding probability。

### 7.2 聚合方式

建议同时报告三种聚合，避免被样本长度、RBP 数量或高 signal RNA 主导：

1. **Micro / pooled**
   - 把所有 RNA residues 拼起来算一次。
   - 优点：稳定；缺点：长 RNA 和多样本 RBP 主导。

2. **Macro by sample**
   - 每个 sample/RNA 单独算指标，再平均。
   - 优点：反映单条 RNA 的 retrieval 能力。

3. **Macro by protein/RBP**
   - 先对同一 protein/RBP 的样本平均，再跨 RBP 平均。
   - 优点：避免 HNRNPC 等大数据 RBP 主导结果。

如果某个样本没有 positive site，AUPRC 不可定义；建议跳过该样本并记录 `num_skipped_no_positive`。

### 7.3 与文献数值的合理对标

可以设置如下目标线：

| 目标线 | 含义 | 参考 |
|---|---|---|
| Random baseline | `positive_rate` | 任何 AUPRC 都必须高于它 |
| Weak useful | AUPRC >= 1.5x random | 如果 positive rate = 0.07，则约 0.105 |
| Moderate | AUPRC >= 2x random；Spearman > 0.1 | eCLIP signal 已有可用 ranking |
| Strong | AUPRC 接近 0.16-0.20；Spearman/Pearson 接近 0.2-0.3 | 接近 RBPNet/iDeepB full-transcript profile benchmark |
| Specialized upper reference | base-level Spearman 0.6 左右 | Reformer，但它有 RBP/cell token 且偏强 peak 区域，不应直接要求当前模型达到 |

你目前如果观察到：

- `profile_bce` 下降，但 `auprc` 约 0.09；
- `positive_rate` 约 0.07；
- `pearson` 接近 0；

那么合理解释是：模型学到了类别先验或概率校准的一部分，但没有学到 residue ranking。此时 BCE 下降不等价于 binding site retrieval 改善。

### 7.4 当前 Protenix contact-prob signal loss 的特殊注意点

你的当前预测是：

```text
p_bind_r = max_p p_contact[r, p]
```

这个定义有三个潜在问题：

1. `max` 很容易让多个 RNA residues 得到偏高分数，导致 `p_bind_mean` 偏高。
2. contact probability 是结构 contact 的模型内部概率，不一定与 eCLIP crosslink intensity calibration 一致。
3. eCLIP signal 受 RNA abundance、cell line、RBP expression、RNA modification、RNA accessibility、UV/crosslink bias 影响；而 structure contact 只覆盖其中一部分机制。

因此，当前模型最应该对标的是 **AUPRC/top-k retrieval**，而不是要求 `p_bind` 绝对值与 normalized signal 强校准。

## 8. 推荐优先复现实验

### 8.1 最小可行 benchmark

在你的 eCLIP validation set 上实现：

```text
positive = signal > 0
score = p_bind_dist
metric = AUPRC, AUROC, precision@num_positive, top10% overlap
aggregation = sample macro + protein macro + pooled micro
```

同时输出：

```text
positive_rate
auprc / positive_rate
num_eval_samples
num_skipped_no_positive
p_bind_mean
p_bind_positive_mean
p_bind_negative_mean
```

其中 `auprc / positive_rate` 是非常直观的 fold-over-random 指标。

### 8.2 与文献最接近的 strict benchmark

增加一套 label：

```text
positive = signal >= 2
regression_target = per-sample normalized log1p(signal)
```

输出：

```text
signal_ap_strict
signal_auc_strict
spearman_log_signal
```

这套定义与 iDeepB 的 signal AP/AUC 最接近。

### 8.3 结构数据 benchmark

对于 PDB-derived signal：

```text
positive = RNA residue has any protein heavy atom within contact_radius
score = max protein contact probability from distogram/contact_probs
```

指标：

```text
contact_auprc
contact_auroc
precision@num_contacts
topk_overlap
```

这里可以把 Protenix 原始 distogram/contact probability 当作天然 baseline：

1. frozen original Protenix。
2. RNA signal SFT 后模型。
3. eCLIP PPFT 后模型。

如果 SFT 后结构 contact benchmark 下降明显，说明 signal loss 正在破坏原有 contact/contact-prob calibration。

## 9. 文献与资料链接

1. RBPNet  
   - Paper: `https://link.springer.com/article/10.1186/s13059-023-03015-7`  
   - Code: `https://github.com/mhorlacher/rbpnet`

2. iDeepB  
   - Paper: `https://academic.oup.com/nar/article/53/14/gkaf748/8223173`  
   - Web server: `http://www.csbio.sjtu.edu.cn/bioinf/iDeepB/`

3. Reformer  
   - Paper: `https://pmc.ncbi.nlm.nih.gov/articles/PMC11783876/`  
   - DOI: `https://doi.org/10.1016/j.patter.2024.101150`

4. TransRBP  
   - Preprint: `https://www.biorxiv.org/content/10.1101/2024.11.23.624962v1`

5. ProtScan  
   - Preprint: `https://arxiv.org/abs/2412.20933`  
   - Code: `https://github.com/gianlucacorrado/ProtScan/`

6. ENCODE RBP/eCLIP reference  
   - Van Nostrand et al., *A large-scale binding and functional map of human RNA-binding proteins*, Nature, 2020.  
   - `https://www.nature.com/articles/s41586-020-2077-3`
