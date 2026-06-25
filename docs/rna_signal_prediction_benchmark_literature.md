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

