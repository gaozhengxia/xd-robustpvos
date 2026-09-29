# TABLE-13 消融矩阵（脚本生成，勿手改）

- 协议：`configs/_tau013.yaml`，stride 1，level 3，max_objects 0（30 段 / 61 实例）
- 聚合：`sequence_level` → `collapse_objects` → 段等权平均
- Δ = 臂 − A0（按序列配对）；CI = 配对差的自助法 95% 百分位区间；p = 双侧 Wilcoxon 符号秩（同相位 7 臂 Holm 校正）
- **两个相位分别报告、不合并**：A0 在 `fog` 与复合 `C1_fog_noise` 上难度不同（0.650 / 0.459）
- **读数按效应量优先**：|Cliff's δ| < 0.147 一律写「无可检出贡献」，不看 p（n = 30 时 0.02 的位移天然显著）
- **臂生效检查（C1）**：每个臂至少改变 49/61 个实例 ⇒ 开关确实到达了 pipeline；与 A0 逐位相同的臂会直接报错
- **臂生效检查（fog）**：每个臂至少改变 44/61 个实例 ⇒ 开关确实到达了 pipeline；与 A0 逐位相同的臂会直接报错

| 相位 | # | 变体 | J&F | Δ vs A0 | 95% CI | p (Holm) | Cliff's δ | 胜/负 | 编码次数 | 读数 |
|---|---|---|---|---|---|---|---|---|---|---|
| fog@3 | A0 | full DAG-RS | 0.6503 | — | — | — | — | — | 138.5 | reference |
| fog@3 | A1 | equal-weight fusion over the SAME operator set | 0.6506 | +0.0003 | [-0.0023, +0.0034] | 1 | -0.002 | 15/15 | 138.4 | no material contribution (delta=+0.0003, cliff=-0.002 < 0.147; 15/15 seqs, Holm p=1) |
| fog@3 | A2 | no time-consistency term (w1 = 0) | 0.6495 | -0.0007 | [-0.0175, +0.0161] | 1 | -0.013 | 13/17 | 111.4 | no material contribution (delta=-0.0007, cliff=-0.013 < 0.147; 13/17 seqs, Holm p=1) |
| fog@3 | A3 | random operator per frame | 0.6300 | -0.0203 | [-0.0425, -0.0027] | 0.026 | -0.071 | 7/23 | 65.6 | no material contribution (delta=-0.0203, cliff=-0.071 < 0.147; 7/23 seqs, Holm p=0.026) |
| fog@3 | A4 | single fixed domain-prior operator | 0.6524 | +0.0022 | [-0.0272, +0.0329] | 1 | +0.000 | 15/15 | 65.6 | no material contribution (delta=+0.0022, cliff=+0.000 < 0.147; 15/15 seqs, Holm p=1) |
| fog@3 | A5 | no degradation profile (full bank every decision) | 0.6459 | -0.0044 | [-0.0143, +0.0059] | 1 | -0.030 | 12/17 | 240.0 | no material contribution (delta=-0.0044, cliff=-0.030 < 0.147; 12/17 seqs, Holm p=1) |
| fog@3 | A6 | no vacuity gating | 0.6465 | -0.0038 | [-0.0171, +0.0060] | 1 | -0.022 | 15/11 | 102.6 | no material contribution (delta=-0.0038, cliff=-0.022 < 0.147; 15/11 seqs, Holm p=1) |
| fog@3 | A7 | no global re-anchoring | 0.6395 | -0.0108 | [-0.0272, +0.0053] | 1 | -0.039 | 10/15 | 129.6 | no material contribution (delta=-0.0108, cliff=-0.039 < 0.147; 10/15 seqs, Holm p=1) |
| C1_fog_noise@3 | A0 | full DAG-RS | 0.4592 | — | — | — | — | — | 133.8 | reference |
| C1_fog_noise@3 | A1 | equal-weight fusion over the SAME operator set | 0.4603 | +0.0010 | [-0.0134, +0.0136] | 1 | +0.004 | 15/15 | 132.5 | no material contribution (delta=+0.0010, cliff=+0.004 < 0.147; 15/15 seqs, Holm p=1) |
| C1_fog_noise@3 | A2 | no time-consistency term (w1 = 0) | 0.4755 | +0.0163 | [-0.0024, +0.0357] | 0.61 | +0.044 | 17/13 | 114.7 | no material contribution (delta=+0.0163, cliff=+0.044 < 0.147; 17/13 seqs, Holm p=0.61) |
| C1_fog_noise@3 | A3 | random operator per frame | 0.4839 | +0.0247 | [+0.0008, +0.0503] | 0.48 | +0.087 | 17/13 | 65.6 | no material contribution (delta=+0.0247, cliff=+0.087 < 0.147; 17/13 seqs, Holm p=0.48) |
| C1_fog_noise@3 | A4 | single fixed domain-prior operator | 0.4869 | +0.0277 | [+0.0066, +0.0490] | 0.038 | +0.078 | 21/9 | 65.6 | no material contribution (delta=+0.0277, cliff=+0.078 < 0.147; 21/9 seqs, Holm p=0.038) |
| C1_fog_noise@3 | A5 | no degradation profile (full bank every decision) | 0.4667 | +0.0074 | [-0.0077, +0.0260] | 1 | +0.039 | 16/13 | 242.9 | no material contribution (delta=+0.0074, cliff=+0.039 < 0.147; 16/13 seqs, Holm p=1) |
| C1_fog_noise@3 | A6 | no vacuity gating | 0.4797 | +0.0205 | [-0.0061, +0.0468] | 0.37 | +0.079 | 19/8 | 102.6 | no material contribution (delta=+0.0205, cliff=+0.079 < 0.147; 19/8 seqs, Holm p=0.37) |
| C1_fog_noise@3 | A7 | no global re-anchoring | 0.4546 | -0.0047 | [-0.0294, +0.0146] | 1 | -0.018 | 13/13 | 119.1 | no material contribution (delta=-0.0047, cliff=-0.018 < 0.147; 13/13 seqs, Holm p=1) |
