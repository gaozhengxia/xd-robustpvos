# CL3 side evidence: config-level IQA vs J&F (zero-GPU replay)

- observational unit: **(sequence x degradation) cell**, n = 360 degraded cells (30 sequences x 13 degradations)
- IQA recomputed by replaying `load_degraded_sequence` (CPU only); fidelity guard vs the landed severity fingerprint: worst |d| = **0.000e+00** over 390 cells
- usable threshold |rho| >= 0.6
- `across` CI: a plain cell-level bootstrap **and** a cluster bootstrap over sequences (resample sequences, taking all of a drawn sequence's cells).  The 360 cells are not independent draws either, so the clustered interval is the one to quote; the point estimate is identical under both
- within 95% CI: **cluster bootstrap over sequences** -- the sequences are the paired unit (the same ones appear in every degradation), so resampling cells or degradations would understate the interval; one rho per degradation is recomputed on every draw and then averaged
- the verdict is decided on **point estimates**; `CI 触线 = yes` means the interval reaches the threshold, so that metric must NOT be described as *definitely* too weak

## arm `dagrs`  (360 cells, 30 seqs)

| 指标 | 类型 | across rho | across CI 朴素格级 | across CI 按序列聚类 | R² | within 均值 | within CI 聚类 | within 逐档 min…max | usable | CI 触线 |
|---|---|---|---|---|---|---|---|---|---|---|
| `iqa_psnr` | full-reference | +0.155 | [+0.051,+0.255] | +0.056 … +0.255 | 0.024 | -0.049 | -0.163 … +0.073 | -0.339 … +0.157 | no | no |
| `iqa_ssim` | full-reference | +0.279 | [+0.178,+0.375] | +0.204 … +0.383 | 0.078 | -0.131 | -0.292 … +0.056 | -0.495 … +0.515 | no | no |
| `iqa_laplacian_var` | no-reference | +0.021 | [-0.082,+0.124] | -0.098 … +0.122 | 0.000 | +0.371 | +0.100 … +0.577 | +0.168 … +0.514 | no | no |
| `iqa_dark_channel` | no-reference | -0.080 | [-0.179,+0.020] | -0.158 … +0.006 | 0.006 | -0.149 | -0.469 … +0.195 | -0.302 … +0.040 | no | no |
| `iqa_contrast` | no-reference | +0.098 | [-0.001,+0.199] | -0.030 … +0.223 | 0.010 | +0.111 | -0.242 … +0.426 | -0.052 … +0.248 | no | no |
| `iqa_entropy` | no-reference | +0.118 | [+0.016,+0.217] | +0.005 … +0.227 | 0.014 | +0.140 | -0.153 … +0.395 | -0.024 … +0.251 | no | no |
| `iqa_tenengrad` | no-reference | +0.190 | [+0.089,+0.286] | +0.048 … +0.304 | 0.036 | +0.436 | +0.158 … +0.639 | +0.307 … +0.540 | no | **yes** |
| `iqa_saturation` | no-reference | +0.007 | [-0.093,+0.108] | -0.059 … +0.078 | 0.000 | -0.058 | -0.322 … +0.218 | -0.218 … +0.191 | no | no |

## arm `greedy`  (360 cells, 30 seqs)

| 指标 | 类型 | across rho | across CI 朴素格级 | across CI 按序列聚类 | R² | within 均值 | within CI 聚类 | within 逐档 min…max | usable | CI 触线 |
|---|---|---|---|---|---|---|---|---|---|---|
| `iqa_psnr` | full-reference | +0.157 | [+0.053,+0.258] | +0.059 … +0.261 | 0.025 | -0.024 | -0.149 … +0.104 | -0.402 … +0.120 | no | no |
| `iqa_ssim` | full-reference | +0.271 | [+0.170,+0.367] | +0.190 … +0.378 | 0.074 | -0.097 | -0.274 … +0.098 | -0.509 … +0.488 | no | no |
| `iqa_laplacian_var` | no-reference | +0.012 | [-0.088,+0.115] | -0.102 … +0.111 | 0.000 | +0.356 | +0.075 … +0.575 | +0.125 … +0.496 | no | no |
| `iqa_dark_channel` | no-reference | -0.069 | [-0.168,+0.032] | -0.157 … +0.021 | 0.005 | -0.096 | -0.429 … +0.244 | -0.234 … +0.011 | no | no |
| `iqa_contrast` | no-reference | +0.088 | [-0.013,+0.189] | -0.048 … +0.219 | 0.008 | +0.091 | -0.272 … +0.413 | -0.071 … +0.232 | no | no |
| `iqa_entropy` | no-reference | +0.108 | [+0.006,+0.208] | -0.009 … +0.222 | 0.012 | +0.112 | -0.197 … +0.379 | -0.087 … +0.260 | no | no |
| `iqa_tenengrad` | no-reference | +0.179 | [+0.078,+0.277] | +0.034 … +0.294 | 0.032 | +0.410 | +0.104 … +0.639 | +0.265 … +0.540 | no | **yes** |
| `iqa_saturation` | no-reference | +0.009 | [-0.094,+0.110] | -0.064 … +0.087 | 0.000 | -0.082 | -0.345 … +0.197 | -0.206 … +0.130 | no | no |

## arm `sam2video`  (360 cells, 30 seqs)

| 指标 | 类型 | across rho | across CI 朴素格级 | across CI 按序列聚类 | R² | within 均值 | within CI 聚类 | within 逐档 min…max | usable | CI 触线 |
|---|---|---|---|---|---|---|---|---|---|---|
| `iqa_psnr` | full-reference | +0.176 | [+0.074,+0.276] | +0.087 … +0.273 | 0.031 | -0.047 | -0.168 … +0.087 | -0.271 … +0.135 | no | no |
| `iqa_ssim` | full-reference | +0.385 | [+0.285,+0.478] | +0.299 … +0.488 | 0.148 | -0.098 | -0.266 … +0.096 | -0.410 … +0.475 | no | no |
| `iqa_laplacian_var` | no-reference | +0.041 | [-0.065,+0.147] | -0.061 … +0.127 | 0.002 | +0.477 | +0.202 … +0.663 | +0.202 … +0.665 | no | **yes** |
| `iqa_dark_channel` | no-reference | -0.065 | [-0.164,+0.034] | -0.140 … +0.005 | 0.004 | -0.042 | -0.361 … +0.290 | -0.208 … +0.178 | no | no |
| `iqa_contrast` | no-reference | +0.121 | [+0.023,+0.220] | +0.013 … +0.239 | 0.015 | +0.227 | -0.093 … +0.498 | +0.108 … +0.297 | no | no |
| `iqa_entropy` | no-reference | +0.145 | [+0.047,+0.240] | +0.044 … +0.248 | 0.021 | +0.250 | -0.026 … +0.482 | +0.093 … +0.394 | no | no |
| `iqa_tenengrad` | no-reference | +0.199 | [+0.100,+0.295] | +0.071 … +0.301 | 0.039 | +0.484 | +0.188 … +0.694 | +0.317 … +0.606 | no | **yes** |
| `iqa_saturation` | no-reference | -0.007 | [-0.109,+0.094] | -0.058 … +0.055 | 0.000 | -0.188 | -0.438 … +0.097 | -0.362 … +0.014 | no | no |

## verdict

- max |rho| over every arm x metric x view = **0.484** (`sam2video/iqa_tenengrad/within`)
- every metric below the usable threshold: **YES**
- within-CI reaching the threshold: **4** (`dagrs/iqa_tenengrad`, `greedy/iqa_tenengrad`, `sam2video/iqa_laplacian_var`, `sam2video/iqa_tenengrad`)  ⇒ those metrics may only be described as *point estimate below threshold*, never as *definitely too weak*
- max |rho| upper bound of the `across` **cluster** CI over every arm x metric: **0.488** (threshold 0.6)
- across cluster-CI reaching the threshold: **0**  ⇒ the headline view excludes the threshold even under sequence clustering

