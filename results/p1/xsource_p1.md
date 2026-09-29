# Cross-source table (xsource_p1)

Aggregation = `sequence_level` -> `collapse_objects` -> equal-weight mean over sequences; CIs are percentile bootstrap over sequences.

| degradation | dagrs | greedy | sam2video |
|---|---|---|---|
| clean | 0.3709 [0.2768, 0.4715] | 0.3495 [0.2632, 0.4411] | 0.6973 [0.6022, 0.7766] |
| fog | 0.2741 [0.2012, 0.3519] | 0.2729 [0.2037, 0.3443] | 0.5797 [0.4714, 0.6790] |
| sensor_noise | 0.3225 [0.2442, 0.4050] | 0.3238 [0.2476, 0.4054] | 0.6311 [0.5216, 0.7296] |
| C1_fog_noise | 0.1993 [0.1526, 0.2515] | 0.2232 [0.1659, 0.2853] | 0.3552 [0.2667, 0.4462] |

## family gap (sam2video - family) and the extra-gap contrast

| family | at clean | degradation | gap | extra vs clean | CI contains 0 |
|---|---|---|---|---|---|
| dagrs | +0.3265 | — | — | — | — |
| dagrs | | fog | +0.3057 [+0.2160, +0.3955] | -0.0208 [-0.0905, +0.0478] | yes |
| dagrs | | sensor_noise | +0.3086 [+0.2265, +0.3946] | -0.0178 [-0.0835, +0.0428] | yes |
| dagrs | | C1_fog_noise | +0.1559 [+0.0864, +0.2291] | -0.1706 [-0.2891, -0.0559] | no |
| greedy | +0.3478 | — | — | — | — |
| greedy | | fog | +0.3068 [+0.2177, +0.3960] | -0.0410 [-0.1046, +0.0208] | yes |
| greedy | | sensor_noise | +0.3073 [+0.2191, +0.3963] | -0.0405 [-0.1013, +0.0115] | yes |
| greedy | | C1_fog_noise | +0.1320 [+0.0759, +0.1900] | -0.2158 [-0.3176, -0.1179] | no |
