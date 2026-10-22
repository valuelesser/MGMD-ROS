# Summary

FPS 为报告项（RTX 5060 Ti 相对分析；绝对 FPS 不上论文对照）。Tier B 协议复现。

| run | config | backend | windows | eff FPS | valid% | recall@100 | precision@100 | RMSEp | RMSEv |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| D1_matched | D1_no_overlap | jax | 100 | 2.82 | 100.00 | 100.00 | 100.00 | 2.84 | 0.36 |
| D1_mf_1.6 | D1_no_overlap | jax | 60 | 2.88 | 100.00 | 100.00 | 100.00 | 3.01 | 0.35 |
| D1_mf_3.5 | D1_no_overlap | jax | 60 | 2.92 | 100.00 | 100.00 | 100.00 | 2.66 | 0.34 |
| D1_min0.3 | D1_no_overlap | jax | 60 | 1.04 | 0.00 | 0.00 | nan | N/A | N/A |
| D1_no_overlap | D1_no_overlap | jax | 80 | 2.46 | 0.00 | 0.00 | 0.00 | N/A | N/A |
| D2_matched | D2_no_overlap | jax | 120 | 2.80 | 100.00 | 100.00 | 100.00 | 17.13 | 0.46 |
| D3_accel | D3_overlap | jax | 100 | 0.31 | 70.00 | 0.00 | 0.00 | N/A | N/A |
| D3_accel4d | D3_overlap | jax | 100 | 10.19 | 0.00 | 0.00 | 0.00 | N/A | N/A |
| D3_accel4d_ms | D3_overlap | jax | 100 | 10.24 | 0.00 | 0.00 | 0.00 | N/A | N/A |
| D3_matched | D3_overlap | jax | 100 | 0.31 | 70.00 | 0.00 | 0.00 | N/A | N/A |
| smoke_D1 | D1_no_overlap | numpy | 12 | 6.65 | 0.00 | 0.00 | nan | N/A | N/A |
| smoke_seq000 | D1_no_overlap | jax | 5 | 0.97 | 0.00 | 0.00 | nan | N/A | N/A |
| smoke_seq000_default | D1_no_overlap | jax | 3 | 1.03 | 0.00 | 0.00 | nan | N/A | N/A |
