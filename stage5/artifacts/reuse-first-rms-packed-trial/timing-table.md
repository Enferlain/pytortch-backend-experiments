# Same-process CUPTI timing (microseconds)

Pooled medians of96 samples/path/case, six fully permuted rounds.

| Graph | Ordinary | Frozen wqc | Packed | Packed wins vs ordinary / wqc |
| --- | ---: | ---: | ---: | ---: |
| corpus_float32_64_1024 | 14.5640 | 23.1460 | 22.2540 | 0/6 / 6/6 |
| corpus_float16_64_1024 | 15.2335 | 29.3285 | 29.2810 | 0/6 / 4/6 |
| corpus_bfloat16_64_1024 | 15.1770 | 30.3810 | 30.3155 | 0/6 / 4/6 |
| vector_row_tail_float32_16_1024 | 14.9870 | 22.9320 | 22.2150 | 0/6 / 6/6 |
| vector_row_tail_float32_65_1024 | 14.4595 | 23.1400 | 22.3590 | 0/6 / 6/6 |
| vector_row_tail_float16_16_1024 | 14.7560 | 28.9055 | 28.9925 | 0/6 / 1/6 |
| vector_row_tail_float16_65_1024 | 15.2220 | 29.2910 | 29.2875 | 0/6 / 3/6 |
| vector_row_tail_bfloat16_16_1024 | 13.9650 | 30.2150 | 29.9865 | 0/6 / 6/6 |
| vector_row_tail_bfloat16_65_1024 | 15.1725 | 30.2775 | 30.1730 | 0/6 / 4/6 |

## Isolated complete round records

Primary FP32 crosses execution/clock states: pooled cost unqualified. No prefix selection.

### corpus_float32_64_1024

| Round | Ordinary | Frozen wqc | Packed |
| --- | ---: | ---: | ---: |
| 0 | 2.3530 | 3.5500 | 3.4505 |
| 1 | 2.3740 | 3.5570 | 3.4590 |
| 2 | 2.3740 | 3.5460 | 3.4560 |
| 3 | 12.6510 | 16.7120 | 3.4590 |
| 4 | 14.5010 | 23.1670 | 22.2910 |
| 5 | 14.5410 | 23.1510 | 22.3175 |

### corpus_float16_64_1024

| Round | Ordinary | Frozen wqc | Packed |
| --- | ---: | ---: | ---: |
| 0 | 15.1945 | 29.2825 | 29.1845 |
| 1 | 15.2125 | 29.4520 | 29.2400 |
| 2 | 15.2090 | 29.4490 | 29.1110 |
| 3 | 15.1940 | 29.3845 | 29.3730 |
| 4 | 15.2530 | 29.3905 | 29.4225 |
| 5 | 15.2360 | 29.4160 | 29.3415 |

### corpus_bfloat16_64_1024

| Round | Ordinary | Frozen wqc | Packed |
| --- | ---: | ---: | ---: |
| 0 | 15.4185 | 30.2480 | 30.2190 |
| 1 | 15.1900 | 30.8000 | 28.8695 |
| 2 | 15.1040 | 30.3470 | 30.4170 |
| 3 | 15.2185 | 30.4030 | 30.4160 |
| 4 | 14.6760 | 30.2405 | 30.4025 |
| 5 | 15.1260 | 30.4975 | 30.2835 |
