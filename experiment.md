# Experiments (ZIP Crack GPU)

This document summarizes the performance experiments and optimizations completed so far for the CUDA PKZIP dictionary attack project.

## Workload

- Target: `pentest.zip` (encrypted PKZIP; entry `pentest.png`)
- Wordlist: `rockyou.best66.txt`
- Expected password (for validation): `P@ssw0rd`

## Measurement Method

- End-to-end timing:
  - `time ./final_project <zip> <wordlist>`
- Per-stage timing breakdown:
  - `ZIPCRACK_TIMING=1 ./final_project <zip> <wordlist>`
  - Stages:
    - `cpu r/p`: CPU dictionary read + packing into GPU input buffers
    - `H2D`: host→device memcpy of password inputs
    - `kernel`: GPU filter kernel execution
    - `D2H`: device→host memcpy of match flags
    - `cpu ver`: CPU verification (try unzip candidates)

Note: once overlap (streams + async copies) is enabled, the per-stage times can overlap and may sum to more than wall-clock time.

## Experiment 1: Batch Size Sweep (end-to-end `time`)

From `time.txt`:

| `BATCH_SIZE` | real (s) | user (s) | sys (s) |
|---:|---:|---:|---:|
| 100,000 | 35.595 | 266.180 | 3.482 |
| 1,000,000 | 37.727 | 100.335 | 2.083 |
| 10,000,000 | 35.487 | 55.356 | 2.149 |
| 100,000,000 | 43.708 | 54.315 | 6.944 |

Observation: `BATCH_SIZE=10,000,000` had the best wall-clock time in this sweep (very similar to 100,000, but with far lower CPU user time). Extremely large batches (100,000,000) hurt wall-clock time, likely due to memory pressure and reduced pipeline efficiency.

## Experiment 2: Stage Breakdown (`ZIPCRACK_TIMING`)

### Run A (baseline stage profile)

```
tested:   710300000
matches:  35750
cpu r/p:  18508.51 ms
cpu ver:  7796.93 ms
H2D:      9725.90 ms
kernel:   174.75 ms
D2H:      322.31 ms
```

Key takeaway: the kernel is a tiny fraction of runtime; the main costs are CPU read/pack, H2D bandwidth, and CPU verification.

### Run B (after removing per-batch `std::vector<std::string>` storage)

Change: stopped storing every password string for a batch; reconstruct candidate passwords from the already-packed host buffer only for `match==1`.

```
tested:   710300000
matches:  35750
cpu r/p:  19290.28 ms
cpu ver:  5507.64 ms
H2D:      8943.27 ms
kernel:   145.00 ms
D2H:      315.37 ms
```

Impact: large reduction in `cpu ver` (fewer allocations/copies), small reduction in kernel time and H2D time.

### Run C (repeat run / post-optimizations)

```
tested:   710279168
matches:  35741
cpu r/p:  19669.77 ms
cpu ver:  4994.65 ms
H2D:      9681.95 ms
kernel:   146.76 ms
D2H:      290.37 ms
```

Impact: further `cpu ver` reduction; other stages are comparable (some run-to-run variance expected).

## Implemented Optimizations (so far)

### GPU-side

- Removed per-thread local copies/large stack usage in the kernel (reduces register spilling and improves occupancy).
- Removed suspicious `l_crc32tab` handling by using a device global CRC table with read-only loads (`__ldg`) instead of copying into shared memory.
- Removed packed struct attributes so GPU loads are naturally aligned (avoids misaligned global memory traffic).

### CPU-side / pipeline

- Faster password packing (avoid `memset` of a full 256B per password).
- CPU verification reads zip entry data via a fixed buffer (avoid allocating full uncompressed file per candidate).
- Overlapped CPU read/pack + GPU work via double-buffering, 2 CUDA streams, async memcpy, and CUDA events.
- Packed passwords “together” to reduce H2D traffic:
  - Host builds a flat `u8` password byte buffer plus `u32` offsets (`count+1`).
  - Kernel reads bytes by offset range per `gid` (variable-length password input).
  - Configurable cap via `ZIPCRACK_MAX_PW_BYTES` (default `64`); longer lines are skipped.

## Next Experiments

- Re-run `ZIPCRACK_TIMING=1` before/after packed password storage on the same workload and report:
  - `H2D` time reduction
  - change in kernel time (may rise slightly due to less-coalesced byte loads)
  - net end-to-end time improvement
- Sweep `ZIPCRACK_MAX_PW_BYTES` (e.g., 32 / 64 / 128) and measure tradeoff between:
  - H2D bandwidth savings
  - skipping-rate for long passwords
- Tighten the GPU-side early rejection to reduce `matches` (fewer CPU unzip attempts).

