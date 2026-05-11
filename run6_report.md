# Federated Sat-NeRF Run 6 — Training Report

## Experiment Configuration

| Parameter | Value |
|-----------|-------|
| Model | Sat-NeRF |
| Scene | JAX_068 (DFC2019 Track3) |
| Federation Framework | Flower (flwr 1.5.0) |
| Strategy | FedAvg (SaveModelStrategy) |
| Number of FL Rounds | 5 |
| Total Clients | 4 |
| Clients Selected Per Round | 2 (fraction_fit = 0.5) |
| Max Train Steps Per Round | 12,000 |
| Batch Size | 4,096 |
| Chunk Size | 20,480 |
| Python Environment | fedsatnerf (torch 2.0.1+cu117, flwr 1.5.0, pytorch_lightning 1.3.7) |

## Hardware

| GPU | Model | Memory |
|-----|-------|--------|
| GPU 0 | NVIDIA RTX A6000 | 49,140 MiB |
| GPU 1 | NVIDIA RTX A6000 | 49,140 MiB |

**GPU Assignment:**
- Client 0 (cl1/client1v4) → GPU 0
- Client 1 (cl2/client2v4) → GPU 1
- Client 2 (cl1/client1v3) → GPU 0
- Client 3 (cl2/client2v3) → GPU 1

## Client Datasets

| Client | Data Directory | Images |
|--------|---------------|--------|
| Client 0 | cl1/client1v4 — JAX_068 | 8 training images (003,004,005,007,009,016,018,022) |
| Client 1 | cl2/client2v4 — JAX_068 | 9 training images (001,002,006,010,011,015,016,020,...) |
| Client 2 | cl1/client1v3 — JAX_068 | 8 training images (001,002,006,007,012,013,016,018,019,020) |
| Client 3 | cl2/client2v3 — JAX_068 | 9 training images (002,003,004,005,009,...) |

## Federated Round Results

| Round | Completed (UTC-5) | Duration | Results | Failures |
|-------|-------------------|----------|---------|----------|
| Round 1 | 2026-04-17 05:04 | ~1h 41m | 2/2 ✅ | 0 |
| Round 2 | 2026-04-17 06:46 | ~1h 42m | 2/2 ✅ | 0 |
| Round 3 | 2026-04-17 08:27 | ~1h 40m | 2/2 ✅ | 0 |
| Round 4 | 2026-04-17 10:07 | ~1h 40m | 2/2 ✅ | 0 |
| Round 5 | 2026-04-17 11:47 | ~1h 40m | 2/2 ✅ | 0 |

**Total Training Time:** 8 hours 24 minutes (30,262 seconds)  
**FL Start:** 2026-04-17 03:23:33  
**FL End:** 2026-04-17 11:47:55  

## Final Client Metrics (After 5 Rounds)

| Client | Dataset | Final Train PSNR (dB) | Final Loss |
|--------|---------|----------------------|------------|
| Client 0 | cl1/client1v4 | 28.00 | 0.189 |
| Client 1 | cl2/client2v4 | 26.80 | 0.209 |
| Client 2 | cl1/client1v3 | 27.80 | 0.179 |
| Client 3 | cl2/client2v3 | 28.00 | 0.196 |
| **Average** | | **27.65** | **0.193** |

## PSNR Progression — Client 0 (Selected Rounds)

Client PSNR improved across federated rounds as the global model aggregated knowledge:

| FL Round | Train PSNR (dB) |
|----------|----------------|
| Round 1 (Epoch 0) | ~19.5 |
| Round 2 (Epoch 2) | ~23.5 |
| Round 3 (Epoch 4) | ~25.9 |
| Round 4 (Epoch 6) | ~27.0 |
| Round 5 (Epoch 8+) | **28.00** |

## Saved Global Model Weights

| File | Description |
|------|-------------|
| `round-1-weights.ckpt` | Global FedAvg weights after round 1 |
| `round-2-weights.ckpt` | Global FedAvg weights after round 2 |
| `round-3-weights.ckpt` | Global FedAvg weights after round 3 |
| `round-4-weights.ckpt` | Global FedAvg weights after round 4 |
| `round-5-weights.ckpt` | Global FedAvg weights after round 5 (final) |

## Client Checkpoints

Each client saved best-epoch checkpoints (monitored by val/psnr):
- `cl1/checkpoints/2026-04-17_03-23-33_JAX_068_ds1_sat-nerf/` — Clients 0 & 2
- `cl2/checkpoints/2026-04-17_03-23-33_JAX_068_ds1_sat-nerf/` — Clients 1 & 3

## Visual Outputs

Validation images saved per epoch under each client's log directory:
- `cl1/logs/2026-04-17_03-23-33_JAX_068_ds1_sat-nerf/val/` — rgb, depth, albedo, beta, sun, sky, dsm
- `cl2/logs/...` — same structure

## Technical Notes

- All 5 rounds completed with **0 failures** — fully successful federated run
- `num_workers=0` used in DataLoader to prevent CUDA fork segfaults when two clients share a GPU
- Server uses FedAvg aggregation; global weights averaged after each round from 2 selected clients
- Compared to run5 (2 clients, 75,000 steps, 20 hrs): run6 uses 4 clients, 12,000 steps/round, 8h 24m total
- SSH disconnections do not affect training — launched with `nohup`
