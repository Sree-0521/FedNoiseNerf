# FederatedNoiseNerf

A federated learning framework for training Satellite Neural Radiance Fields (Sat-NeRF) across distributed clients with heterogeneous and noisy data. Built on [Flower (flwr)](https://flower.dev/) and [Sat-NeRF](https://centreborelli.github.io/satnerf), this project studies how federated NeRF training performs under varying levels of Gaussian noise across clients.

---

## What This Project Does

Standard NeRF training requires a centralized dataset. In real-world satellite imagery scenarios, data is distributed across organizations or geographic partitions and cannot be freely shared. This project addresses that by:

- Splitting the DFC2019 satellite dataset (JAX_068 AOI) across 4 federated clients
- Training Sat-NeRF locally on each client using PyTorch Lightning
- Aggregating model weights via FedAvg using the Flower federated learning framework
- Injecting per-client Gaussian noise at dataset load time to simulate real-world data degradation
- Evaluating the effect of noise heterogeneity on global model quality (PSNR, SSIM, altitude MAE)

---

## Architecture

```
                    ┌─────────────────────────┐
                    │   Flower Server          │
                    │   custom_server.py       │
                    │                          │
                    │  SaveModelStrategy       │
                    │  (FedAvg + checkpointing)│
                    │  num_rounds = 5          │
                    │  fraction_fit = 0.5      │
                    └────────────┬────────────┘
                                 │  FedAvg aggregation
              ┌──────────────────┼──────────────────┐
              │                  │                  │
   ┌──────────▼──────┐ ┌─────────▼───────┐ ┌───────▼─────────┐ ┌──────────────────┐
   │   Client 0      │ │   Client 1      │ │   Client 2      │ │   Client 3       │
   │   GPU 0         │ │   GPU 1         │ │   GPU 0         │ │   GPU 1          │
   │   σ = 0.10      │ │   σ = 0.20      │ │   σ = 0.30      │ │   σ = 0.00       │
   │   (10% noise)   │ │   (20% noise)   │ │   (30% noise)   │ │   (control)      │
   └─────────────────┘ └─────────────────┘ └─────────────────┘ └──────────────────┘
```

Each client runs a full local Sat-NeRF training loop (`client.py`) on its own data partition. The server collects parameters after each round and produces an aggregated global model via weighted FedAvg.

---

## Experiments

### Run 6 — Federated Baseline (4 clients, no noise)

4 clients each trained on a disjoint partition of JAX_068 DFC2019 imagery with no noise injection. Establishes the federated performance baseline.

| Parameter | Value |
|---|---|
| Clients | 4 |
| Rounds | 5 |
| fraction_fit | 0.5 (2 clients/round) |
| Steps per client per round | 12,000 |
| Batch size | 4096 |
| Chunk size | 20,480 |
| Noise | None |

### Run 7 — Federated with Heterogeneous Gaussian Noise

Same setup as Run 6, but RGB pixels are corrupted at dataset load time: `rgb += N(0, σ)`, clipped to [0, 1].

| Client | GPU | Noise Level (σ) |
|---|---|---|
| Client 0 | GPU 0 | 0.10 (10%) |
| Client 1 | GPU 1 | 0.20 (20%) |
| Client 2 | GPU 0 | 0.30 (30%) |
| Client 3 | GPU 1 | 0.00 (control) |

---

## Setup

Three conda environments are used. Only `satnerf` is required for federated training.

```bash
# Main training environment (required)
conda init && bash -i setup_satnerf_env.sh

# Classic satellite MVS evaluation (optional)
conda init && bash -i setup_s2p_env.sh

# RPC bundle adjustment (optional)
conda init && bash -i setup_ba_env.sh
```

If libraries are not found at runtime:

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib
```

---

## Running

### Non-federated baseline (single node)

```bash
bash nonfed_run.sh
```

### Federated Run 6 (no noise, 4 clients)

```bash
bash fed_final_working_custom_run6.sh
```

### Federated Run 7 (heterogeneous Gaussian noise, 4 clients)

```bash
bash fed_final_working_custom_run7.sh
```

Each script starts the Flower server then launches all 4 clients in parallel. Logs are written to:
- `run*_slog.txt` — server log
- `run*_log1.txt` through `run*_log4.txt` — per-client logs

---

## Evaluation

**Generate a DSM from a trained checkpoint:**

```bash
python3 create_satnerf_dsm.py Sat-NeRF <logs_dir>/<run_id> <output_dir> <epoch> \
    <ckpts_dir> <root_dir> <img_dir> <gt_dir>
```

**Run novel view synthesis evaluation:**

```bash
python3 eval_satnerf.py Sat-NeRF <logs_dir>/<run_id> <output_dir> <epoch> val \
    <ckpts_dir> <root_dir> <img_dir> <gt_dir>
```

Metrics reported: PSNR, SSIM, altitude MAE.

---

## Dataset

This project uses the [DFC2019 Track 3](https://ieee-dataport.org/open-access/data-fusion-contest-2019-dfc2019) WorldView-3 satellite imagery dataset, JAX_068 AOI, partitioned across 4 clients (`cl1/`, `cl2/`).

To generate a fresh dataset partition from DFC2019 data:

```bash
python3 create_satellite_dataset.py JAX_068 <dfc_dir> <output_dir>
```

---

## Project Structure

```
FederatedNoiseNerf/
├── client.py                            # Flower federated client (FlowerClient)
├── custom_server.py                     # Flower server with FedAvg + checkpoint saving
├── server.py                            # Basic Flower server (reference)
├── servercopy.py                        # Server variant
├── main.py                              # NeRF_pl Lightning module
├── opt.py                               # Argument parser
├── rendering.py                         # Ray rendering
├── metrics.py                           # PSNR, SSIM, loss functions
├── train_utils.py                       # Training utilities
├── sat_utils.py                         # Satellite geometry utilities
├── dsmr.py                              # DSM registration
├── datasets/                            # Dataset loaders (satellite, blender)
├── models/                              # NeRF, Sat-NeRF, S-NeRF model definitions
├── docs/                                # Paper and diagrams
├── cl1/                                 # Client 1 & 2 data partitions (v1–v4)
├── cl2/                                 # Client 3 & 4 data partitions (v1–v4)
├── fed_final_working_custom_run6.sh     # Run 6: federated baseline (no noise)
├── fed_final_working_custom_run7.sh     # Run 7: heterogeneous Gaussian noise
├── fed_final_working_custom_run[2-5].sh # Earlier experiment runs
├── fed_final_working.sh                 # Initial federated run script
├── nonfed_run.sh                        # Non-federated single-node training
├── run_all.sh                           # Full experiment suite
├── eval_satnerf.py                      # NeRF evaluation
├── eval_satnerf_new.py                  # Extended evaluation
├── eval_s2p.py                          # Classic MVS comparison (S2P)
├── create_satellite_dataset.py          # Dataset creation from DFC2019
├── create_satnerf_dsm.py                # DSM generation from NeRF checkpoint
├── study_depth_supervision.py           # Depth supervision analysis
├── study_solar_interpolation.py         # Solar direction interpolation study
├── setup_satnerf_env.sh                 # Conda env: satnerf (main)
├── setup_s2p_env.sh                     # Conda env: s2p (MVS evaluation)
└── setup_ba_env.sh                      # Conda env: ba (bundle adjustment)
```

---

## Based On

This project builds on [Sat-NeRF](https://centreborelli.github.io/satnerf) by Roger Marí, Gabriele Facciolo, and Thibaud Ehret, presented at CVPR EarthVision Workshop 2022.

```bibtex
@inproceedings{mari2022sat,
  title={{Sat-NeRF}: Learning Multi-View Satellite Photogrammetry With Transient Objects and Shadow Modeling Using {RPC} Cameras},
  author={Mar{\'i}, Roger and Facciolo, Gabriele and Ehret, Thibaud},
  booktitle={CVPR Workshops},
  year={2022}
}
```
