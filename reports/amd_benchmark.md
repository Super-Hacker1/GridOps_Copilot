# AMD ROCm FRA benchmark

Generated: 2026-07-11T11:24:13.443406+00:00

- Device: `AMD gfx1100 GPU (PCI 1002:744b; 96 compute units)`
- PyTorch: `2.9.1+gitff65f5b`
- HIP runtime: `7.2.53211-e1a6bc5663`
- Evaluation scope: `grouped synthetic holdout (not field generalization)`
- Grouped synthetic holdout accuracy: `1.0000`
- Grouped synthetic holdout macro F1: `1.0000`
- CPU batch latency (32 curves): `3.146 ms`
- AMD GPU batch latency (32 curves): `0.280 ms`
- Observed speedup: `11.24x`
- Artifact SHA-256: `3bcac0c65d3729b31271d2048c2b9c52411f088f94991c22f2db8bd4df1cd633`

Train, validation, and test use disjoint seeded synthetic scenario/prototype groups.
These grouped synthetic holdout metrics do not estimate field generalization. The model
is not field validated and provides decision support only; every result requires qualified
engineer confirmation.
