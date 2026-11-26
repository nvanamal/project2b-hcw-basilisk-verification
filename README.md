# Cross-Model Verification of Autonomous Rendezvous Using HCW and Basilisk Dynamics Under Navigation and Actuation Uncertainty

**Nagendrababu Vanamala, Ph.D.**

Development-stage reproducibility repository for Project 2B. The manuscript reports deterministic completion times of 2784 s (HCW) and 2786 s (Basilisk), and 100/100 successful simulations per branch with zero modeled keep-out violations. These are reported results, not independently reproduced by this repository yet.

## Scope
700 km circular orbit, HCW-compensated PD control, matched navigation and actuation uncertainty, paired Monte Carlo analysis. No flight validation or learned-policy improvement is claimed.

## Provenance
New scripts are labeled reconstructions. Original simulator source, seed-level CSVs, logs and exact integration/termination ordering must be recovered before archival release. Seed 8 was rerun following a worker file-lock error; retain original incident evidence. No synthetic data may substitute for original results.

## Reproduction
See configuration and analysis scripts as they become available. Reported environment: Python 3.8.0, bsk 2.9.1. Exact dependency freeze pending.
