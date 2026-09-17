# 🤖 # Humanoid Gait Optimization using Quadratic Programming

<p align="center">
  <b>Optimization-based gait planning for stable and efficient humanoid locomotion</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python">
  <img src="https://img.shields.io/badge/Optimization-QP-orange">
  <img src="https://img.shields.io/badge/Robotics-Humanoid-green">
  <img src="https://img.shields.io/badge/Status-Research%20Project-purple">
</p>

---

## Overview

This repository presents a **humanoid gait optimization framework based on classical Quadratic Programming (QP)**.

The objective is to generate stable walking trajectories while satisfying the physical constraints of humanoid locomotion.

The optimization framework considers:

- Robot dynamics
- Balance and stability
- Support polygon constraints
- Contact constraints
- Trajectory tracking
- Optimization of gait behavior

The project combines both the **mathematical formulation** of the gait-planning problem and its **Python implementation**.

---

## Research Objective

The main objective is to formulate humanoid gait generation as a constrained optimization problem:

\[
\boxed{
\text{Find the optimal walking trajectory while respecting robot dynamics and stability constraints}
}
\]

The Quadratic Program minimizes tracking and stability errors while maintaining physically feasible motion.

---

## Framework

```text
Desired Walking Motion
        │
        ▼
┌───────────────────┐
│ Gait / Footstep   │
│     Planning      │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ Reference COM /   │
│ DCM Trajectories  │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ Quadratic Program │
│       (QP)        │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ Constraints       │
│ • Dynamics        │
│ • Contact         │
│ • Support Polygon │
│ • Stability       │
└─────────┬─────────┘
          │
          ▼
   Optimized Gait




Gait-Optimization/
│
├── 📄 Documentation.pdf
│   └── Mathematical formulation and theoretical background
│
├── 🐍 locomotion_test_prime.py
│   └── Locomotion testing and simulation
│
├── 🐍 themis_gait_prime.py
│   └── Main gait generation framework
│
├── 🐍 themis_wbc_qp.py
│   └── Quadratic Programming / Whole-Body Control implementation
│
└── 📖 README.md
