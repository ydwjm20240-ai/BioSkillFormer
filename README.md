# BioSkillFormer: A Transformer-Based Multimodal Fusion Framework for Ordinal Motor Skill-Level Assessment

Official PyTorch implementation of **BioSkillFormer** for multimodal ordinal motor skill classification and biomechanical interpretability analysis.

## Overview
BioSkillFormer integrates synchronized multi-sensor streams collected during athletic actions (pose kinematics, surface EMG, plantar foot pressure, eye-tracking gaze, appearance video, and continuous wavelet scalograms) along with 18 biomechanical indicators into a unified ordinal Transformer.

### Key Capabilities
- **7-Stream Multimodal Encoders:** Tailored 1D TCNs, ResNet-18 visual appearance, CWT 2D patch-embedding, and MLP indicator projection.
- **Reliability-Aware Attention:** Dynamic gating mechanism penalizing unreliable, noisy, or dropped sensor channels via attention bias matrix $R$.
- **Ordinal Sequential Ranking Head:** Formulates skill assessment as sequential binary probabilities $P(y > 0)$ and $P(y > 1)$ to penalize severe cross-tier misclassifications.
- **Biomechanical Grounding:** Auxiliary regression head ensuring latent embeddings correlate with joint range of motion, movement smoothness (SPARC), and muscle firing synchrony.

## Installation
```bash
git clone https://github.com/ydwjm20240-ai/BioSkillFormer.git
cd BioSkillFormer
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
