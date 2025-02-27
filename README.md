# How to Probe
The official implementation of our work How to Probe: Simple Yet Effective Techniques for Improved Post-hoc Explanations @ ICLR 2025 ([OpenReview Link](https://openreview.net/pdf?id=57NfyYxh5f)).

The code is divided into 3 steps: 
* Pre-training SSL and supervised model backbones.
* Probing: training linear or MLP probes on top of the frozen pre-trained SSL or supervised models.
* Evaluating Model Attributions: we support GridPG (Grid Pointing Game) and EPG (Energy Pointing Game) to evaluate the class-specificity of different post-hoc heatmap-based explanation methods.

## Step 1: Pre-training
See folder `how-to-probe/pretraining/` and `README.md`.

## Step 2: Probing
See folder `how-to-probe/probing/` and `README.md`.

## Step 3: Evaluating Attributions
See folder `how-to-probe/evaluate_attributions/` and `README.md`.
