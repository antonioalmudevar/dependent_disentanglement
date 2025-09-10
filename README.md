<h1 align="center">Rethinking Disentanglement under Dependent Factors of Variation</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2408.07016"><img src="https://img.shields.io/badge/arXiv-2408.07016-b31b1b.svg"></a>
  <img src="https://img.shields.io/badge/Python-3.8-blue.svg">
  <img src="https://img.shields.io/badge/PyTorch-1.12.1-green.svg">
</p>

<p align="center">
  <i>Official implementation of <b>Rethinking Disentanglement under Dependent Factors of Variation</b> (Under Review)</i><br>
  <b>Antonio Almudévar</b>, Alfonso Ortega
</p>


---

## 📖 Overview

Most definitions and metrics of disentanglement assume that factors of variation are **statistically independent**.  
However, in realistic scenarios, factors are often dependent (e.g., fruit shape and color).  

In this work, we:

- Introduce four **information-theoretic properties** (factors-invariance, nuisances-invariance, representations-invariance, explicitness).
- Prove that these properties are satisfied **iff** each sub-representation is **minimal and sufficient**.
- Propose new **metrics for minimality and sufficiency**, which quantify disentanglement under dependencies.
- Demonstrate that these metrics better correlate with downstream performance and robustness than existing alternatives.

This repository contains the full experimental framework used in the paper.

<p align="center">
  <img src="docs/abstract.svg" width="550" alt="Method overview">
</p>

---


## 📦 Features

- **Datasets**: dSprites, MPI3D, Shapes3D (easily extendable).
- **Models**: VAE backbones (Burgess, Chen-MLP, Locatello, Montero-small/large).
- **Losses**: β-VAE, AnnealedVAE, FactorVAE, β-TCVAE, AdaGVAE, FactorizedSupport variants.
- **Metrics**: Standard disentanglement scores (MIG, SAP, DCI, Modularity, IRS) + new **Minimality & Sufficiency** metrics.
- **Training & Evaluation Pipelines**:
  - Train models with YAML-based configs.
  - Extract latent representations across correlation regimes.
  - Compute all metrics in a reproducible way.
- **Extensible**: Add new datasets, models, or losses with minimal boilerplate.

---

## 📂 Repository Structure
```
.
├── bin/              # Entry-point scripts (train, predict, calc_metrics)
├── configs/          # YAML configs (data, losses, models, training)
├── constraints/      # YAML constraints (correlations)
├── src/              # Core library
│ ├── datasets/         # Dataset loaders
│ ├── experiments/      # Training, prediction, metrics orchestration
│ ├── losses/           # Disentanglement losses
│ ├── metrics/          # Evaluation metrics
│ └── models/           # Encoder/decoder architectures
└── results/          # Created during runs (checkpoints, predictions, metrics)
```

---

## ⚙️ Installation
```
git clone https://github.com/antonioalmudevar/dependent_disentanglement.git
cd dependent_disentanglement
python3 -m venv venv
source ./venv/bin/activate
python setup.py develop
pip install -r requirements.txt
```

---
## 🚀 Usage

#### 1. Train
```
python bin/train.py dsprites.yaml beta.yaml burgess.yaml unsupervised.yaml \
    --device cuda:0 --epochs 50 --save-epochs 10 --seed 42
```
Outputs are stored under ```results/<data>_<loss>_<model>_<training>/```.

#### 2. Predict
Generate latent representations across correlation regimes:
```
python bin/predict.py dsprites.yaml beta.yaml burgess.yaml unsupervised.yaml \
    --device cuda:0 --seed 42
```

#### 3. Metrics
Evaluate standard metrics + our Minimality & Sufficiency:
```
python bin/calc_metrics.py dsprites.yaml beta.yaml burgess.yaml unsupervised.yaml \
    --device cuda:0 --seed 42
```

---
## 📊 What’s Inside?

- **Datasets**: dSprites, MPI3D, Shapes3D  
- **Models**: Six VAE-based encoders (Burgess, Chen-MLP, Locatello, Montero-small/large)  
- **Losses**: β-VAE, FactorVAE, β-TCVAE, AdaGVAE, HFSVAE  
- **Metrics**:  
  - *Standard*: MIG, SAP, DCI, Modularity, IRS  
  - *Ours*: Minimality & Sufficiency (definition of disentanglement under dependencies)  

---
## 📑 Citation
```
@article{almudevar2025rethinking,
    title={Rethinking Disentanglement under Dependent Factors of Variation},
    author={Almud{\'e}var, Antonio and Ortega, Alfonso},
    journal={arXiv preprint arXiv:2408.07016},
    year={2025}
}
```

---
## 🙏 Acknowledgements

This repository builds upon prior open-source efforts.  

- We use **models, losses and datasets** from [facebookresearch/disentangling-correlated-factors](https://github.com/facebookresearch/disentangling-correlated-factors).  

- We use **metrics** from [ubisoft/ubisoft-laforge-disentanglement-metrics](https://github.com/ubisoft/ubisoft-laforge-disentanglement-metrics).  