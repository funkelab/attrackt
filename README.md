<h2 align="center">Attrackt</h2>

- **[Introduction](#introduction)**
- **[Installation](#installation)**
- **[Getting Started](#getting-started)**
- **[Citation](#citation)**
- **[Issues](#issues)**


## Introduction
This repository hosts the version of the code used for the **[publication](https://openaccess.thecvf.com/content/ICCV2025W/BIC/html/Lalit_An_Investigation_of_Unsupervised_Cell_Tracking_and_Interactive_Fine-Tuning_ICCVW_2025_paper.html)** titled ***An Investigation of Unsupervised Cell Tracking and Interactive Fine-Tuning***. This work was accepted to the Bio-Image Computing Workshop at the International Conference for Computer Vision (ICCV), 2025.

We refer to the proposed loss described in the publication as **Attrackt** - Using Attrackt, one can link or track instance segmentations in 2D or 3D microscopy images in an unsupervised fashion i.e. requiring no ground truth labels during training of the deep neural network. Additionally, we provide a strategy for interactive fine-tuning of the trained model using user-provided corrections.


## Installation

### 1. Clone the Repository
First, clone the repository and navigate to the project directory:

```bash
git clone https://github.com/funkelab/attrackt
cd attrackt
```

### 2. Set Up the Conda Environment
Create and activate a dedicated Conda environment named `attrackt`:

```bash
conda create -n attrackt python==3.10
conda activate attrackt
```

### 3. Install Dependencies
Install the required packages, including `ilpy`, `scip`, and all dependencies:

```bash
conda install -c conda-forge -c funkelab -c gurobi ilpy
conda install -c conda-forge scip
```

### 4. Install Attrackt in Editable Mode
Finally, install the attrackt repository in editable mode (useful for development):

```bash
pip install -e .
```

Full Setup Summary:

```bash
git clone https://github.com/funkelab/attrackt.git
cd attrackt
conda create -n attrackt python=3.10
conda activate attrackt
conda install -c conda-forge -c funkelab -c gurobi ilpy
conda install -c conda-forge scip
pip install -e .
```

## Getting Started

Try out the examples available **[here](https://funkelab.github.io/attrackt_experiments)**.

## Citation


If you find our work useful in your research, please consider citing:

```bibtex
@InProceedings{Lalit_2025_ICCV,
    author    = {Lalit, Manan and Funke, Jan},
    title     = {An Investigation of Unsupervised Cell Tracking and Interactive Fine-Tuning},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops},
    month     = {October},
    year      = {2025},
    pages     = {5792-5800}
}
```

## Issues

If you encounter any problems, please **[file an issue](https://github.com/funkelab/attrackt/issues)** along with a description.

