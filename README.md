

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



