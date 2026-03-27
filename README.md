# 2025-ML-Final-Project# 2025-ML-Final

This is the Readme file for how to run our code. This work contains different source of github repository, including trainind phase improvment and inference phase improvement. Therefore, we will seperate different parts to introduce how to run our codes.

This repository contains the official implementation of our 2025 Machine Learning Final Project.Our work introduces improvements to both the **training phase** and the **inference phase** of flow-based generative models. Specifically, we propose:

- **Directional Flow Decomposition** — A training strategy that decomposes the ground-truth flow into *energy* and *direction* components, enabling more stable learning and improved angular accuracy in the predicted vector fields.
- **Non-Uniform Timestep Sampling** — An inference-time technique that accelerates ODE integration by allocating integration steps more efficiently along the trajectory.

Because this project integrates components from multiple external repositories and adds several new modules, we provide a clear guide on how to set up the environment, reproduce our experiments, and run both training and sampling pipelines.

The sections below describe the overall structure of the project and how to execute each component.

Cloning the repository and starts run the code.

```
git clone https://github.com/Samuel20040521/2025-ML-Final.git
cd 2025-ML-Final/
```

## Running non-uniform timestep sampling

#### Meanflow


We run the non-uniform timestep sampling using Meanflow and conditional flow matching. We will introduce how to run on the Meanflow github repo first.

Install the environment

```
conda create -n meanflow python=3.10
conda activate meanflow
cd MeanFlow/
pip install -r requirements.txt

```

Download the checkpoint [here](https://drive.google.com/drive/folders/1oWt6tdm5WIeVaZnBuUVheKIG3cNDffl9) with used for cifar10.

```
python3 evaluate_fid_timesteps.py --ckpt "Your checkpoint place" --output_dir "output_directory" --batch_size "batchsize" --num_fid_samples "numbers of sample for FID computation"
```

We will give the sample commands to run our code

```
python3 evaluate_fid_timesteps.py --ckpt unet_cifar10_meanflow_100k_ema.py --output_dir result/ --batch_size 500 --num_fid_samples 50000
```

#### Conditional flowmatching

To run the analysis on CFM, please follow the below commands

Install the environment.

```
cd conditional-flow-matching
conda create -n torchcfm python=3.10
conda activate torchcfm
pip install -r requirements.txt
pip install -e .
cd examples/images/cifar10
```

Download the chekpoint

[icfm](https://github.com/atong01/conditional-flow-matching/releases/download/1.0.4/cfm_cifar10_weights_step_400000.pt)

[otcfm](https://github.com/atong01/conditional-flow-matching/releases/download/1.0.4/otcfm_cifar10_weights_step_400000.pt)

[fm](https://github.com/atong01/conditional-flow-matching/releases/download/1.0.4/fm_cifar10_weights_step_400000.pt)

[our icfm](https://drive.google.com/file/d/1UCve4zEHVSuDg1dQyRW7pqYJe3ageMS7/view?usp=drive_link)

To run the comparison for different timesteps using icfm, otcfm. Please run the following commands

```
python3 analysis_compute_fid_nonuniform.py --ckpt "Your checkpoint path" --num_fid_samples "number of generated images for FID scores" --
```

We will give the example commands to run our code.

```
python3 analysis_compute_fid_nonuniform.py --ckpt results/icfm/icfm_cifar10_weights_step_400000.py --num_fid_samples 50000
```

## Running Angle-Energy Decomposition Architecture Training

To run the Angle-Energy Decomposition Architecture Training, please follow the below commands.

Ensure you are in the `conditional-flow-matching/examples/images/cifar10` directory and have the `torchcfm` environment activated (as described in the Conditional flowmatching section).

```bash
cd conditional-flow-matching/examples/images/cifar10
```

Run the training script:

```bash
python3 train_cifar10_decompose.py --model icfm --output_dir ./results_decompose --lambda_vec 1.0 --lambda_energy 0.1 --lambda_shape 1.0
```

**Arguments:**
- `--model`: Flow matching model type (`otcfm`, `icfm`, `fm`, `si`). Default: `otcfm`.
- `--output_dir`: Directory to save results and checkpoints.
- `--lambda_vec`: Weight for the total reconstruction loss (vector field MSE).
- `--lambda_energy`: Weight for the energy (scalar) loss.
- `--lambda_shape`: Weight for the shape (spatial cosine) loss.
- `--total_steps`: Total training steps. Default: 400001.

### Conda environment setup

```bash
# clone project
cd conditional-flow-matching

# [OPTIONAL] create conda environment
conda create -n torchcfm python=3.10
conda activate torchcfm

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt

# install torchcfm
pip install -e .

