# Language-Aligned Waypoint (LAW) Supervision for Vision-and-Language Navigation in Continuous Environments

This repository is the official implementation of [Language-Aligned Waypoint (LAW) Supervision for Vision-and-Language Navigation in Continuous Environments](https://github.com/3dlg-hcvc/LAW-VLNCE) [[Project Website]](https://github.com/3dlg-hcvc/LAW-VLNCE).

In the Vision-and-Language Navigation (VLN) task an embodied agent navigates a 3D environment, following natural language instructions. A challenge in this task is how to handle 'off the path' scenarios where an agent veers from a reference path.
Prior work supervises the agent with actions based on the shortest path from the agent’s location to the goal, but such goal-oriented supervision is often not in alignment with the instruction. Furthermore, the evaluation metrics employed by prior work do not measure how much of a language instruction the agent is able to follow. In this work, we propose a simple and effective language-aligned supervision scheme, and a new metric that measures the number of sub-instructions the agent has completed during navigation.

## Setup

We build on top of VLN-CE codebase. Please follow the set-up instructions and download the data as described in the [VLN-CE codebase](https://github.com/jacobkrantz/VLN-CE). Next, clone this repository and install dependencies from requirements.py:

```bash
git clone git@github.com:3dlg-hcvc/LAW-VLNCE.git
cd LAW-VLNCE
python -m pip install -r requirements.txt
```


## Issues

If you find an issue installing torch-scatter, use the following and replace {cuda-version} with your cuda version and {torch-version} with your installed torch version: 
pip install torch-scatter==latest+{cuda-version} -f https://pytorch-geometric.com/whl/torch-{torch-version}.html

[Refer torch-scatter](https://github.com/rusty1s/pytorch_scatter)

## Usage

The `run.py` script is how training and evaluation is done for all model configurations. Specify a configuration file and a run type as such:

```bash
python run.py \
  --exp-config path/to/experiment_config.yaml \
  --run-type {train | eval | inference}
```

## Training

We follow a similar training regime as [VLN-CE](https://github.com/jacobkrantz/VLN-CE), by first training with teacher forcing on the augmented data and then fine-tuning with Dagger on the original Room-to-Room data.

For our `LAW pano` model, we first train using [`cma_pm_aug.yaml`](https://github.com/3dlg-hcvc/LAW-VLNCE/blob/master/vlnce_baselines/config/paper_configs/law_pano_config/cma_pm_aug.yaml) config. We then evaluate all the checkpoints and select the best performing one, on the nDTW metric. This checkpoint is then fine-tuned using [`cma_pm_da_aug_tune`](https://github.com/3dlg-hcvc/LAW-VLNCE/blob/master/vlnce_baselines/config/paper_configs/law_pano_config/cma_pm_da_aug_tune.yaml) config, by updating the `LOAD_FROM_CKPT` and `CKPT_TO_LOAD` fields.


## Evaluation

The same config may be used for evaluating the models, where `EVAL_CKPT_PATH_DIR` specifies the path of the checkpoint (or a folder for evaluating multiple checkpoints), `STATS_EVAL_DIR` specifies the folder where evaluations are to be saved, `EVAL.SPLIT` specifies dataset split (val-seen or val_unseen), and `EVAL.EPISODE_COUNT` specifies number of episodes to be evaluated.

```bash
python run.py \
  --exp-config vlnce_baselines/config/paper_configs/law_pano_config/cma_pm_aug.yaml \
  --run-type eval
```

## Citing

If you use LAW-VLNCE in your research, please cite the following paper:

## Acknowledgements

We thank Jacob Krantz for the [VLN-CE](https://github.com/jacobkrantz/VLN-CE) codebase, on which we build our repository.
