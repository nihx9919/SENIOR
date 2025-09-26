# SENIOR

This is the official codebase of IROS 2025 paper: "[SENIOR: Efficient Query Selection and Preference-Guided Exploration in Preference-based Reinforcement Learning](https://2025senior.github.io/)". This codebase is largely originated and modified from [MRN](https://github.com/RyanLiu112/MRN).


## Install
1. Install [Mujoco210](https://github.com/openai/mujoco-py)
2. Installing conda environment
```python
conda env create -f conda_env.yml
conda activate senior
pip install -e .[docs,tests,extra]
pip install git+https://github.com/rlworkgroup/metaworld.git@04be337a12305e393c0caf0cbf5ec7755c7c8feb
```



## Run experiment

Run the following command fot six MetaWorld tasks:

```python
bash run_SENIOR.sh
```

## Citation

```python
@article{ni2025senior,
  title={SENIOR: Efficient Query Selection and Preference-Guided Exploration in Preference-based Reinforcement Learning},
  author={Ni, Hexian and Lu, Tao and Hu, Haoyuan and Cai, Yinghao and Wang, Shuo},
  journal={arXiv preprint arXiv:2506.14648},
  year={2025}
}
```



## Acknowledgments

- We thank the authors of [MRN](https://github.com/RyanLiu112/MRN) and [PEBBLE](https://github.com/rll-research/BPref) for open-sourcing their code, upon which our code is built.
- We thank [MetaWorld](https://github.com/Farama-Foundation/Metaworld) for providing the robot manipulation simulation environment.