# SENIOR

This is the official codebase of IROS 2025 paper: "[SENIOR: Efficient Query Selection and Preference-Guided Exploration in Preference-based Reinforcement Learning](https://2025senior.github.io/)". This codebase is largely originated and modified from [MRN](https://github.com/RyanLiu112/MRN).



## Install

### Install Mujoco

```python
sudo apt update
sudo apt install unzip gcc libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf libegl1 libopengl0
sudo ln -s /usr/lib/x86_64-linux-gnu/libGL.so.1 /usr/lib/x86_64-linux-gnu/libGL.so
wget https://www.roboti.us/download/mujoco200_linux.zip -P /tmp
unzip /tmp/mujoco200_linux.zip -d ~/.mujoco
wget https://www.roboti.us/file/mjkey.txt -P /tmp
mv /tmp/mjkey.txt ~/.mujoco/
```



### Install conda environment

```python
conda env create -f conda_env.yml
conda activate senior
pip install -e .[docs,tests,extra]
pip install git+https://github.com/rlworkgroup/metaworld.git@04be337a12305e393c0caf0cbf5ec7755c7c8feb
pip install termcolor pybullet
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