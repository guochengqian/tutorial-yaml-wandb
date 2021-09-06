#!/usr/bin/env bash
# Using #!/usr/bin/env NAME makes the shell search for the first match of NAME in the $PATH environment variable.
# make sure command is : source init.sh

# install anaconda3.
#cd ~/
#wget https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh
#bash Anaconda3-2019.10-Linux-x86_64.sh
# source ~/.bashrc

conda deactivate
export TORCH_CUDA_ARCH_LIST="3.5;6.1;6.2;7.0;7.5"   # v100: 7.0; 2080ti: 7.5; titan xp: 6.1

# uncomment if you are using slurm
#module purge
#module load cuda/10.1.243
#module load gcc

# make sure system cuda version is the same with pytorch cuda
# follow the instruction of PyTorch Geometric: https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
conda create -n wandb -y
conda activate wandb

# make sure pytorch version >=1.4.0
conda install -y pytorch=1.7.0 torchvision cudatoolkit=10.1 python=3.7 -c pytorch -y

# install useful modules
pip install tensorboard pyyaml wandb multimethod
