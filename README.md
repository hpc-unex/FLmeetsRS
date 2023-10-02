# FLmeetsRS

![RS-FL (17)](https://github.com/hpc-unex/FLmeetsRS/assets/36038967/d8475c26-5968-4386-a7e6-c01b624d9ef2)

Requirements (bold is the tested version):
torch >= **1.12.1**
cuda >= **10.2**
mpi compatible with torch+cuda version >= **4.0.4**
fedml (https://github.com/FedML-AI/FedML) **v0.7.331**


## Getting Started
### Installation

```
git clone https://github.com/hpc-unex/FLmeetsRS.git
cd FLmeetsRS
> Install previous requirements
git clone https://github.com/FedML-AI/FedML.git
cd FedML
git checkout 7aaa278d15f8852e1e2af49bf8baf3666151192f
cd ..
sh patch_and_install.sh
```

### Run Code

```
# With 2 nodes 20 clients (MPI)
sbatch --nodes=2 -p volta --wait-all-nodes=1 -t 01:00:00 --gpus-per-node=2 ./launch.sh -e 10 -c MPI -w 4 -s 123 -i 256 -n 20 -k 10 -t 100 -b 64

# With 2 nodes 20 clients (SP)
sbatch --nodes=2 -p volta --wait-all-nodes=1 -t 01:00:00 --gpus-per-node=2 ./launch.sh -e 10 -c SP -w 4 -s 123 -i 256 -n 20 -k 10 -t 100 -b 64
```

## Some Results

```
![Captura desde 2023-10-02 16-16-46](https://github.com/hpc-unex/FLmeetsRS/assets/36038967/39a398eb-588d-406f-bb65-3cb2d57cbdf8)

```

