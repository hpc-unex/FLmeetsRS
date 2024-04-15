# FLmeetsRS
<img src="https://github.com/hpc-unex/FLmeetsRS/assets/36038967/d8475c26-5968-4386-a7e6-c01b624d9ef2" alt="Texto alternativo" style="width:600px;height:300px;">

## Requirements (bold is the tested version):

torch >= **1.12.1**

cuda >= **10.2**

mpi compatible with torch+cuda version >= **4.0.4**

fedml (https://github.com/FedML-AI/FedML) **v0.7.331**

## Data types (This study focus on high-resolution optical images)

<img src="https://github.com/hpc-unex/FLmeetsRS/assets/36038967/a4018816-e3af-496d-a081-973ab73cdf31" style="width:350px;height:350px;">



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
cd patch
sh execute_patch.sh
```

RSICB256 dataset should be collected in ~/FLmeetsRS/FedML/fedml_data/RSICB256 as:
```
.
├── construction land/
│   ├── city_building/
│   ├── container/
│   ├── residents/
│   └── storage_room/
├── cultivate land/
│   ├── bare_land/
│   ├── dry_farm/
│   └── green_farmland/
├── other land/
│   ├── desert/
│   ├── mountain/
│   ├── sandbeach/
│   └── snow_mountain/
├── other objects/
│   ├── airplane/
│   ├── pipeline/
│   └── town/
├── transportation/
│   ├── airport_runway/
│   ├── avenue/
│   ├── bridge/
│   ├── crossroads/
│   ├── highway/
│   ├── marina/
│   └── parkinglot/
├── water area/
│   ├── coastline/
│   ├── dam/
│   ├── hirst/
│   ├── lakeshore/
│   ├── river/
│   ├── sea/
│   └── stream/
└── woodland/
    ├── artificial_grassland/
    ├── forest/
    ├── mangrove/
    ├── river_protection_forest/
    ├── sapling/
    ├── shrubwood/
    └── sparse_forest/
```
### Run Code

```
# With 10 nodes 20 clients (MPI)
sbatch --nodes=10 -p volta --wait-all-nodes=1 -t 01:00:00 --gpus-per-node=2 ./launch.sh -e 10 -c MPI -w 20 -s 123 -i 256 -n 20 -k 10 -t 100 -b 64

# With 1 nodes 20 clients (SP)
sbatch --nodes=1 -p volta --wait-all-nodes=1 -t 01:00:00 --gpus-per-node=2 ./launch.sh -e 10 -c SP -w 4 -s 123 -i 256 -n 20 -k 10 -t 100 -b 64
```

## Some Results

<img src="https://github.com/hpc-unex/FLmeetsRS/assets/36038967/99a4e49f-3290-4921-b54f-d5d06c0935a5" alt="Texto alternativo" style="width:470px;height:190px;">



