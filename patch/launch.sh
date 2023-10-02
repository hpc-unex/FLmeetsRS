#!/bin/bash
module load cuda/10.1.105
module load gcc/7.5.0
# module load openmpi_gcc/4.0.4
module load libfabric/1.11.2
module load ucx/1.9.0
module load hwloc/2.2.0


# .Parse named parameters using etopts
while getopts ":e:c:w:s:i:n:k:t:b:" opt; do
    case $opt in
        e) epochs="$OPTARG";;
        c) communication="$OPTARG";;
        w) workers="$OPTARG";;
        s) seed="$OPTARG";;
        i) imagesize="$OPTARG";;
        n) clients="$OPTARG";;
        k) ccommrounds="$OPTARG";;
        t) totalrounds="$OPTARG";;
        b) batchsize="$OPTARG";;
        \?) echo "Invalid option: -$OPTARG" >&2;;
    esac
done
yaml='.yaml'


declare -a SP=("0SPDec" "1SPFedAvg" "2SPFedOpt" "3SPHierarchical" "4SPVertical" "5SPTurbo")
declare -a MPI=("1MPIFedAvg" "3MPIFedProx" "4MPIFedGKT" "2MPIFedOpt")


if [ "$comm" = "MPI" ]; then
	# Multiple machine with MPI communications
	for alg in "${MPI[@]}"; do
		CONFIG="yamls/"$alg$yaml
		PROCESS_NUM=`expr $WORKER_NUM + 1`
		hostname > mpi_host_file
		
		mpirun -n $PROCESS_NUM --hostfile mpi_host_file --oversubscribe -report-bindings --display-map --mca btl openib,tcp,self --mca btl_openib_if_include mlx5_0 python main.py --algorithm $alg --manualSeed $seed --config $CONFIG --img_size $imagesize --client_num_in_total $clients --client_num_per_round $ccommrounds --comm_round $totalrounds --epochs $epochs --batch_size $batchsize --workers $workers
	done
elif [ "$comm" = "SP" ]; then
	# Single machine with local communications
	for alg in "${SP[@]}"; do
		CONFIG="yamls/"$alg$yaml
		
		python main.py --algorithm $alg --manualSeed $seed --config $CONFIG --img_size $imagesize --client_num_in_total $clients --client_num_per_round $ccommrounds --comm_round $totalrounds --epochs $epochs --batch_size $batchsize
	done

