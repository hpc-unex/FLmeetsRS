import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import random
import mpi4py.rc
mpi4py.rc.threads = True
from mpi4py import MPI
from algorithms import *
#from algorithms.mpi import *

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Random seed
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(args.manualSeed)
        #cudnn.deterministic = True
        #cudnn.benchmark = False
        
        
    #SINGLE PROCESS             
    if args.algorithm in ["0SPDec","1SPFedAvg","2SPFedOpt","2SPFedNova","3SPHierarchical", "4SPVertical","5SPTurbo"]:
        launchSP(args)
        

    #MPI EXEC   
    elif args.algorithm in ["0MPIDec", "1MPIFedAvg", "2MPIFedOpt", "3MPIFedProx","4MPIFedNAS_search", "4MPIFedGKT","4MPIByzantine_atk", "4MPIByzantine"]:
        comm = MPI.COMM_WORLD
        if comm.Get_rank() == 0:
            f = open("/home/smoreno/fedml/FedML/outputs/"+args.algorithm+".txt", "a+")
            f.write("#Epoch\t#CommRound\t#Clients\t#TrainAcc\t#TrainLoss\t#TestAcc\t#TestLoss\n")
            f.close()
        launchMPI(args)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
    
    parser.add_argument('--algorithm', type=str, default='0SPDec', \
                                choices=['0SPDec', '1SPFedAvg', '2SPFedOpt', '2SPFedNova', '3SPHierarchical', '4SPVertical', '5SPTurbo','0MPIDec', '1MPIFedAvg', '2MPIFedOpt', '3MPIFedProx',"4MPIByzantine_atk","4MPIByzantine", "4MPIFedNAS_search", "4MPIFedGKT"], \
                                help='Algorithm name to test with')
    parser.add_argument('--manualSeed', default=1111, type=int, help='random seed')
    parser.add_argument('--config', default='', type=str, help='path yaml')
    parser.add_argument('--workers', default=1, type=int, help='number of workers')
    parser.add_argument('--img_size', default=256, type=int, help='image size')
    parser.add_argument('--client_num_in_total', default=50, type=int, help='total clients')
    parser.add_argument('--client_num_per_round', default=20, type=int, help='clients communications per round')  
    parser.add_argument('--comm_round', default=40, type=int, help='rounds of communication') 
    parser.add_argument('--epochs', default=10, type=int, help='epochs') 
    parser.add_argument('--batch_size', default=32, type=int, help='images batch_size') 
    

    args = parser.parse_args()
    main(args)
    
