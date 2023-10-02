import argparse
import logging
import os
import random
import socket
import sys
import yaml
import fedml
import torch
import torch.nn as nn
from fedml.simulation import SimulatorMPI
from fedml.model.cv.darts import genotypes
from fedml.model.cv.darts.model import NetworkCIFAR
from fedml.model.cv.darts.model_search import Network

from fedml.model.cv.resnet_gn import resnet18
from fedml import FedMLRunner




def launchMPI(params):
    # init FedML framework
    args = fedml.init(params.manualSeed, params.config, params.algorithm)
    args.client_num_in_total = params.client_num_in_total
    args.client_num_per_round = params.client_num_per_round
    args.comm_round = params.comm_round
    args.batch_size = params.batch_size
    args.epochs = params.epochs
    # init device
    device = fedml.device.get_device(args)
    # load data
    dataset, output_dim = fedml.data.load(args)

    # load model (the size of MNIST image is 28 x 28)
    
    #if algorithm == "4MPIFedNAS_search":
        #criterion = nn.CrossEntropyLoss()
        #model = Network(args.init_channels, output_dim, args.layers, criterion)
    #elif algorithm == "4MPIFedNAS_train":
        #genotype = genotypes.FedNAS_V1
        #model = NetworkImageNet(args.init_channels, output_dim, args.layers, args.auxiliary, genotype)
    #else:
    model = fedml.model.create(args, output_dim, params.img_size)

    # start training
    simulator = SimulatorMPI(args, device, dataset, model)
    simulator.run()


def launchSP(params):

    # init FedML framework
    args = fedml.init(params.manualSeed, params.config, params.algorithm)
    args.client_num_in_total = params.client_num_in_total
    args.client_num_per_round = params.client_num_per_round
    args.comm_round = params.comm_round
    args.batch_size = params.batch_size
    args.epochs = params.epochs
    

    # init device
    device = fedml.device.get_device(args)

    # load data
    dataset, output_dim = fedml.data.load(args)
    # load model
    model = fedml.model.create(args, output_dim, params.img_size)

    # start training
    fedml_runner = FedMLRunner(args, device, dataset, model)
    fedml_runner.run()



