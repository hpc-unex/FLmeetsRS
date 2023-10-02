
import argparse
import logging
import os
import random
import socket
import sys
import yaml
import fedml
import torch
from fedml.simulation import SimulatorMPI

from fedml.model.cv.resnet_gn import resnet18




def launch0MPIDec(manualSeed, config, algorithm, img_size):
    # init FedML framework
    args = fedml.init(manualSeed, config, algorithm)

    # init device
    device = fedml.device.get_device(args)
    # load data
    dataset, output_dim = fedml.data.load(args)

    # load model (the size of MNIST image is 28 x 28)
    model = fedml.model.create(args, output_dim, img_size)

    # start training
    simulator = SimulatorMPI(args, device, dataset, model)
    simulator.run()




#import fedml
#from fedml import FedMLRunner

#def launch0MPIDec(manualSeed, config, algorithm):
    ## init FedML framework
    #args = fedml.init(manualSeed, config, algorithm)
    
    ## init device
    #device = fedml.device.get_device(args)

    ## load data
    #dataset, output_dim = fedml.data.load(args)
    ## load model
    #model = fedml.model.create(args, output_dim)

    ## start training
    #fedml_runner = FedMLRunner(args,device, dataset, model)
    #fedml_runner.run()
