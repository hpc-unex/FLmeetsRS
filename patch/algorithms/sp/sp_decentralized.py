#!/usr/bin/env python

import fedml
from fedml import FedMLRunner

def launchSP(manualSeed, config, algorithm, img_size):

    # init FedML framework
    args = fedml.init(manualSeed, config, algorithm)

    # init device
    device = fedml.device.get_device(args)

    # load data
    dataset, output_dim = fedml.data.load(args)
    # load model
    model = fedml.model.create(args, output_dim, img_size)

    # start training
    fedml_runner = FedMLRunner(args, device, dataset, model)
    fedml_runner.run()
