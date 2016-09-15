#!/usr/bin/env python2

from misc.util import Struct
from misc.experience import Transition
import models
import trainers
import worlds

import time
import logging
import numpy as np
import random
import tensorflow as tf
import yaml
import ray

def main():
    ray.init(start_ray_local=True, num_workers=5)
    ray.register_class(Transition)
    ray.register_class(worlds.lattice.LatticeState)
    ray.register_class(worlds.lattice.LatticeScenario)
    ray.register_class(Struct)
    ray.register_class(set, pickle=True)
    config = configure()
    def world_initializer():
        return worlds.load(config)
    def world_reinitializer(world):
        return world
    ray.reusables.world = ray.Reusable(world_initializer, world_reinitializer)
    def model_initializer():
        model = models.load(config)
        model.prepare(ray.reusables.world)
        return model
    def model_reinitializer(model):
        return model
    ray.reusables.model = ray.Reusable(model_initializer, model_reinitializer)
    trainer = trainers.load(config)
    start_time = time.time()
    trainer.train(ray.reusables.model, ray.reusables.world)
    end_time = time.time()
    print "Training took " + str(end_time - start_time) + " seconds."

def configure():
    np.random.seed(0)
    random.seed(0)
    tf.set_random_seed(0)
    with open("config.yaml") as config_f:
        config = Struct(**yaml.load(config_f))
    log_name = "logs/%s-%d_%s-%d_%s.log" % (
            config.world.name,
            config.world.size,
            config.model.name,
            config.model.depth,
            config.trainer.name)
    logging.basicConfig(filename=log_name, level=logging.DEBUG,
            format='%(asctime)s %(levelname)-8s %(message)s')
    logging.info("BEGIN")
    logging.info(str(config))
    return config

if __name__ == "__main__":
    main()
