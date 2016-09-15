from misc.experience import Transition

import logging
import numpy as np
import tensorflow as tf
import random

#N_ITERS = 10
N_ITERS = 100
N_UPDATE = 5000
MAX_TIMESTEPS = 40

class ReinforcementTrainer(object):
    def __init__(self, config):
        pass

    def do_rollout(self, model, world, seed=0):
        np.random.seed(seed)
        random.seed(seed)
        tf.set_random_seed(seed)

        transitions = []

        scenario = world.sample_scenario()
        state_before = scenario.init()
        model.init(state_before)

        total_reward = 0.
        for t in range(MAX_TIMESTEPS):
            action = model.act(state_before)
            reward, state_after, terminate = state_before.step(action)
            transitions.append(Transition(state_before, action, state_after, reward))
            total_reward += reward
            state_before = state_after
            if terminate:
                break

        return transitions, total_reward

    def train(self, model, world):
        model.prepare(world)
        total_err = 0.
        total_reward = 0.
        for i_iter in range(N_ITERS):
            transitions, reward = self.do_rollout(model, world, seed=i_iter)
            model.experience(transitions)
            total_reward += reward
            total_err += model.train_rl()
            print "reward " + str(total_reward)

            if (i_iter + 1) % N_UPDATE == 0:
                logging.info("sample transitions: " + \
                        str([t.a for t in transitions]))
                logging.info("[err] %8.3f" % (total_err / N_UPDATE))
                logging.info("[rew] %8.3f" % (total_reward / N_UPDATE))
                logging.info("")

                total_err = 0.
                total_reward = 0.
                model.roll()
