from misc.experience import Transition

import logging
import numpy as np
import tensorflow as tf
import random
import ray

N_ITERS = 5
N_UPDATE = 5000
MAX_TIMESTEPS = 40
N_PARALLEL_ROLLOUTS = 5
ROLLOUT_BATCH_SIZE = 100

@ray.remote(num_return_vals=2)
def do_rollout(weights, num_rollouts=1, seed=0):
    np.random.seed(seed)
    random.seed(seed)
    tf.set_random_seed(seed)

    world = ray.reusables.world
    model = ray.reusables.model

    model.set_weights(weights)

    transitions_batch = []
    total_reward_batch = []
    for _ in range(num_rollouts):
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

        transitions_batch.append(transitions)
        total_reward_batch.append(total_reward)

    return transitions_batch, total_reward_batch

class ReinforcementTrainer(object):
    def __init__(self, config):
        pass

    def train(self, model, world):
        total_err = 0.
        total_reward = 0.
        for i_iter in range(N_ITERS):
            weights = model.get_weights()
            transitions_batch_id_list = []
            reward_batch_id_list = []
            for _ in range(N_PARALLEL_ROLLOUTS):
                transitions_batch_id, reward_batch_id = do_rollout.remote(weights, num_rollouts=ROLLOUT_BATCH_SIZE, seed=i_iter)
                transitions_batch_id_list.append(transitions_batch_id)
                reward_batch_id_list.append(reward_batch_id)
            transitions_batch_list = ray.get(transitions_batch_id_list)
            reward_batch_list = ray.get(reward_batch_id_list)
            transitions_list = [transitions for transitions_batch in transitions_batch_list for transitions in transitions_batch]
            reward_list = [reward for reward_batch in reward_batch_list for reward in reward_batch]
            for transitions in transitions_list:
                model.experience(transitions)
            total_reward += sum(reward_list)
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
