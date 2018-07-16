import tensorflow as tf
import numpy as np
from pysc2.agents.base_agent import BaseAgent
from pysc2.lib import actions
from pysc2.env import environment
from utils import Constants, print_tensors, discount_rewards
from networks import StateNet, RecurrentNet, A3CNet, A3CWorker


class Brain:
    def __init__(self, race="T", action_set=Constants.DEFAULT_ACTIONS):
        self.race = race
        self.action_set = sorted(action_set)

    def reset(self):
        pass

    def step(self, obs):
        return 0, []


# This represents the actual agent which will play StarCraft II
class MyAgent(BaseAgent):
    def __init__(self, brain=Brain()):
        super().__init__()  # call parent constructor
        assert isinstance(brain, Brain)
        self.brain = brain

    def reset(self):
        self.brain.reset()

    def step(self, obs):  # This function is called once per frame to give the AI observation data and return its action
        super().step(obs)  # call parent base method
        action, params = self.brain.step(obs)
        return actions.FunctionCall(action, params)


class A3CBrain(Brain):
    def __init__(self, scope, race="T", action_set=Constants.DEFAULT_ACTIONS, step_buffer_size=30):
        super().__init__(race, action_set)
        self.state_net = StateNet(scope, action_size=len(self.action_set))
        self.rnn = RecurrentNet(scope, self.state_net)
        self.a3c_net = A3CWorker(scope, self.state_net, self.rnn)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.feature_placeholders = {
            'available_actions': self.state_net.available_actions,
            'last_actions': self.state_net.used_actions,
            'cargo': self.state_net.cargo,
            'multi_select': self.state_net.multi_select,
            'single_select': self.state_net.single_select,
            'build_queue': self.state_net.build_queue,
            'player': self.state_net.structured_observation,
            'control_groups': self.state_net.control_groups,
            'feature_screen': self.state_net.screen_features,
            'feature_minimap': self.state_net.minimap_features
        }
        self.step_buffer_size = step_buffer_size
        self.step_buffer = []

    def reset(self):
        self.step_buffer.clear()

    def train(self):
        self.step_buffer.clear()

    def step(self, obs):
        return 7, [[0]]

    def process_observations(self, observation):
        # is episode over?
        episode_end = (observation.step_type == environment.StepType.LAST)
        # reward
        reward = observation.reward  # scalar?
        # features
        features = observation.observation
        # the shapes of some features depend on the state (eg. shape of multi_select depends on number of units)
        # since tf requires fixed input shapes, we set a maximum size then pad the input if it falls short
        processed_features = {}
        for feature_label in self.feature_placeholders:
            feature = features[feature_label]
            if feature_label in ['available_actions', 'last_actions']:
                action_inputs = np.zeros(len(self.action_set))
                for i, action in enumerate(self.action_set):
                    if action in feature:
                        action_inputs[i] = 1
                feature = action_inputs
            elif feature_label in ['single_select', 'multi_select', 'cargo', 'build_queue']:
                if feature_label in self.state_net.variable_feature_sizes:
                    padding = np.zeros(
                        (self.state_net.variable_feature_sizes[feature_label] - len(feature), Constants.UNIT_ELEMENTS))
                    feature = np.concatenate((feature, padding))
                feature = feature.reshape(-1, Constants.UNIT_ELEMENTS)
            placeholder = self.feature_placeholders[feature_label]
            processed_features[placeholder] = np.expand_dims(feature, axis=0)
        return reward, processed_features, episode_end


class A3CAgent(MyAgent):
    def __init__(self, name='test'):
        super().__init__(A3CBrain(name))
