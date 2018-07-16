import tensorflow as tf
import numpy as np
from pysc2.agents.base_agent import BaseAgent
from pysc2.lib import actions
from pysc2.env import environment
from utils import print_tensors, discount_rewards

DEFAULT_ACTIONS = [
    0,  # no_op                                              ()
    1,  # move_camera                                        (1/minimap [64, 64])
    5,  # select_unit                                        (8/select_unit_act [4]; 9/select_unit_id [500])
    7,  # select_army                                        (7/select_add [2])
    331,  # Move_screen                                        (3/queued [2]; 0/screen [84, 84])
    332  # Move_minimap                                       (3/queued [2]; 1/minimap [64, 64])
]

UNIT_ELEMENTS = 7
MAXIMUM_CARGO = 10
MAXIMUM_BUILD_QUEUE = 10
MAXIMUM_MULTI_SELECT = 10


class Brain:
    def __init__(self, race="T", actions=DEFAULT_ACTIONS):
        self.race = race
        self.actions = actions

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


class SimpleNet:
    def __init__(self, scope, state_net, resolution=84):
        self.resolution = resolution
        self.state_net = state_net
        with tf.variable_scope(scope):
            self.input = state_net.output
            self.hidden_layer = tf.layers.dense(self.input, 128, activation=tf.nn.relu,
                                                kernel_initializer=tf.random_normal_initializer,
                                                bias_initializer=tf.random_normal_initializer,
                                                name='SimpleHidden')
            self.raw_output_x = tf.layers.dense(self.hidden_layer, resolution, activation=tf.nn.relu,
                                                kernel_initializer=tf.random_normal_initializer,
                                                bias_initializer=tf.random_normal_initializer,
                                                name='RawOutputX')
            self.raw_output_y = tf.layers.dense(self.hidden_layer, resolution, activation=tf.nn.relu,
                                                kernel_initializer=tf.random_normal_initializer,
                                                bias_initializer=tf.random_normal_initializer,
                                                name='RawOutputY')
            self.soft_output_x = tf.nn.softmax(self.raw_output_x)
            self.soft_output_y = tf.nn.softmax(self.raw_output_y)
            self.x = tf.argmax(self.soft_output_x, axis=1)
            self.y = tf.argmax(self.soft_output_y, axis=1)
            self.output = self.x, self.y
            self.value = tf.layers.dense(self.hidden_layer, 1, activation=tf.nn.relu, name='Value')
            print_tensors(self)


class SimpleTrainer:
    def __init__(self, scope, simple_net,
                 trainer=tf.train.AdamOptimizer(learning_rate=0.001)):
        with tf.variable_scope(scope):
            self.chosen_x = tf.placeholder(tf.int32, [None, ])
            self.chosen_y = tf.placeholder(tf.int32, [None, ])
            self.chosen_x_one_hot = tf.one_hot(self.chosen_x, simple_net.resolution)
            self.chosen_y_one_hot = tf.one_hot(self.chosen_y, simple_net.resolution)

            self.reward = tf.placeholder(tf.float32, [None, ], name='ValueTarget')
            self.value_loss = tf.reduce_mean(simple_net.value - self.reward)
            self.entropy_loss_x = tf.multiply(simple_net.soft_output_x, self.chosen_x_one_hot)
            self.entropy_loss_y = tf.multiply(simple_net.soft_output_y, self.chosen_x_one_hot)

            self.entropy_loss_x = tf.reduce_mean(self.entropy_loss_x) * self.reward
            self.entropy_loss_y = tf.reduce_mean(self.entropy_loss_y) * self.reward

            self.loss = self.entropy_loss_x + self.entropy_loss_y + self.value_loss

            self.train_op = trainer.minimize(self.loss)


class SimpleBrain(Brain):
    def __init__(self, scope, race="T", actions=DEFAULT_ACTIONS, step_buffer_size=30, epsilon=0.25, noop_rate=7):
        super().__init__(race, actions)
        self.state_net = StateNet(scope)
        self.printed = False
        self.simple_net = SimpleNet(scope, self.state_net)
        self.simple_trainer = SimpleTrainer(scope, self.simple_net)
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
        self.epsilon = epsilon
        self.noop_rate = noop_rate
        self.noop_count = 0
        # step buffer full of (previous_state, reward)
        self.previous_state = None
        self.step_count = 0
        self.episode_rewards = 0
        self.losses = []

    def reset(self):
        self.step_buffer.clear()
        self.noop_count = 0
        self.previous_state = None
        self.episode_rewards = 0
        # self.step_count = 0

    def train(self):
        # do some data formatting
        feed_dict = {
            self.simple_trainer.reward: discount_rewards([reward for state, reward, x, y in self.step_buffer]),
            self.simple_trainer.chosen_x: [x for state, reward, x, y in self.step_buffer],
            self.simple_trainer.chosen_y: [y for state, reward, x, y in self.step_buffer]

        }
        for text_label, feature_label in self.feature_placeholders.items():
            feed_dict[feature_label] = np.array([state[feature_label][0] for state, reward, x, y in self.step_buffer])
        # feed into nn
        # run train op
        self.sess.run(self.simple_trainer.train_op, feed_dict)
        loss = self.sess.run(self.simple_trainer.loss, feed_dict)
        self.losses.append(loss)
        self.step_buffer.clear()

    def step(self, obs):
        # check our value/reward from  our observation

        if 331 in obs.observation.available_actions:
            if self.noop_count <= 0:
                reward, feed_dict, episode_end = self.process_observations(obs)
                self.episode_rewards += reward
                outputs = self.sess.run([self.simple_net.x, self.simple_net.y], feed_dict)
                outputs = [output[0] for output in outputs]
                if episode_end or (len(self.step_buffer) % self.step_buffer_size == 0
                                   and len(self.step_buffer) > 0):
                    self.train()
                self.step_count += 1
                if np.random.uniform() < np.power(1 - self.epsilon, np.log(self.step_count)):
                    outputs = np.random.randint(0, self.simple_net.resolution - 1, 2)
                # pass in old state, x, y with reward
                if self.previous_state:
                    self.step_buffer.append((self.previous_state[0], self.episode_rewards,
                                             self.previous_state[1], self.previous_state[2]))
                # set previous state, x, y to current
                self.previous_state = (feed_dict, outputs[0], outputs[1])
                self.noop_count = self.noop_rate
                return 331, [[0], outputs]
            else:
                self.noop_count -= 1
                return 0, []
        else:
            return 7, [[0]]

    def process_observations(self, observation):
        if not self.printed:
            print('Observation.observation vars:', dir(observation.observation))
            self.printed = True
        # is episode over?
        episode_end = (observation.step_type == environment.StepType.LAST)
        # reward
        reward = observation.reward  # scalar?
        # features
        features = observation.observation
        # the shapes of some features depend on the state (eg. shape of multi_select depends on number of units)
        # since tf requires fixed input shapes, we set a maximum size then pad the input if it falls short
        processed_features = {}
        # for feature_label, feature in observation.observation.items():
        for feature_label in self.feature_placeholders:
            feature = observation.observation[feature_label]
            if feature_label in ['available_actions', 'last_actions']:
                actions = np.zeros(len(self.actions))
                for i, action in enumerate(self.actions):
                    if action in feature:
                        actions[i] = 1
                feature = actions
            elif feature_label in ['single_select', 'multi_select', 'cargo', 'build_queue']:
                if feature_label in self.state_net.variable_feature_sizes:
                    padding = np.zeros(
                        (self.state_net.variable_feature_sizes[feature_label] - len(feature), UNIT_ELEMENTS))
                    feature = np.concatenate((feature, padding))
                feature = feature.reshape(-1, UNIT_ELEMENTS)
            placeholder = self.feature_placeholders[feature_label]
            processed_features[placeholder] = np.expand_dims(feature, axis=0)
        return reward, processed_features, episode_end


class SimpleAgent(MyAgent):
    def __init__(self, name='test'):
        super().__init__(SimpleBrain(name))
