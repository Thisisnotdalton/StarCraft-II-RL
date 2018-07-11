import tensorflow as tf
import numpy as np
from pysc2.agents.base_agent import BaseAgent
from pysc2.lib import actions, features
from pysc2.env import environment

default_actions = [
   0,  # no_op                                              ()
   1,  # move_camera                                        (1/minimap [64, 64])
   5,  # select_unit                                        (8/select_unit_act [4]; 9/select_unit_id [500])
   7,  # select_army                                        (7/select_add [2])
 331,  # Move_screen                                        (3/queued [2]; 0/screen [84, 84])
 332  # Move_minimap                                       (3/queued [2]; 1/minimap [64, 64])
]

_PLAYER_SELF = features.PlayerRelative.SELF
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL  # beacon/minerals
_PLAYER_ENEMY = features.PlayerRelative.ENEMY

class Brain:
    def __init__(self, race="T", actions = default_actions):
        self.race = race
        self.actions = actions
    def step(self, obs):
        return 0, []

#This represents the actual agent which will play StarCraft II
class MyAgent(BaseAgent):
    def __init__(self, brain = Brain()):
        super().__init__() #call parent constructor
        assert isinstance(brain, Brain)
        self.brain = brain
        
    def step(self, obs): #This function is called once per frame to give the AI observation data and return its action
        super().step(obs) #call parent base method
        action, params = self.brain.step(obs)
        return actions.FunctionCall(action, params)

UNIT_ELEMENTS = 7
MAXIMUM_CARGO = 10
MAXIMUM_BUILD_QUEUE = 10
MAXIMUM_MULTI_SELECT = 10
class StateNet:
    def __init__(self, scope, nonspatial_actions = len(default_actions),
                 resolution=84, screen_channels=17, minimap_channels=7, max_multi_select=MAXIMUM_MULTI_SELECT,
                 max_cargo=MAXIMUM_CARGO, max_build_queue=MAXIMUM_BUILD_QUEUE,
                 l2_scale=0.01, hidden_size=256):
        self.resolution = resolution
        self.variable_feature_sizes = {
            'multi_select' : max_multi_select,
            'cargo' :  max_cargo, 
            'build_queue' : max_build_queue
        }
        #The following assumes that we will stack our minimap and screen features (and they will have the same size)
        with tf.variable_scope('State-{}'.format(scope)):
            self.structured_observation = tf.placeholder(tf.float32, [None, 11], 'StructuredObservation')
            self.single_select = tf.placeholder(tf.float32, [None, 1, UNIT_ELEMENTS], 'SingleSelect')
            self.cargo = tf.placeholder(tf.float32, [None,  max_cargo, UNIT_ELEMENTS], 'Cargo')
            self.multi_select = tf.placeholder(tf.float32, [None, max_multi_select, UNIT_ELEMENTS], 'Multiselect')
            self.build_queue = tf.placeholder(tf.float32, [None,  max_build_queue, UNIT_ELEMENTS], 'BuildQueue')
            self.units = tf.concat([self.single_select,
                                    self.multi_select,
                                    self.cargo,
                                    self.build_queue], axis=1,
                                    name='Units')
            self.control_groups = tf.placeholder(tf.float32, [None, 10, 2], 'ControlGroups')
            self.available_actions = tf.placeholder(tf.float32, [None, nonspatial_actions], 'AvailableActions')
            self.used_actions = tf.placeholder(tf.float32, [None, nonspatial_actions], 'UsedActions')
            self.actions = tf.concat([self.available_actions,
                                      self.used_actions], axis=1,
                                      name='Actions')
            self.nonspatial_features = tf.concat([
                    self.structured_observation,
                    tf.reshape(self.units, [-1, UNIT_ELEMENTS * (1+sum(self.variable_feature_sizes.values()))]),
                    tf.reshape(self.control_groups, [-1, 20]),
                    tf.reshape(self.actions, [-1, 2 * nonspatial_actions])
                ], axis=1, name='NonspatialFeatures')

            self.screen_features = tf.placeholder(tf.float32,
                                                   [None, screen_channels, resolution, resolution],
                                                   'ScreenFeatures')

            self.minimap_features = tf.placeholder(tf.float32,
                                                   [None, minimap_channels, resolution, resolution],
                                                   'MinimapFeatures')


            self.spatial_features = tf.concat([
                self.screen_features, self.minimap_features
            ], axis=1, name='SpatialFeatures')

            self.spatial_features = tf.transpose(self.spatial_features, [0, 2, 3, 1])

            self.conv1 = tf.layers.conv2d(inputs=self.spatial_features, filters=32,
                                          kernel_size=[5, 5],
                                          kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_scale),
                                          kernel_initializer=tf.random_normal_initializer,
                                          activation=tf.nn.relu, name='Convolutional1')
            self.max_pool1 = tf.layers.max_pooling2d(inputs=self.conv1, pool_size=[2, 2],
                                                     strides=2, name='Pool1')
            self.conv2 = tf.layers.conv2d(inputs=self.max_pool1, filters=64,
                                          kernel_size=[5, 5],
                                          kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_scale),
                                          kernel_initializer=tf.random_normal_initializer,
                                          activation=tf.nn.relu, name='Convolutional2')
            self.max_pool2 = tf.layers.max_pooling2d(inputs=self.conv2, pool_size=[2, 2],
                                                     strides=2, name='Pool2')
            self.max_pool2_flat = tf.reshape(self.max_pool2, [-1, 18 * 18 * 64], name='Pool2_Flattened')
            self.state_flattened = tf.concat([self.max_pool2_flat, self.nonspatial_features],
                                             1, name='StateFlattened')
            self.hidden_1 = tf.layers.dense(self.state_flattened, hidden_size, tf.nn.relu,
                                            kernel_initializer=tf.random_normal_initializer,
                                            name='Hidden1')
            self.output = self.hidden_1
            for variable_name, tensor in vars(self).items():
                if isinstance(tensor, tf.Tensor):
                    print('{}:\t({} Shape={})'.format(variable_name, tensor.name, tensor.shape))


class SimpleNet:
    def __init__(self, scope, state_net, resolution=84):
        self.state_net = state_net
        with tf.variable_scope(scope):
            self.input = state_net.output
            self.hidden_layer = tf.layers.dense(self.input, 256, activation=tf.nn.relu,
                                                kernel_initializer=tf.random_normal_initializer)
            self.raw_output = tf.layers.dense(self.hidden_layer, 2)
            self.clipped_output = tf.clip_by_value(self.raw_output, 0, resolution-1)
            self.output = tf.cast(self.clipped_output, tf.int32)
        for variable_name, tensor in vars(self).items():
            if isinstance(tensor, tf.Tensor):
                print('{}:\t({} Shape={})'.format(variable_name, tensor.name, tensor.shape))


class SimpleBrain(Brain):
    def  __init__(self, scope, race="T", actions = default_actions):
        super().__init__(race, actions)
        self.state_net = StateNet(scope)
        self.printed = False
        self.simple_net = SimpleNet(scope, self.state_net)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.feature_placeholders = {
            'available_actions' : self.state_net.available_actions,
            'last_actions' : self.state_net.used_actions,
            'cargo' : self.state_net.cargo,
            'multi_select' : self.state_net.multi_select,
            'single_select' : self.state_net.single_select,
            'build_queue' : self.state_net.build_queue,
            'player' : self.state_net.structured_observation,
            'control_groups' : self.state_net.control_groups,
            'feature_screen' : self.state_net.screen_features,
            'feature_minimap' : self.state_net.minimap_features
        }
        
    def step(self, obs):
        if 331 in obs.observation.available_actions:
            reward, feed_dict, episode_end = self.process_observations(obs)
            outputs = self.sess.run(self.simple_net.output, feed_dict)[0]
            return 331, [[0], outputs]
        else:
            return 7, [[0]]
            
    def process_observations(self, observation):
        if not self.printed:
            print('Observation.observation vars:', dir(observation.observation))
            self.printed = True
        # is episode over?
        episode_end = (observation.step_type == environment.StepType.LAST)
        # reward
        reward = observation.reward #scalar?
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
                    padding = np.zeros((self.state_net.variable_feature_sizes[feature_label] - len(feature), UNIT_ELEMENTS))
                    feature = np.concatenate((feature, padding))
                feature = feature.reshape(-1, UNIT_ELEMENTS)
            placeholder = self.feature_placeholders[feature_label]
            processed_features[placeholder] = np.expand_dims(feature, axis=0)
        return reward, processed_features, episode_end


class SimpleAgent(MyAgent):
    def __init__(self, name='test'):
        super().__init__(SimpleBrain(name))
