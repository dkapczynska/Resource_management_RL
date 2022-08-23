from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random


class SimEnv(Env):
    def __init__(self,
                 qos,
                 cost,
                 sla,
                 vm_1,
                 vm_2,
                 vm_3,
                 alpha,
                 beta,
                 gamma,
                 timeunits):
        """
        Custom environment that follows gym interface.

        Attributes
        -----------
        qos(int):
            Quality of service metric describing how many requests were complited per seconds of simulation

        cost(int):
            Current cost of system performance

        vm_1(int):
            Number of virtual machine type I that are currently in use

        vm_2(int):
            Number of virtual machine type II that are currently in use

        vm_3(int):
            Number of virtual machine type III that are currently in use

        alpha(float):
            Constant numerical value that assign weight to CPU in total computing power equation

        beta(float):
            Constant numerical value that assign weight to memory in total computing power equation

        gamma(float):
            Numerical value of discounting factor

        request_completed(int):
            Number of requests that have been successfully completed since beginning of episode

        queue_count(int):
            Number of request waiting in the queue (for the current state)

        computing_power(float):
            Numerical value of metric that represents computing power (for the current state)

        action_space(Discrete object):
            Discrete object of class from OpenAI Gym used to describe structure of action space

        state(int):
            Numerical value describing current state, value of utility functions

        observation_space(Box object):
            Box object of class from OpenAI Gym used to describe structure of observation space

        sim_lenght(int):
            Number of time units that simulation will last. Each step takes one time unit

        """
        self.qos = qos
        self.cost = cost
        self.sla = sla
        self.vm_1 = vm_1
        self.vm_2 = vm_2
        self.vm_3 = vm_3
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.request_completed = 0
        self.queue_count = 0
        self.computing_power = 8.405
        self.action_space = Discrete(7)
        self.observation_space = Box(low=np.array([10]), high=np.array([30]))
        self.sim_length = timeunits

    def _generate_workload(self):
        """
        Number of new requests coming arriving to the system every simulation step
        """
        self.queue_count += (100 + random.randint(-25, 150))  # range: 25 - 250

    def _calculate_total_comp_power(self):
        vm1_sum = self.vm_1 * (self.alpha * 2 + self.beta * 8)
        vm2_sum = self.vm_2 * (self.alpha * 3 + self.beta * 14)
        vm3_sum = self.vm_3 * (self.alpha * 5 + self.beta * 29)
        self.computing_power = vm1_sum + vm2_sum + vm3_sum

    def _calculate_utility_function(self):
        """
        1. Check if SLA was satisfied
        2. Return utility function value
        """
        pass

    def _apply_action(self, action):
        """
        Action space is discrete from 0 to 6
            0 - "do nothing"
            1,2 - add/remove VM 1
            3,4 - add/remove VM 2
            5,6 - add/remove VM 1

        To apply action:
        0. What action was taken?
        1. Calculate total computing power
        2. Use the computing power to process the reqests from the queue
        3. Remove them from the queue
        4. One sec has passed
        5. Calculate QoS
        6. Calculate the cost
        7. Calculate the utility function
        8. Pass the state information (score of utility function)

        Return state (of utility function)
        """
        if action == 0:
            pass

        elif action == 1:
            self.vm_1 = self.vm_1 + 1  # add VM type I

        elif action == 2:
            self.vm_2 = self.vm_2 + 1  # add VM type II

        elif action == 3:
            self.vm_3 = self.vm_3 + 1  # add VM type III

        elif (action == 4) & (self.vm_1 >= 1):  # VM type I available in the storage
            self.vm_1 = self.vm_1 - 1

        elif (action == 4) & (self.vm_1 < 1):  # VM type I NOT available in the storage
            pass

        elif (action == 5) & (self.vm_2 >= 1):  # VM type II available in the storage
            self.vm_2 = self.vm_2 - 1

        elif (action == 5) & (self.vm_2 < 1):  # VM type II NOT available in the storage
            pass

        elif (action == 6) & (self.vm_3 >= 1):  # VM type III available in the storage
            self.vm_3 = self.vm_3 - 1

        elif (action == 6) & (self.vm_3 < 1):  # VM type III NOT available in the storage
            pass

        else:
            raise ValueError("Action number out of range.")

        vm_total_count = self.vm_1 + self.vm_2 + self.vm_3
        self._calculate_total_comp_power()
        requests_possible_to_complete = self.computing_power / vm_total_count

        if requests_possible_to_complete > self.queue_count:
            self.request_completed = self.queue_count
            self.queue_count = 0
        else:
            self.request_completed = requests_possible_to_complete
            self.queue_count -= self.request_completed

        self.sim_length -= 1
        self.qos = self.request_completed / (60 - self.sim_length)

        self.cost = np.sqrt(self.vm_1 * 20 + self.vm_2 * 35 + self.vm_3 * 75)
        return vm_total_count
        # self._calculate_utility_function()

    def _calculate_reward(self):
        """
        ATTENTION: ADD IMPACT OF DISCOUNTING FACTOR HERE !!!!
        """
        if self.qos < self.sla:
            reward = -10
        elif self.cost < 15.64:
            reward = 5
        elif (self.cost >= 15.65) & (self.cost < 24.12):
            reward = 2
        elif self.cost >= 24.12:
            reward = 1
        else:
            raise ValueError("Total cost out of range.")
        return float(reward)

    def step(self, action):
        """
        Simulation step:
        1. New requests arrive every simulation step
        2. Agent takes action that is passed to the env
        3. Reward is calculated
        4. Check if step is finished
        """
        self._generate_workload()
        vm_total_count = self._apply_action(action)
        reward = self._calculate_reward()

        if self.sim_length <= 0:
            done = True
        else:
            done = False

        info = {'Quality of Service ': self.qos,
                'Total cost': self.cost,
                'TCP': self.computing_power,
                'VM type I': self.vm_1,
                'VM type II': self.vm_2,
                'VM type III': self.vm_3,
                'No of VMs': vm_total_count,
                }
        return self.cost, reward, done, info  # state : cost

    def reset(self, **kwargs):
        self.request_completed = 0
        self.queue_count = 0
        self.computing_power = 8.405
        return self.cost
