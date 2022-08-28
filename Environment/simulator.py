from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random
from Environment.storage import *
from Environment.distributed_app import *


class Sim(Env):
    def __init__(self,
                 sla,
                 vm_1,
                 vm_2,
                 vm_3,
                 steps,
                 alpha=0.5,
                 beta=0.5):
        """
        Custom environment that follows gym interface.

        Attributes
        -----------
        qos(int):
            Quality of service metric describing how many requests were completed per seconds of simulation

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

        request_completed(int):
            Number of requests that have been successfully completed since beginning of episode

        queue_array(int):
            Number of request waiting in the queue (for the current state)

        computing_power(float):
            Numerical value of metric that represents computing power (for the current state)

        action_space(Discrete object):
            Discrete object of class from OpenAI Gym used to describe structure of action space

        state(int):
            Numerical value describing current state, value of utility functions

        observation_space(Box object):
            Box object of class from OpenAI Gym used to describe structure of observation space

        sim_length(int):
            Number of time units that simulation will last. Each step takes one time unit

        """
        self.sla = sla
        self.vm_1 = vm_1
        self.vm_2 = vm_2
        self.vm_3 = vm_3
        self.steps = steps
        self.alpha = alpha
        self.beta = beta
        self.request_completed = 0
        self.queue_array = []
        self.state = 0
        self.storage = Storage({}, {}, {})
        self.action_space = Discrete(7)
        self.observation_space = Box(low=np.float32(np.array([-500])), high=np.float32(np.array([500])))

    def _generate_workload(self):  # <-------------------------------------- FIX:  postac kolejki
        """
        Number of new requests coming arriving to the system every simulation step
        """
        self.queue_array = []  # clear the queue
        for request in range(0, random.randint(10, 15)):  # fill in the queue with requests
            request_size = random.randint(1, 30)
            self.queue_array.append(request_size)
        # self.queue_array = [12, 12, 4, 2, 4, 5, 7, 4, 3, 2, 5, 6, 5, 5, 6]

    def _calculate_total_comp_power(self):
        vm1_sum = self.vm_1 * (self.alpha * self.storage.vm1['cpu'] + self.beta * self.storage.vm1['memory'])
        vm2_sum = self.vm_2 * (self.alpha * self.storage.vm2['cpu'] + self.beta * self.storage.vm2['memory'])
        vm3_sum = self.vm_3 * (self.alpha * self.storage.vm3['cpu'] + self.beta * self.storage.vm3['memory'])
        self.computing_power = np.round_(vm1_sum + vm2_sum + vm3_sum, 2)

    def _calculate_cost(self):
        vm1_cost = self.storage.vm1['cost'] * self.vm_1
        vm2_cost = self.storage.vm2['cost'] * self.vm_2
        vm3_cost = self.storage.vm3['cost'] * self.vm_3
        cost = np.round_((vm1_cost + vm2_cost + vm3_cost) / 3, 2)  # np.sqrt
        return cost

    def _calculate_utility_function(self, qos, cost):  # <---------------------------- MAYBE BETTER WAY TO SET THIS UP
        if qos >= self.sla:
            penalty = 0
        else:
            penalty = 10
        self.state = ((self.state + cost) / 2) + penalty

    def _calculate_quality_of_service(self, time_per_step):
        request_completed = len(self.queue_array)
        if request_completed == 0:
            qos = 0
        else:
            qos = np.round_(time_per_step / request_completed, 4)
        return qos

    def _calculate_reward(self):  # <---------------------------- MAYBE BETTER WAY TO SET THIS UP
        """
         reward clipping (values from -1  to 1)
         so the gradient doesn't take "too big" steps
         + discounting factor gamma <------------------ TO ADD ?
        """
        return np.round(self.state / 1000, 2)

    def _apply_action(self, action):
        """
        Action space is discrete from 0 to 6
            0 - "do nothing"
            1,2 - add/remove VM 1
            3,4 - add/remove VM 2
            5,6 - add/remove VM 1
        """
        # Take an action
        if (action == 1) & (self.vm_1 < self.storage.vm1["capacity"]):  # add VM type I
            self.vm_1 += 1

        elif (action == 2) & (self.vm_2 < self.storage.vm2["capacity"]):  # add VM type II
            self.vm_2 += 1

        elif (action == 3) & (self.vm_3 < self.storage.vm3["capacity"]):  # add VM type III
            self.vm_3 += 3

        elif (action == 4) & (self.vm_1 > 0):  # remove VM type I from system
            self.vm_1 -= 1

        elif (action == 5) & (self.vm_2 > 0):  # remove VM type II from the system
            self.vm_2 -= 1

        elif (action == 6) & (self.vm_3 > 0):  # remove VM type III from the system
            self.vm_3 -= 1

        else:
            pass

    def _simulate_distributed_app(self):
        # prepare arguments to pass to the application
        vm1_lst = ['vm1'] * self.vm_1  # <---------------- STOP
        vm2_lst = ['vm2'] * self.vm_2
        vm3_lst = ['vm3'] * self.vm_3
        machines = vm1_lst + vm2_lst + vm3_lst
        random.shuffle(machines)

        resources = {'vm1': [self.storage.vm1['cpu'], self.storage.vm1['memory']],
                     'vm2': [self.storage.vm2['cpu'], self.storage.vm2['memory']],
                     'vm3': [self.storage.vm3['cpu'], self.storage.vm3['memory']]}

        distributed_app = DistributedApp(queue_array=self.queue_array,
                                         alpha=self.alpha,
                                         beta=self.beta,
                                         machines=machines,
                                         resources=resources)

        """distributed_app = DistributedApp(queue_array=self.queue_array,
                                         alpha=0.5,
                                         beta=0.5,
                                         machines=['vm1', 'vm2', 'vm1', 'vm3'],
                                         resources={'vm1': [10, 50],
                                                    'vm2': [15, 75],
                                                    'vm3': [25, 75]})"""

        time_per_step = distributed_app._perform_all_requests()
        return time_per_step

    def step(self, action):
        """
        Simulation step:
        1. New requests arrive every simulation step
        2. Agent takes action that is passed to the env
        3. Reward is calculated
        4. Check if step is finished
        """
        # Generate new workload fot the current step
        self._generate_workload()

        # perform an action
        self._apply_action(action)

        # run parallel computation (it results in updated timer)
        time_per_step = self._simulate_distributed_app()

        # calculate Quality of Service
        qos = self._calculate_quality_of_service(time_per_step)

        # Calculate current cost
        cost = self._calculate_cost()

        # calculate next state
        self._calculate_utility_function(qos, cost)

        # calculate the reward
        reward = self._calculate_reward()

        self.steps -= 1
        if self.steps <= 0:
            done = True
            reward = 0
        else:
            done = False

        info = {'Quality of Service': qos,
                'Current cost': cost,
                'Queue len:': len(self.queue_array),
                'Timer': time_per_step,
                'VM type I': self.vm_1,
                'VM type II': self.vm_2,
                'VM type III': self.vm_3,
                }

        return np.array([self.state, ], dtype=np.float32), reward, done, info

    def reset(self, **kwargs):
        # create storage
        self.storage = Storage(vm1={"cpu": 10, "memory": 50, "capacity": 25, "cost": 20},
                               vm2={"cpu": 17, "memory": 65, "capacity": 18, "cost": 35},
                               vm3={"cpu": 25, "memory": 90, "capacity": 10, "cost": 75})

        # reset parameters for the initial setting
        self.state = 0
        self.request_completed = 0
        self.queue_array = []
        return np.array([self.state, ], dtype=np.float32)

    def render(self, **kwargs):
        pass

    def close(self):
        """
        Clean up <------------------- what should be included here?
        """
        return 0
