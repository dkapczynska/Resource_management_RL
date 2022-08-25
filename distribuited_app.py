from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random
from storage import *


class DistribiutedApp(Env):
    def __init__(self,
                 sla,
                 vm_1,
                 vm_2,
                 vm_3,
                 alpha=0.5,
                 beta=0.5,
                 gamma=0.97,
                 timeunits=60):
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
        super(DistribiutedApp, self).__init__()

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
        self.state = 0
        self.qos = 0  # actual value set in reset
        self.cost = 0  # actual value set in reset
        self.storage = Storage({}, {}, {})
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

    def _calculate_cost(self):
        vm1_cost = self.storage.vm1['cost'] * self.vm_1
        vm2_cost = self.storage.vm2['cost'] * self.vm_2
        vm3_cost = self.storage.vm3['cost'] * self.vm_3
        self.cost = np.sqrt(vm1_cost + vm2_cost + vm3_cost)

    def _calculate_utility_function(self):
        if self.qos >= self.sla:
            penalty = 0
        else:
            penalty = 10
        self.state = self.cost + penalty

    def _calculate_quality_of_service(self):
        if self.request_completed == 0:
            return 0
        else:
            self.qos = self.request_completed / (61 - self.sim_length)

    def _calculate_reward(self):
        """
         <====== FIX: experimental formula
        """
        return self.state * self.gamma

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
        # Take an action <---------------------------------- FIX: podejrzanie wyglada ta czesc
        if action == 0:
            pass

        elif action == 1:  # add VM type I
            if self.storage._check_if_available(1):
                self.vm_1 += 1
                self.storage._add_machine_from_storage(1)
            else:
                pass

        elif action == 2:  # add VM type II
            if self.storage._check_if_available(2):
                self.vm_2 += 1
                self.storage._add_machine_from_storage(2)
            else:
                pass

        elif action == 3:  # add VM type III
            if self.storage._check_if_available(3):
                self.vm_1 += 3
                self.storage._add_machine_from_storage(3)
            else:
                pass

        elif action == 4:  # remove VM type I from the storage
            self.vm_1 -= 1
            self.storage._return_machine_to_storage(1)

        elif action == 5:  # remove VM type II from the storage
            self.vm_2 -= 1
            self.storage._return_machine_to_storage(2)

        elif action == 6:  # remove VM type III from the storage
            self.vm_3 = self.vm_3 - 1
            self.storage._return_machine_to_storage(3)

        else:
            raise ValueError("Action number out of range.")

        # Calculate current computing power
        self._calculate_total_comp_power()

        # Discount number of requests that were completed due to given computing power
        vm_total_count = self.vm_1 + self.vm_2 + self.vm_3
        requests_possible_to_complete = self.computing_power / vm_total_count

        if requests_possible_to_complete > self.queue_count:
            self.request_completed = self.queue_count
            self.queue_count = 0
        else:
            self.request_completed = requests_possible_to_complete
            self.queue_count -= self.request_completed

        # Discount 1 second from run time and calculate Quality of Service for this step
        self._calculate_quality_of_service()

        # Calculate the cost for this step
        self._calculate_cost()

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
        vm_total_count = self._apply_action(action)
        reward = self._calculate_reward()

        # Check termination condition
        self.sim_length -= 1
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
        return self.cost, reward, done, info

    def reset(self, **kwargs):
        # create storage
        self.storage = Storage(vm1={"cpu": 10, "memory": 50, "capacity": self.vm_1, "cost": 25},
                               vm2={"cpu": 17, "memory": 65, "capacity": self.vm_2, "cost": 35},
                               vm3={"cpu": 25, "memory": 90, "capacity": self.vm_3, "cost": 75})

        # reset the queue
        self.request_completed = 0
        self.queue_count = 0

        # calculate parameters for initial setting
        self._calculate_total_comp_power()
        self._calculate_quality_of_service()
        self._calculate_cost()
        self._calculate_utility_function()
        return self.state
