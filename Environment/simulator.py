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
                 penalty,
                 scenario='train',  # possible 'train' , 'test_1', 'test_2', 'test_3'
                 alpha=0.5,
                 beta=0.5):
        """
        Custom environment that follows gym interface.

        
        """
        super(Sim, self).__init__()

        self.sla = sla
        self.vm_1 = vm_1
        self.vm_2 = vm_2
        self.vm_3 = vm_3
        self.steps = steps
        self.vm_1_init = vm_1
        self.vm_2_init = vm_2
        self.vm_3_init = vm_3
        self.steps_init = steps
        self.alpha = alpha
        self.beta = beta
        self.penalty = penalty
        self.scenario = scenario
        self.request_completed = 0
        self.max_util = 0
        self.request = 0
        self.state = 0
        self.storage = Storage({}, {}, {})
        self.action_space = Discrete(7)
        self.observation_space = Box(low=np.float32(np.array([0])), high=np.float32(np.array([1])))

    def _generate_workload(self):
        """
         FIX: READS ONLY ONE REQUEST PER SIMULATION STEP AND DELETES THEM
        """
        # read only first line from a file
        filename = str(self.scenario)
        with open(filename, 'r') as fr:
            self.request = float(fr.readline().rstrip())*10
    
        # delete this line from a file
            cnt = 1        
            lines = fr.readlines()
            with open(filename, 'w') as fw:
                for line in lines:  
                    if cnt == 1:
                        fw.write(line)
                    else:
                        break
        

    def _calculate_cost(self):
        vm1_cost = self.storage.vm1['cost'] * self.vm_1
        vm2_cost = self.storage.vm2['cost'] * self.vm_2
        vm3_cost = self.storage.vm3['cost'] * self.vm_3
        cost = vm1_cost + vm2_cost + vm3_cost
        return cost

    def _calculate_utility_function(self, qos, cost):
        if qos >= self.sla:
            p = 0
        else:
            p = self.penalty
        self.state = ((((self.state + cost) / 2) + p)/self.max_util)*100

    def _calculate_quality_of_service(self, time_per_step):
        qos = self.request / time_per_step
        return qos

    def _calculate_reward(self, qos, prev_cost, cost):
        """
         reward clipping (values from -1  to 1)
         so the gradient doesn't take "too big" steps
        """
        # print("Prev cost: ", prev_cost, "current cost", cost)
        if qos < self.sla:
            return -1
        elif prev_cost >= cost:
            return 1
        elif prev_cost == cost:
            return 0
        else:
            return -0.5

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
            self.vm_3 += 1

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
        vm1_lst = ['vm1'] * self.vm_1
        vm2_lst = ['vm2'] * self.vm_2
        vm3_lst = ['vm3'] * self.vm_3
        machines = vm1_lst + vm2_lst + vm3_lst
        random.shuffle(machines)

        resources = {'vm1': [self.storage.vm1['cpu'], self.storage.vm1['memory']],
                     'vm2': [self.storage.vm2['cpu'], self.storage.vm2['memory']],
                     'vm3': [self.storage.vm3['cpu'], self.storage.vm3['memory']]}

        distributed_app = DistributedApp(request=self.request,
                                         alpha=self.alpha,
                                         beta=self.beta,
                                         machines=machines,
                                         resources=resources)

        time_per_step = distributed_app._perform_request()
        return time_per_step

    def step(self, action):
        """
        Simulation step:
        1. New requests arrive every simulation step
        2. Agent takes action that is passed to the env
        3. Reward is calculated
        4. Check if step is finished
        """
        # Previous cost
        prev_cost = self._calculate_cost()

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

        # calculate the reward
        reward = self._calculate_reward(qos, cost, prev_cost)
        
        # calculate next state
        self._calculate_utility_function(qos, cost)

        self.steps -= 1
        if self.steps <= 0:
            done = True
            reward = 0
        else:
            done = False

        info = {'Quality of Service': qos,
                'Current cost': cost,
                'Request': self.request,
                'Timer': time_per_step,
                'VM type I': self.vm_1,
                'VM type II': self.vm_2,
                'VM type III': self.vm_3,
                }

        return np.array([self.state, ], dtype=np.float32), reward, done, info

    def reset(self, **kwargs):

        # create storage
        self.storage = Storage(vm1={"cpu": 8, "memory": 8, "capacity": 15, "cost": 20},
                               vm2={"cpu": 16, "memory": 10, "capacity": 10, "cost": 35},
                               vm3={"cpu": 32, "memory": 24, "capacity": 3, "cost": 75})

        # reset parameters for the initial setting
        self.vm_1 = self.vm_1_init
        self.vm_2 = self.vm_2_init
        self.vm_3 = self.vm_3_init
        self.steps = self.steps_init
        
        # calculate max utility function (for normalization) 
        vm1_max_cost = self.storage.vm1['cost'] * self.storage.vm1['capacity']
        vm2_max_cost = self.storage.vm2['cost'] * self.storage.vm2['capacity']
        vm3_max_cost = self.storage.vm3['cost'] * self.storage.vm3['capacity']
        max_cost = vm1_max_cost + vm2_max_cost + vm3_max_cost
        self.max_util = max_cost * self.steps
        
        self.state = self._calculate_cost()/self.max_util * 100
        self.request_completed = 0
        self.queue_array = []
        return np.array([self.state, ], dtype=np.float32)

    def render(self, **kwargs):
        pass

    def close(self):
        """
        Clean up
        """
        return 0
