# Automatic adjusting resources of a distributed application with the use of Reinforcement Learning
Project aims to test several RL agents based on different policies in order to how machine learning techniques can improve dynamic resource allocation for distribuited allocation.

Resource allocation strategies that have been tested:
00. Static allocation
01. Random action agent
02. Adventage Actor-Critic agent 
03. Proximal Policy Optimization agent
04. Deep Q-network agent

## Simulation environement
The agents are trained in custom envinment that follow OpenAI Gym structure. The simulation enviroment simulates real distribuited application behaviors performing parallel processes depending on number of virtual machines that can be removed or added dynamically after every simulation step. All the processes share the same queue of requests. Workload is genereted randomly, requests have different sizes. Quality of service is measured by average speed per requests for each simulation step. The objective is to minimize the cost of resources while maintaining SLA, so utility function is given by:

 ```sh
  def _calculate_utility_function(self, qos, cost):
        if qos >= self.sla:
            penalty = 0
        else:
            penalty = 100
        self.state = ((self.state + cost) / 2) + penalty
  ```

Reward is given to the agent accordingly to the formula:
  ```sh
  def _calculate_reward(self, qos, prev_cost, cost):
        if qos < self.sla:
            return -1
        elif prev_cost >= cost:
            return 1
        else:
            return 0
  ```
Current capacity of the storage is:
 
 ```sh
Storage(vm1={"cpu": 8, "memory": 8, "capacity": 15, "cost": 20},
        vm2={"cpu": 16, "memory": 10, "capacity": 10, "cost": 35},
        vm3={"cpu": 32, "memory": 24, "capacity": 3, "cost": 75})
  ```

