# Resource_management_RL
Automatic adjusting resources of a distributed application with the use of Reinforcement Learning

## Mock-up simulator of distribiuted application 
The issue of optimization is about minimizing the cost of resources while maintaining SLA. SLA is measured by QoS metric that is based on speed of processing the requests per second of the simulation. 
 ```sh
  self.qos = self.request_completed / (60 - self.sim_length)
  ```

The speed depends on the current total computing power that is based on current number of virtual machines in use:
  ```sh
   vm1_sum = self.vm_1 * (self.alpha * 2 + self.beta * 8)
   vm2_sum = self.vm_2 * (self.alpha * 3 + self.beta * 14)
   vm3_sum = self.vm_3 * (self.alpha * 5 + self.beta * 29)
   self.computing_power = vm1_sum + vm2_sum + vm3_sum
  ```
  Parameters α and β represent weights of CPU and memory and their contribution to the computing power.
  
 Each simulation step last one second. Every episode takes 1 minute to complete. Every second there are new requests coming to the system. Number of requests is randomized but warries between 25 and 250. Total cost is a square root of a sum of all the virtual machines’ costs multiples by their quantity. 
 
 ```sh
  self.cost = np.sqrt(self.vm_1 * 20 + self.vm_2 * 35 + self.vm_3 * 75)
  ```
Current storage capacity is set for:



![image](https://user-images.githubusercontent.com/72708718/186509527-ec4caca1-5bc0-4631-a14f-ad5e1fdfb6dd.png)

