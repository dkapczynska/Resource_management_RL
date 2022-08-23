# Resource_management_RL
Automatic adjusting resources of a distributed application with the use of Reinforcement Learning

## Mock-up simulator of distribiuted application 
Current setting:
* Total computing power
  ```sh
   vm1_sum = self.vm_1 * (self.alpha * 2 + self.beta * 8)
        vm2_sum = self.vm_2 * (self.alpha * 3 + self.beta * 14)
        vm3_sum = self.vm_3 * (self.alpha * 5 + self.beta * 29)
        self.computing_power = vm1_sum + vm2_sum + vm3_sum
  ```
  *
