class Storage:
    """
    Class that represents the virtual storage where virtual machines are kept.

    Attributes
    -----------
    vm1(dict):
            dictionary that has 4 elements with keys "cpu", "memory", "cost" and "capacity"  and values
            representing attributes of Virtual Machine Type I

    vm2(dict):
            dictionary that has 4 elements with keys "cpu", "memory", "cost" and "capacity"  and values
            representing attributes of Virtual Machine II

    vm3(dict):
            dictionary that has 4 elements with keys "cpu", "memory", "cost" and "capacity"  and values
            representing attributes of Virtual Machine III
    """
    def __init__(self, vm1, vm2, vm3):
        self.vm1 = vm1
        self.vm2 = vm2
        self.vm3 = vm3

    def _select_machine(self, number):
        if number == 1:
            vm_in_use = self.vm1
        elif number == 2:
            vm_in_use = self.vm2
        elif number == 3:
            vm_in_use = self.vm3
        else:
            raise ValueError("Selected VM type does not exist.")
        return vm_in_use

    def _check_if_available(self, number):
        """
        Method that checks whether there is machine in the storage that can be added

        Parameters:
            number(int) - number of type of virtual machines to be checked)
        """
        vm_in_use = self._select_machine(number)
        cap = vm_in_use["capacity"]
        if cap == 0:
            return 0
        else:
            return 1

    def _add_machine_from_storage(self, number):
        """
        Method that adds virtual machine from the storage to the Distributed App Simulator

        Parameters:
            number(int) - number of type of virtual machines to be added (possible 1, 2 or 3)
        """
        vm_in_use = self._select_machine(number)
        if self._check_if_available(number):
            vm_in_use["capacity"] -= 1
        else:
            pass  # if not machine is available do nothing

    def _return_machine_to_storage(self, number):
        vm_in_use = self._select_machine(number)
        vm_in_use["capacity"] += 1
