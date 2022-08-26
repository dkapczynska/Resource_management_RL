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

    def _check_if_available_(self, number, currently_in_use):
        """
        Method that checks whether there is machine in the storage that can be added

        Parameters:
            number(int) - number of type of virtual machines to be checked)
        """
        vm_in_use = self._select_machine(number)
        vm_left = vm_in_use["capacity"] - currently_in_use
        if vm_left <= 0:
            return 0
        else:
            return 1
