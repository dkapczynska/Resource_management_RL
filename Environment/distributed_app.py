from multiprocessing import Process, Queue
import time
import warnings
warnings.filterwarnings("ignore")


class DistributedApp:

    def __init__(self, queue_array, alpha, beta, machines, resources):
        self.queue_array = queue_array
        self.alpha = alpha
        self.beta = beta
        self.machines = machines  # list of element 'vm1, vm2, vm3
        self.resources = resources  # dict 'vm1': [cpu, memory]

    def _complete_request(self, request_size, machine_type):
        """  takes time out dependently on size of request and machine parameters """

        cpu = self.resources[machine_type][0]
        memory = self.resources[machine_type][1]
        time_out = 10 * request_size/(self.alpha * cpu + self.beta * memory)  # the bigger the resources the smaller time out
        time.sleep(time_out)

    def _worker_node_proc(self, queue, machine_type):  # <--------- FIX: vm_type add
        """Take from the queue; this spawns as a separate Process"""
        while True:
            queue_item = queue.get()
            assert 0 < queue_item < 50
            if str(queue_item) == "DONE":
                break
            else:
                self._complete_request(queue_item, machine_type)

    def _master_node_proc(self, count, queue):
        """Write integers into the queue.  A reader_proc() will read them from the queue"""
        for ii in range(0, count-1):
            queue.put(count)  # Put 'count' numbers into queue

        for ii in range(0, len(self.machines)):  # Tell all workers to stop...
            queue.put("DONE")

    def _start_worker_procs(self, qq):
        """Start the worker processes and return all in a dict {proc : VM type}"""
        all_worker_procs = list()
        for ii in range(0, len(self.machines)):
            machine_type = self.machines[ii]
            worker_p = Process(target=self._worker_node_proc, args=(qq, machine_type))
            # print("Machine type: ", machine_type, "process: ", worker_p)
            worker_p.daemon = True
            worker_p.start()
            all_worker_procs.append(worker_p)
        return all_worker_procs

    def _perform_all_requests(self):
        qq = Queue()

        start = time.time()
        # prepare the queue
        for count in self.queue_array:
            self._master_node_proc(count, qq)
        # run all worker processes
        assert 0 < len(self.machines) < 53
        all_worker_procs = self._start_worker_procs(qq)

        # wait the processes to finish
        for _, a_worker_proc in enumerate(all_worker_procs):
            a_worker_proc.join()  # Wait for worker node to finish
            # print("Finished:", a_worker_proc)
        qq.close()

        end = time.time()
        return end - start
