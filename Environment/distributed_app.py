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
        time_out = request_size / (self.alpha * cpu + self.beta * memory)  # the bigger the resources the smaller time out
        time.sleep(time_out)

    def _worker_node_proc(self, queue, done_q, machine_type):
        """Take from the queue; this spawns as a separate Process"""
        while True:
            queue_item = queue.get()
            if str(queue_item) == "DONE":
                done_q.put(queue_item)
                break
            else:
                self._complete_request(queue_item, machine_type)
        return True

    def _master_node_proc(self, count, queue):
        """Write integers into the queue.  A reader_proc() will read them from the queue"""
        for ii in range(0, count - 1):
            queue.put(count)  # Put 'count' numbers into queue
        time.sleep(0.1)

        for ii in range(0, len(self.machines)):  # Tell all workers to stop...
            queue.put("DONE")
        time.sleep(0.1)

    def _start_worker_procs(self, qq, done_q):
        """Start the worker processes and return all in a dict {proc : VM type}"""
        all_worker_procs = list()
        for ii in range(0, len(self.machines)):
            machine_type = self.machines[ii]
            worker_p = Process(target=self._worker_node_proc, args=(qq, done_q, machine_type))
            # print("Machine type: ", machine_type, "process: ", worker_p)
            worker_p.daemon = True
            worker_p.start()
            all_worker_procs.append(worker_p)
        return all_worker_procs

    def _perform_all_requests(self):
        qq = Queue()
        done_q = Queue()
        start = time.time()

        # prepare the queue
        for count in self.queue_array:
            self._master_node_proc(count, qq)
        # run all worker processes
        all_worker_procs = self._start_worker_procs(qq, done_q)

        # wait the processes to finish
        for _, a_worker_proc in enumerate(all_worker_procs):
            a_worker_proc.join()  # Wait for worker node to finish
            # print("Finished:", a_worker_proc)

        done_q.close()
        qq.close()
        done_q.join_thread()
        qq.join_thread()

        end = time.time()
        return end - start
