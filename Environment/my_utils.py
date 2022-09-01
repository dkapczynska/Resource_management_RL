import matplotlib.pyplot as plt


def viz_performance(cumulative_reward, utility_function, qos, queue_len, cost, no_of_vm, steps, sla):
    # plot config
    plt.rc('axes', labelsize=12) 
    plt.rc('axes', titlesize=15)
    plt.rc('xtick', labelsize=12) 
    plt.rc('ytick', labelsize=12) 
    
    figure, axis = plt.subplots(nrows=3, ncols=2, figsize=(12, 10))
    
    axis[0, 0].plot(steps, cumulative_reward)
    axis[0, 0].set_title("Cumulative reward")
    
    axis[0, 1].plot(steps, utility_function)
    axis[0, 1].set_title("Utility function")

    axis[1, 0].plot(steps, qos)
    axis[1, 0].axhline(y=sla, color='r', linestyle='-')
    axis[1, 0].set_title("QoS & SLA")

    axis[1, 1].plot(steps, queue_len)
    axis[1, 1].set_title("No of requests send")

    axis[2, 0].plot(steps, cost)
    axis[2, 0].set_title("Current cost")

    axis[2, 1].plot(steps, no_of_vm)
    axis[2, 1].set_title("No of virtual machines in use")

    plt.show()
    
def print_stats(value, quality_of_service, queue_len, cost_lst,no_of_vm_lst):
    print("Cumulative reward at the end of the test: ", value[-1])
    print("Avg cost per step: ", np.round(np.mean(cost_lst), 2))
    print("Avg quality of service: ", np.round(np.mean(quality_of_service), 2))
    print("Avg no of requests: ", np.mean(queue_len))
    print("Avg no of virtual machines: ", np.mean(no_of_vm_lst))
