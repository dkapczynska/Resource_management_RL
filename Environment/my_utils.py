import matplotlib.pyplot as plt


def viz_performance(cumulative_reward, utility_function, qos, queue_len, cost, no_of_vm, steps, sla):
    # plot config
    plt.rc('axes', labelsize=15) 
    plt.rc('axes', titlesize=20)
    plt.rc('xtick', labelsize=15) 
    plt.rc('ytick', labelsize=15) 
    
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
