import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Environment.simulator import *


def viz_performance(cumulative_reward, utility_function, qos, no_of_vm, steps, sla, queue_avg):
    # plot config
    plt.rc('axes', labelsize=12)
    plt.rc('axes', titlesize=15)
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)

    figure, axis = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))

    axis[0, 0].plot(steps, cumulative_reward)
    #axis[0, 0].plot(steps, queue_avg, color='gray', linestyle='--')
    axis[0, 0].set_title("Cumulative reward")

    axis[0, 1].plot(steps, utility_function)
    #axis[0, 1].plot(steps, queue_avg, color='gray', linestyle='--')
    axis[0, 1].set_title("Utility function")

    axis[1, 0].plot(steps, qos)
    #axis[1, 0].plot(steps, queue_avg, color='gray', linestyle='--')
    axis[1, 0].axhline(y=sla, color='r', linestyle='-')
    axis[1, 0].set_title("QoS & SLA")

    axis[1, 1].plot(steps, no_of_vm)
    #axis[1, 1].plot(steps, queue_avg, color='gray', linestyle='--')
    axis[1, 1].set_title("No of virtual machines in use")

    plt.show()


def print_stats(value, quality_of_service, state_lst, no_of_vm_lst):
    print("Cumulative reward at the end of the test: ", value[-1])
    print("Avg cost minus penalty per step: ", np.round(np.mean(state_lst), 2))
    print("Avg quality of service: ", np.round(np.mean(quality_of_service), 2))
    print("Avg no of virtual machines: ", np.mean(no_of_vm_lst))
    
# ------------------------------------------------------------------------------------------------- TEST FOR STABLE BASELINES
def test_model_SB(model, SLA, VM_TYPE1, VM_TYPE2, VM_TYPE3, STEPS, SCENARIO, PENALTY, ALPHA, BETA):
    env = Sim(sla = SLA,
              vm_1 = VM_TYPE1,
              vm_2 = VM_TYPE2,
              vm_3 = VM_TYPE3,
              steps = STEPS,
              penalty = PENALTY,
              scenario = SCENARIO,
              alpha = ALPHA,
              beta = BETA)
    
    # restart env
    obs = env.reset()
    done = False

    culumative_reward = 0
    step = 0
    value = []
    quality_of_service = []
    utility_function = []
    no_of_vm_lst = []
    steps_lst = []
    queues = {}


    while True:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)

        # save the resuls
        culumative_reward+=rewards
        step +=1
        steps_lst.append(step)
        value.append(culumative_reward)
        queues[step] = info['Semi-stochastic queue']
        quality_of_service.append(info['Quality of Service'])
        utility_function.append(obs)
        no_of_vm_lst.append(info['VM type I'] + info['VM type II'] + info['VM type III'])

        if done: 
            print('info', info)
            break
            
    # average queue
    queue_df = pd.DataFrame(queues)
    queue_avg = list(queue_df.mean(axis=1))
    utility_function = list(np.reshape(utility_function, [STEPS,]))
    # return value, utility_function, quality_of_service, no_of_vm_lst, steps_lst, queue_avg
    viz_performance(cumulative_reward=value,
                       utility_function=utility_function,
                       qos=quality_of_service,
                       no_of_vm = no_of_vm_lst,
                       steps=steps_lst,
                       sla=SLA,
                       queue_avg=queue_avg)

    print_stats(value, quality_of_service, utility_function, no_of_vm_lst)
    
# ------------------------------------------------------------------------------------------------- TEST FOR DDQN AND D3QN
    
def test_model_Qlearning(best_net, SLA, VM_TYPE1, VM_TYPE2, VM_TYPE3, STEPS, SCENARIO, PENALTY, ALPHA, BETA):
    
    sim = Sim(sla = SLA,
              vm_1 = VM_TYPE1,
              vm_2 = VM_TYPE2,
              vm_3 = VM_TYPE3,
              scenario = SCENARIO,
              penalty = PENALTY,
              steps = STEPS,
              alpha = ALPHA,
              beta = BETA)

    state = sim.reset()
    state = np.reshape(state, [1, 1])

    # lists for results
    culumative_reward = 0
    step = 0
    value = []
    quality_of_service = []
    utility_function = []
    queue_len = []
    cost_lst = []
    no_of_vm_lst = []
    steps_lst = []
    queues = {}


    while True:
                # Get action to take (se eval mode to avoid dropout layers)
                best_net.eval()
                action = best_net.act(state)
                # Act
                state_next, reward, terminal, info = sim.step(action)
                # Reshape state into 2D array with state observations as first 'row'
                state_next_re = np.reshape(state_next, [1, 1])
                # Update state
                state = state_next_re

                # save the resuls
                culumative_reward+=reward
                step +=1
                value.append(culumative_reward)
                queues[step] = info['Semi-stochastic queue']
                steps_lst.append(step)
                quality_of_service.append(info['Quality of Service'])
                utility_function.append(state_next)
                no_of_vm_lst.append(info['VM type I'] + info['VM type II'] + info['VM type III'])

                if terminal: 
                    print('info', info)
                    break
                    
    queue_df = pd.DataFrame(queues)
    queue_avg = list(queue_df.mean(axis=1))
    utility_function = list(np.reshape(utility_function, [STEPS,]))
    viz_performance(cumulative_reward=value,
                   utility_function=utility_function,
                   qos=quality_of_service,
                   no_of_vm = no_of_vm_lst,
                   steps=steps_lst,
                   sla=SLA,
                   queue_avg=queue_avg)
    
    print_stats(value, quality_of_service, utility_function, no_of_vm_lst)
