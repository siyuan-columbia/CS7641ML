import numpy as np
import gym
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings("ignore")

def value_iteration(env, max_iterations = 1000, gamma=0.99, epsilon=1e-6):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    V = np.zeros(n_states)
    value_history = [V.copy()]
    episode_time=[]
    interim_policy = np.zeros(n_states, dtype=int)
    policy_history = []

    algo_start_time= time.time()
    for _ in range(max_iterations):
        episode_start_time= time.time()
        delta = 0
        for state in range(n_states):
            max_value = float('-inf')
            action_value=[]
            for action in range(n_actions):
                value = 0
                for prob, next_state, reward, done in env.P[state][action]:
                    value += prob * (reward + gamma * V[next_state])
                action_value.append(value)
                max_value = max(max_value, value)
            delta = max(delta, abs(max_value - V[state]))
            interim_policy[state] = np.argmax(action_value)
            V[state] = max_value
        policy_history.append(interim_policy.copy())
        value_history.append(V.copy())
        if delta < epsilon:
            break
        episode_end_time= time.time()
        episode_time.append(episode_end_time-episode_start_time)
    algo_end_time= time.time()
    print("Time taken for VI for {}: ".format(env.spec.id), algo_end_time-algo_start_time)
    final_policy = np.zeros(n_states, dtype=int)
    for s in range(n_states):
        final_policy[s] = np.argmax([sum([p * (r + gamma * V[s_]) for p, s_, r, _ in env.P[s][a]]) for a in range(n_actions)])

    return value_history, final_policy, policy_history, episode_time


def policy_iteration(env, max_iterations = 1000,gamma=0.99,epsilon=1e-6,iter_stop_sign = -10):
    def compute_value(policy, state):
        action = policy[state]
        value = 0
        for prob, next_state, reward, done in env.P[state][action]:
            value += prob * (reward + gamma * V[next_state])
        return value

    n_states = env.observation_space.n
    n_actions = env.action_space.n
    V = np.zeros(n_states)
    policy = np.zeros(n_states, dtype=np.int)
    policy_history = [policy.copy()]
    value_history = [V.copy()]
    algo_start_time= time.time()
    episode_time=[]

    for _ in range(max_iterations):
        episode_start_time= time.time()
        # Policy evaluation
        while True:
            delta = 0
            for state in range(n_states):
                old_value = V[state]
                V[state] = compute_value(policy, state)
                delta = max(delta, abs(old_value - V[state]))
            if delta < epsilon:
                break

        # Policy improvement
        for state in range(n_states):
            max_value = float('-inf')
            best_action = None
            for action in range(n_actions):
                value = 0
                for prob, next_state, reward, done in env.P[state][action]:
                    value += prob * (reward + gamma * V[next_state])
                if value > max_value:
                    max_value = value
                    best_action = action

            policy[state] = best_action
        policy_history.append(policy.copy())
        value_history.append(V.copy())
        if np.all([policy == previous_policy for previous_policy in policy_history[iter_stop_sign:]]): #policy stable after 10 iterations
            break

        episode_end_time= time.time()
        episode_time.append(episode_end_time-episode_start_time)
    algo_end_time= time.time()
    print("Time taken for PI for {}: ".format(env.spec.id), algo_end_time-algo_start_time)

    return value_history, policy_history,episode_time

def q_learning(env, episodes=5000, alpha=0.1, gamma=0.99, epsilon=0.1, strategy='non-greedy'):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))
    episode_time = []
    policy_history = []
    V = np.zeros(n_states)
    value_history = [V.copy()]
    algo_start_time = time.time()
    for episode in range(episodes):
        episode_start_time = time.time()
        state = env.reset()[0]
        while True:
            if strategy == 'non-greedy':
                action = np.argmax(Q[state]) if np.random.rand() > epsilon else env.action_space.sample()
            elif strategy == 'greedy':
                action = np.argmax(Q[state])
            next_state, reward, done, info, _ = env.step(action)
            target = reward + gamma * np.max(Q[next_state])
            Q[state][action] += alpha * (target - Q[state][action])
            if done:
                break
            state = next_state
        episode_end_time = time.time()
        episode_time.append(episode_end_time - episode_start_time)
        policy_history.append(np.argmax(Q, axis=1))
        value_history.append(np.max(Q, axis=1))
    final_policy = np.argmax(Q, axis=1)
    print("Time taken for Q-learning for {} {}: ".format(env.spec.id,strategy), time.time() - algo_start_time)

    return value_history, final_policy, policy_history, episode_time

def plot_learning_curve(value_history, env_name, epsilon=0.1):
    value_improvement = np.sum(np.abs(np.diff(value_history, axis=0)), axis=1)
    plt.plot(value_improvement, label=env_name)
    plt.xlabel("Iteration")
    plt.ylabel("Value Improvement")
    plt.title("{} Learning Curve".format(env_name))
    plt.legend()
    plt.savefig("result/{}_learning_curve_{}.png".format(env_name, epsilon))
    plt.clf()
def plot_episode_time(episode_time, env_name, epsilon=0.1):
    plt.plot(episode_time, label=env_name)
    plt.xlabel("Iteration")
    plt.ylabel("Time")
    plt.title("{} Episode Time".format(env_name))
    plt.legend()
    plt.savefig("result/{}_time_{}.png".format(env_name, epsilon))
    plt.clf()

def plot_policy_change(policy_history, env_name, algo_name, epsilon=0.1):
    def compute_policy_changes(policy_history):
        changes = []
        for i in range(1, len(policy_history)):
            changes.append(np.sum(policy_history[i] != policy_history[i - 1]))
        return changes
    changes = compute_policy_changes(policy_history)
    plt.plot(changes, label=algo_name)
    plt.xlabel("Iteration")
    plt.ylabel("Policy Changes")
    plt.title("Policy Stability {}_{}".format(env_name, algo_name))
    plt.legend()
    plt.savefig("result/{} {}_policy_change_.png".format(env_name, algo_name, epsilon))
    plt.clf()
def discretize_state(obs, state_bins): #for mountain car problem
    position, velocity = obs
    position_idx = np.digitize(position, state_bins[0])
    velocity_idx = np.digitize(velocity, state_bins[1])
    return position_idx, velocity_idx

# Create Taxi-v3 environment
print('START: Taxi service problem')
taxi_env = gym.make('Taxi-v3')
# Value Iteration
V_taxi_vi,final_policy_taxi_vi,policy_history_taxi_vi, episode_time_taxi_vi = value_iteration(taxi_env)
plot_learning_curve(V_taxi_vi, "Taxi VI")
plot_episode_time(episode_time_taxi_vi, "Taxi VI")
plot_policy_change(policy_history_taxi_vi, "Taxi", "VI")
print('finished value iteration')


# Policy Iteration
V_taxi_pi, policy_history_taxi_pi,episode_time_taxi_pi = policy_iteration(taxi_env)
plot_learning_curve(V_taxi_pi, "Taxi PI")
plot_episode_time(episode_time_taxi_pi, "Taxi PI")
plot_policy_change(policy_history_taxi_pi, "Taxi", "PI")
print('finished policy iteration')


# Q-learning - non-greedy
V_taxi_ql_non_greedy, final_policy_taxi_ql_non_greedy, policy_history_taxi_ql_non_greedy, episode_time_taxi_ql_non_greedy= q_learning(taxi_env, strategy='non-greedy', epsilon=0.25)
plot_learning_curve(V_taxi_ql_non_greedy, "Taxi QL-non-greedy",epsilon=0.25)
plot_episode_time(episode_time_taxi_ql_non_greedy, "Taxi QL-non-greedy",epsilon=0.25)
plot_policy_change(policy_history_taxi_ql_non_greedy, "Taxi", "QL-non-greedy", epsilon=0.25)
print('finished q-learning non-greedy')

# Q-learning - greedy
V_taxi_ql_greedy, final_policy_taxi_ql_greedy, policy_history_taxi_ql_greedy, episode_time_taxi_ql_greedy= q_learning(taxi_env, strategy='greedy')
plot_learning_curve(V_taxi_ql_greedy, "Taxi QL-greedy")
plot_episode_time(episode_time_taxi_ql_greedy, "Taxi QL-greedy")
plot_policy_change(policy_history_taxi_ql_greedy, "Taxi", "QL-greedy")
print('finished q-learning greedy')

##############START: Forest Management problem####################
print('START: Forest Management problem')
import mdptoolbox.example
def run_forest_management_with_model(num_state=2000, num_steps=10, method='Value Iteration'):
    print('running forest management with model {}'.format(method))
    P, R = mdptoolbox.example.forest(S=num_state)
    value_history = [0] * num_steps
    policy_history = [0] * num_steps
    iters = [0] * num_steps
    time_array = [0] * num_steps
    gamma_arr = [0] * num_steps
    for i in range(num_steps):
        if method == 'Value Iteration':
            pi = mdptoolbox.mdp.ValueIteration(P, R, (i + 0.5) / num_steps)
        elif method == 'Policy Iteration':
            pi = mdptoolbox.mdp.PolicyIteration(P, R, (i + 0.5) / num_steps)
        pi.run()
        gamma_arr[i] = (i + 0.5) / num_steps
        value_history[i] = np.mean(pi.V)
        policy_history[i] = pi.policy
        iters[i] = pi.iter
        time_array[i] = pi.time

    plt.plot(gamma_arr, time_array)
    plt.xlabel('Gammas')
    plt.title('Forest Management - {} - Running Time'.format(method))
    plt.ylabel('running time (s)')
    plt.grid()
    plt.savefig('result/Forest Management - {} - running time.png'.format(method))
    plt.clf()

    plt.plot(gamma_arr, value_history)
    plt.xlabel('Gammas')
    plt.ylabel('Average Rewards')
    plt.title('Forest Management - {} - Reward Analysis'.format(method))
    plt.grid()
    plt.savefig('result/Forest Management - {} - reward.png'.format(method))
    plt.clf()

    plt.plot(gamma_arr, iters)
    plt.xlabel('Gammas')
    plt.ylabel('# of Iterations')
    plt.title('Forest Management - {}- Convergence Analysis'.format(method))
    plt.grid()
    plt.savefig('result/Forest Management - {} - Convergence.png'.format(method))
    plt.clf()

def run_forest_management_model_free(num_state=2000, epsilons=[0.05, 0.15, 0.25, 0.5, 0.75, 0.95], method='e-greedy'):
    print('running forest management model free {}'.format(method))
    P, R = mdptoolbox.example.forest(S=num_state, p=0.01)
    value_history = []
    policy_history = []
    time_array = []
    Q_table = []
    rew_array = []
    for epsilon in epsilons:
        st = time.time()
        pi = mdptoolbox.mdp.QLearning(P, R, 0.95)
        end = time.time()
        pi.run(epsilon,method)
        rew_array.append(pi.reward_array)
        value_history.append(np.mean(pi.V))
        policy_history.append(pi.policy)
        time_array.append(end - st)
        Q_table.append(pi.Q)

    plt.plot(range(0, 10000), rew_array[0], label='epsilon={}'.format(epsilons[0]))
    plt.plot(range(0, 10000), rew_array[1], label='epsilon={}'.format(epsilons[1]))
    plt.plot(range(0, 10000), rew_array[2], label='epsilon={}'.format(epsilons[2]))
    plt.plot(range(0, 10000), rew_array[3], label='epsilon={}'.format(epsilons[3]))
    plt.plot(range(0, 10000), rew_array[4], label='epsilon={}'.format(epsilons[4]))
    plt.plot(range(0, 10000), rew_array[5], label='epsilon={}'.format(epsilons[5]))
    plt.legend()
    plt.xlabel('Iterations')
    plt.grid()
    plt.title('Forest Management - Q Learning-{} - Decaying Epsilon'.format(method))
    plt.ylabel('Average Reward')
    plt.savefig('result/Forest Management - QL-{} - reward.png'.format(method))
    plt.clf()


run_forest_management_with_model(method='Value Iteration')
run_forest_management_with_model(method='Policy Iteration')
run_forest_management_model_free(method='e-greedy')
run_forest_management_model_free(method='greedy')
print('finish')