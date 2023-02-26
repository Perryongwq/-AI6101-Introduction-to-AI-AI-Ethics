from environment import CliffBoxPushingBase
from collections import defaultdict
import numpy as np
import random
import matplotlib.pyplot as plt

class QAgent(object):
    def __init__(self):
        self.action_space = [1,2,3,4]
        self.V = {}
        self.Q = defaultdict(lambda: np.zeros(len(self.action_space)))
        self.discount_factor=0.99
        self.alpha=0.5
        self.epsilon=0.01

    def take_action(self, state):
        if random.random() < self.epsilon/10:
            action = random.choice(self.action_space)
        else:
            action = self.action_space[np.argmax(self.Q[state])]
        return action

    # implement your train/update function to update self.V or self.Q
    # you should pass arguments to the train function
    def train(self, state, action, next_state, reward):
        # print(state, action, next_state, reward)

        #temporal difference rule: Q-learning
        best_next = np.argmax(self.Q[next_state])
        target = reward + self.discount_factor*self.Q[next_state][best_next]
        diff = target - self.Q[state][action-1]
        self.Q[state][action-1] += self.alpha*diff

        # keep track of the motions
        self.V[state[0]] = action

        # r = self.Q.items()
        # print(sorted(r))

if __name__ == '__main__':
    env = CliffBoxPushingBase()
    # you can implement other algorithmss
    agent = QAgent()
    rewards = []
    time_step = 0
    initial_epsilon=0.01 #initial epsilon greedy for the agent
    num_iterations = 5000 #increased iterations cos needed more training time
    hist = []
    ep_lengths = []
    agent.epsilon=initial_epsilon

    for i in range(num_iterations):
        terminated = False
        env.reset()
        agent.V = {} # reset agent policy after each iteration
        bestV = {a:a for a in range(75)} # giant placeholder dict to get replaced in a bit
        while not terminated:
            state = env.get_state()
            action = agent.take_action(state)
            # print(action)
            reward, terminated, _ = env.step([action])
            next_state = env.get_state()
            rewards.append(reward)
            # print(f'step: {time_step}, actions: {action}, reward: {reward}')
            time_step += 1
            agent.train(state, action, next_state, reward)
            if len(agent.V)<len(bestV):
                # chekcs if current path was most efficient
                bestV = agent.V
        if i%250 == 0:
            print(f'Iteration #{i+250} of {num_iterations}. Rewards: {sum(rewards)}')
        hist.append(sum(rewards))
        ep_lengths.append(len(env.episode_actions))
        teminated = False
        rewards = []
    from clipboard import copy
    # -------------  Graph 1 ------------------
    plt.figure(1)
    hist2 = []
    for i, x in enumerate(hist):
        if i%20==0:
            hist2.append(x)
    # copy(str(hist2)) # used to inspect data and figure out plotting
    

    plt.plot([x*10 for x in range(len(hist2))], hist2, 'r',linewidth=0.75)
    plt.ylabel("Episode Rewards")
    plt.xlabel('Episode')
    plt.title("Episode Rewards Over Time (Smoothed 10x)")
    # plt.axis([0,250,0, -3350])
    # plt.show()
    plt.savefig("Episode Rewards Over Time.png", dpi=900)

    # --------------------- Graph 2 ------------------
    plt.figure(2)
    eplen2 = []
    for i, x in enumerate(ep_lengths):
        if i%20==0:
            eplen2.append(x)
    # copy(str(eplen2))
    plt.plot([x*10 for x in range(len(hist2))], eplen2, 'g',linewidth=0.75)
    plt.ylabel("Episode Length")
    plt.xlabel('Episode')
    plt.title("Episode Length Over Time (Smoothed 10x)")
    # plt.axis([0,250,0, -3350])
    # plt.show()
    plt.savefig("Episode Length Over Time.png", dpi=900)
    print(f'print the historical actions: {env.episode_actions}')

    # ----------------------- Print World -----------------------
    def printgrid(grid):
        '''Prints out character arrays in a nice way'''
        for i,x in enumerate(grid):
            for y in x:
                print(y.decode(), end='  ')
            print("\n")

    print("Final V Table:")
    printgrid(env.world)
    grid = env.world
    key = {1:'^', 2:'v', 3:'<', 4:'>'}
    for c in bestV:
        x,y = c
        grid[x][y] = key[bestV[c]]
    # copy(str(grid))
    print("Most Successful Policy:")
    printgrid(grid)
    # print(list(grid))