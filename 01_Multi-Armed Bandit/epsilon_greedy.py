import numpy as np
import matplotlib.pyplot as plt

NUM_TRIALS = 10000
EPS = 0.1
BANDIT_PROBABILITIES = [0.5, 0.4, 0.55, 0.3]

class Bandit():
    def __init__(self, p):
        self.p = p
        self.p_estimate = 0
        self.N = 0

    def pull(self):
        return np.random.rand() < self.p
    
    def update(self, x):
        self.N += 1
        self.p_estimate += (1/self.N)*(x-self.p_estimate)

def experiment():
    bandits = [Bandit(p) for p in BANDIT_PROBABILITIES]
    rewards = np.zeros(NUM_TRIALS)
    num_exploration = 0
    num_exploitation = 0
    
    for i in range(NUM_TRIALS):
        if np.random.rand() < EPS:
            num_exploration+=1
            index = np.random.randint(0, len(bandits))
        else:
            num_exploitation+=1
            index = np.argmax([b.p_estimate for b in bandits])
        bandit = bandits[index]
        r = bandit.pull()
        bandit.update(r)
        rewards[i] = r

    print("--------- RESULTS e-GREEDY ---------")
    for i, b in enumerate(bandits):
        print(f"Mean estimate for bandit {i+1}: {b.p_estimate:.3f}")

    print(f"\nTotal reward: {rewards.sum()}")
    print(f"Overall win rate: {rewards.sum()/NUM_TRIALS}")
    print(f"Total times exploration: {num_exploration}")
    print(f"Total times exploitation: {num_exploitation}")

    cumulative_rewards = rewards.cumsum()
    win_rates = cumulative_rewards/(np.arange(NUM_TRIALS)+1)
    _, (ax1, ax2) = plt.subplots(1,2,figsize=(10,5))
    ax1.set_title("Win Rate")
    ax1.plot(win_rates, label="Overall Win Rate")
    ax1.plot(np.ones(NUM_TRIALS)*np.max(BANDIT_PROBABILITIES), color="red", label="Maximum Bandit Win Rate")
    ax2.set_title("Bandit Choice")
    ax2.bar(["Bandit "+str(i+1) for i in range(len(bandits))], [b.N/NUM_TRIALS for b in bandits])
    ax1.legend()
    plt.show()


if __name__ == "__main__":
   experiment()