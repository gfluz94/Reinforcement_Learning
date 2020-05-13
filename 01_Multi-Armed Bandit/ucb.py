import numpy as np
import matplotlib.pyplot as plt

NUM_TRIALS = 10000
BANDIT_PROBABILITIES = [0.4, 0.5, 0.75]

class Bandit():
    def __init__(self, p):
        self.p = p
        self.p_estimate = 0
        self.N = 0

    def pull(self):
        return np.random.rand() < self.p
    
    def update(self, x):
        self.N += 1.0
        self.p_estimate = ((self.N-1)*self.p_estimate + x)/self.N

def ucb(estimated_mean, n_total, n_bandit):
    if n_total==0:
        return np.inf
    return estimated_mean+np.sqrt(2*np.log(n_total)/n_bandit)

def experiment():
    bandits = [Bandit(p) for p in BANDIT_PROBABILITIES]
    rewards = np.zeros(NUM_TRIALS)
    
    for i in range(NUM_TRIALS):
        index = np.argmax([ucb(b.p_estimate, i, b.N) for b in bandits])
        bandit = bandits[index]
        r = bandit.pull()
        bandit.update(r)
        rewards[i] = r

    print("--------- RESULTS e-GREEDY ---------")
    for i, b in enumerate(bandits):
        print(f"Mean estimate for bandit {i+1}: {b.p_estimate:.3f}")

    print(f"\nTotal reward: {rewards.sum()}")
    print(f"Overall win rate: {rewards.sum()/NUM_TRIALS}")

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