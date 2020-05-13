import numpy as np
import matplotlib.pyplot as plt

NUM_TRIALS = 10000

class Bandit():
    def __init__(self, mean):
        self.mean = mean
        self.mean_estimate = 0
        self.N = 0

    def pull(self):
        return np.random.randn() + self.mean
    
    def update(self, x):
        self.N += 1
        self.mean_estimate += (1/self.N)*(x-self.mean_estimate)

def experiment(bandit_means, epsilons):
    bandits = [Bandit(m) for m in bandit_means]
    rewards = np.zeros((NUM_TRIALS, len(epsilons)))
    num_exploration = 0
    num_exploitation = 0
    
    for j in range(len(epsilons)):
        for i in range(NUM_TRIALS):
            if np.random.rand() < epsilons[j]:
                num_exploration+=1
                index = np.random.randint(0, len(bandits))
            else:
                num_exploitation+=1
                index = np.argmax([b.mean_estimate for b in bandits])
            bandit = bandits[index]
            r = bandit.pull()
            bandit.update(r)
            rewards[i, j] = r

    print("--------- RESULTS e-GREEDY ---------")
    for i, b in enumerate(bandits):
        print(f"Mean estimate for bandit {i+1}: {b.mean_estimate:.3f}")

    print(f"\nTotal reward: {rewards.sum():.3f}")
    print(f"Total times exploration: {num_exploration//3}")
    print(f"Total times exploitation: {num_exploitation//3}")

    cumulative_rewards = rewards.cumsum(axis=0)
    win_rates = np.zeros(cumulative_rewards.shape)
    _, (ax1, ax2) = plt.subplots(1,2,figsize=(10,5))
    ax1.set_title("Mean Reward")
    for j, eps in enumerate(epsilons):
        win_rates[:,j] = cumulative_rewards[:,j]/(np.arange(NUM_TRIALS)+1)
        ax1.plot(win_rates[:,j], label=f"Epsilon = {eps}")
    ax1.plot(np.ones(NUM_TRIALS)*np.max(bandit_means), label="Known Mean Reward")
    ax2.set_title("Comparing epsilons")
    for j, eps in enumerate(epsilons):
        win_rates[:,j] = cumulative_rewards[:,j]/(np.arange(NUM_TRIALS)+1)
        ax2.semilogx(win_rates[:,j], label=f"Epsilon = {eps}")
    ax1.legend()
    ax2.legend()
    plt.show()


if __name__ == "__main__":
    epsilons = [0.01, 0.05, 0.1]
    means = [1.5, 2.5, 3.5]
    experiment(means, epsilons)