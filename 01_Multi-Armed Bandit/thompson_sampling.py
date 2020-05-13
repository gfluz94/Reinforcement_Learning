import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

NUM_TRIALS = 2000
BANDIT_PROBABILITIES = [0.25, 0.5, 0.75]

class Bandit():
    def __init__(self, p):
        self.p = p
        self.N = 0
        self.alfa = 1
        self.beta = 1

    def pull(self):
        return np.random.rand() < self.p
    
    def update(self, x):
        self.N += 1.0
        self.alfa += x
        self.beta += 1-x

    def sample(self):
        return np.random.beta(self.alfa, self.beta)

def experiment():
    bandits = [Bandit(p) for p in BANDIT_PROBABILITIES]
    rewards = np.zeros(NUM_TRIALS)
    
    for i in range(NUM_TRIALS):
        index = np.argmax([b.sample() for b in bandits])
        bandit = bandits[index]
        r = bandit.pull()
        bandit.update(r)
        rewards[i] = r

    print("--------- RESULTS e-GREEDY ---------")
    for i, b in enumerate(bandits):
        print(f"Mean estimate for bandit {i+1}: {np.median(np.random.beta(b.alfa, b.beta, size=1000)):.3f}")

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

    _, ax = plt.subplots(1,1,figsize=(10,5))
    ax.set_title("Posterior Distributions")
    df_bandits = pd.DataFrame({"Bandit "+str(i+1):np.random.beta(b.alfa, b.beta, size=1000) for i, b in enumerate(bandits)})
    for col in df_bandits.columns:
        sns.distplot(df_bandits[col], hist=False, label=col, ax=ax)
    ax.set_xlabel("Win Rate")
    ax.legend()
    plt.show()


if __name__ == "__main__":
   experiment()