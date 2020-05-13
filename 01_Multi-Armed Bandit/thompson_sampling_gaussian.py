import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

NUM_TRIALS = 2000
BANDIT_MEANS = [1, 2, 3]

class Bandit():
    def __init__(self, true_mean):
        self.true_mean = true_mean
        self.tau = 1
        self.N = 0
        self.sum_x = 0
        self.m = 0
        self.lbda = 1

    def pull(self):
        return np.random.normal(loc=self.true_mean, scale=(1/self.tau)**0.5)
    
    def update(self, x):
        self.N += 1.0
        self.sum_x += x
        self.lbda += self.tau
        self.m = self.tau*self.sum_x/self.lbda
        

    def sample(self):
        return np.random.normal(loc=self.m, scale=(1/self.lbda)**0.5)

def experiment():
    bandits = [Bandit(p) for p in BANDIT_MEANS]
    rewards = np.zeros(NUM_TRIALS)
    
    for i in range(NUM_TRIALS):
        index = np.argmax([b.sample() for b in bandits])
        bandit = bandits[index]
        r = bandit.pull()
        bandit.update(r)
        rewards[i] = r

    print("--------- RESULTS e-GREEDY ---------")
    for i, b in enumerate(bandits):
        dist = np.random.normal(loc=b.m, scale=(1/b.lbda)**0.5, size=1000)
        print(f"Mean estimate for bandit {i+1}: {np.median(dist):.3f}")

    print(f"\nTotal reward: {rewards.sum():.2f}")
    print(f"Overall win rate: {rewards.sum()/NUM_TRIALS:.2f}")

    cumulative_rewards = rewards.cumsum()
    win_rates = cumulative_rewards/(np.arange(NUM_TRIALS)+1)
    _, (ax1, ax2) = plt.subplots(1,2,figsize=(10,5))
    ax1.set_title("Mean Reward")
    ax1.plot(win_rates, label="Mean Reward")
    ax1.plot(np.ones(NUM_TRIALS)*np.max(BANDIT_MEANS), color="red", label="Known Mean Reward")
    ax2.set_title("Bandit Choice")
    ax2.bar(["Bandit "+str(i+1) for i in range(len(bandits))], [b.N/NUM_TRIALS for b in bandits])
    ax1.legend()
    plt.show()

    _, ax = plt.subplots(1,1,figsize=(10,5))
    ax.set_title("Posterior Distributions")
    df_bandits = pd.DataFrame({"Bandit "+str(i+1):np.random.normal(loc=b.m, scale=(1/b.lbda)**0.5, size=1000) for i, b in enumerate(bandits)})
    for col in df_bandits.columns:
        sns.distplot(df_bandits[col], hist=False, label=col, ax=ax)
    ax.set_xlabel("Mean Reward")
    ax.legend()
    plt.show()


if __name__ == "__main__":
   experiment()