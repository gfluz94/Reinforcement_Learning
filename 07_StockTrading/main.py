from stockmarket import Environment, Agent, Account

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

def get_scaler(env, train_idx):
  states = []
  for _ in range(train_idx-1):
    action = env.actions[np.random.randint(low=0, high=len(env.actions))]
    state, _, _ = env.perform(action)
    states.append(state)

  scaler = StandardScaler()
  scaler.fit(states)
  return scaler

def create_model(input_dim, n_output):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(n_output, input_shape=[input_dim]))
    model.compile(loss="mse", optimizer="adam")
    return model
    

def play_one_episode(agent, env, train_mode=False):
    s = env.reset()
    agent.reset_history()
    done = False
    loss = 0
    while not done:
        action = agent.get_action(s, train_mode)
        next_s, r, done = env.perform(action)
        if train_mode:
            loss += agent.train(s, action, r, next_s, done)
        s = next_s
    agent.states_history.append(s)
    return env._calculate_portfolio(), loss/env.n_step

def plot_training(portfolio_values, loss):
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(16,8))
    ax1.set_title("Portfolio during Training")
    ax1.plot(portfolio_values)
    ax2.set_title("Loss during Training")
    ax2.plot(np.arange(start=1, stop=len(loss)+1),loss)
    fig.savefig("training_rl.png", dpi=1200)
    plt.show()

def plot_test(stock_prices, stock_names, actions_test, states_test):
    fig, (ax1, ax2, ax3) = plt.subplots(3,1,figsize=(16,8))
    for i, ax in enumerate((ax1, ax2, ax3)):
        ax.set_title(f"Stock Price - {stock_names[i]}")
        ax.plot(stock_prices[:,i], color="lightblue")
        sell_points_x = []
        sell_points_y = []
        buy_points_x = []
        buy_points_y = []
        for j, a in enumerate(actions_test[:,i]):
            if a==0 and states_test[j, len(stock_names)+i]>0:
                sell_points_x.append(j)
                sell_points_y.append(stock_prices[j,i])
            elif a==2:
                buy_points_x.append(j)
                buy_points_y.append(stock_prices[j,i])
        ax.scatter(sell_points_x, sell_points_y, color="red", marker="*", label="Sell Time")
        ax.scatter(buy_points_x, buy_points_y, color="blue", marker="*", label="Buy Time")
        ax.legend()
    plt.tight_layout()
    fig.savefig("predicting_rl.png", dpi=1200)
    plt.show()


if __name__ == "__main__":

    df = pd.read_csv("stocks.csv")
    stock_names = df.columns
    stock_prices = df.values
    train_index = int(len(df)*0.7)
    train_prices = stock_prices[:train_index, :]
    test_prices = stock_prices[train_index:, :]

    initial_investment = 20000
    my_account = Account(initial_investment)
    env = Environment(train_prices, my_account)

    scaler = get_scaler(env, train_index)
    with open("scaler.pkl", "wb") as file:
        pickle.dump(scaler, file)
    model = create_model(2*env.n_stocks+1, len(env.actions))
    agent = Agent(model, scaler)

    portfolio_values = [env.cash_in_hand]
    loss = []
    num_episodes = 2000
    print("TRAINING...")
    for i in range(num_episodes):
        if (i+1)%50==0:
            print(f"{i+1}/{num_episodes}")
            agent.model.save(f"stock_model_{i+1}.h5")
        val, l = play_one_episode(agent, env, train_mode=True)
        portfolio_values.append(val)
        loss.append(l)
    agent.model.save("stock_model.h5")
    print("TRAINING FINISHED!\n")

    plot_training(portfolio_values, loss)

    initial_investment = 20000
    my_account = Account(initial_investment)
    env = Environment(test_prices, my_account)
    val, _ = play_one_episode(agent, env)
    actions_test = np.array(agent.actions_history)
    states_test = np.array(agent.states_history)
    
    plot_test(test_prices, stock_names, actions_test, states_test)

    print(f"\nFINAL PORTFOLIO: ${val:.2f}")
    print(f"\tCash: ${states_test[-1,-1]:.2f}")
    for i, name in enumerate(stock_names):
        price = states_test[-1, i]
        qte = states_test[-1, env.n_stocks+i]
        total = price*qte
        print(f"\t{name}: {int(qte)} stocks at ${price:.2f} [${total:.2f}]")