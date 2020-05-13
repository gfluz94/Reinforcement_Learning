import numpy as np
from itertools import product

class Account():
    def __init__(self, balance):
        self.balance = balance
    
    def take(self, amount):
        if amount<=self.balance:
            self.balance-=amount

    def put(self, amount):
        self.balance+=amount


class Environment():
    def __init__(self, data, account):
        self.pointer = 0
        self.account = account
        self.stock_price_data = data
        self.n_step, self.n_stocks = self.stock_price_data.shape

        self.stock_owned = np.zeros(self.n_stocks)
        self.stock_price = self.stock_price_data[self.pointer, :]
        self.cash_in_hand = self.account.balance
        
        # 0 = sell / 1 = hold / 2 = buy
        self.actions = list(product([i for i in range(self.n_stocks)], repeat=self.n_stocks))
        self.actions = np.array(self.actions)

    def reset(self):
        self.pointer = 0
        self.stock_owned = np.zeros(self.n_stocks)
        self.stock_price = self.stock_price_data[self.pointer, :]
        self.account = Account(self.cash_in_hand)
        return self._get_s()

    def _get_s(self):
        output = [price for price in self.stock_price]
        for owned in self.stock_owned:
            output.append(owned)
        output.append(self.account.balance)
        return np.array(output)

    def _calculate_portfolio(self):
        return self.stock_owned.T@self.stock_price+self.account.balance


    def _trade(self, action):
        buy_idx = [i for i, a in enumerate(action) if a==2]
        for i, a in enumerate(action):
            if a==0:
                self.account.put(self.stock_price[i]*self.stock_owned[i])
                self.stock_owned[i] = 0
        if len(buy_idx)>0:
            while self.account.balance>0:
                old_major = self.account.balance
                for i in buy_idx:
                    old = self.account.balance
                    self.account.take(self.stock_price[i])
                    if self.account.balance!=old:
                        self.stock_owned[i] += 1
                if old_major==self.account.balance:
                    break


    def perform(self, action):
        assert action in self.actions, "Action not possible!"

        current_portfolio = self._calculate_portfolio()
        self._trade(action)
        self.pointer+=1
        self.stock_price = self.stock_price_data[self.pointer]
        new_portfolio = self._calculate_portfolio()
        
        next_state = self._get_s()
        reward = new_portfolio-current_portfolio
        done = (self.pointer==self.n_step-1)

        return next_state, reward, done

class Agent():
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler
        self.epsilon = 1
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.95
        self.actions_history = []
        self.states_history = []

    def get_action(self, state, train_mode=False):
        if np.random.rand()<self.epsilon and train_mode:
            idx = np.random.randint(low=0, high=self.model.layers[-1].units)
        else:
            inpt = self.scaler.transform(state.reshape(1,-1))
            idx = np.argmax(self.model.predict(inpt))
        first = idx//9
        second = (idx%9)//3
        third = (idx%9)%3
        action = np.array([first, second, third])
        self.states_history.append(state)
        self.actions_history.append(action)
        return action

    def train(self, s, a, r, next_s, done):
        if done:
            target = r
        if not done:
            inpt = self.scaler.transform(next_s.reshape(1,-1))
            target = r+self.gamma*np.max(self.model.predict(inpt))

        action_index = 0
        exp = 0
        for a_ in a[::-1]:
            action_index += (3**exp)*a_
            exp+=1
        
        inpt = self.scaler.transform(s.reshape(1,-1))
        pred = self.model.predict(inpt)
        true_label = self.model.predict(inpt)
        true_label[0,action_index] = target

        self.model.train_on_batch(inpt, true_label)

        loss = np.sum((true_label-pred)**2)

        if self.epsilon>self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss

    def reset_history(self):
        self.actions_history = []
        self.states_history = []