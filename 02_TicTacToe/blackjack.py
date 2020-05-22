import numpy as np
from functools import reduce
import json


CARDS_VALUES = {
    "A": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "10": 10,
    "J": 10,
    "Q": 10,
    "K": 10
}

class Dealer():
    def __init__(self):
        self.available_cards = 4*list(CARDS_VALUES.keys())
        self.open_card = self.hit()
        self.hand = [self.open_card]
        self.get_sum()
        self.initial_sum = self.sum
        self.secret_card = self.hit()

    def get_sum(self):
        self.sum = reduce(lambda a,b: a+b, map(lambda x: CARDS_VALUES[x], self.hand))
        if usable_ace(self.hand):
            self.sum += 10

    def hit(self):
        new_card = np.random.choice(self.available_cards)
        self.available_cards.remove(new_card)
        return new_card

    def reset_deck(self):
        self.available_cards = 4*list(CARDS_VALUES.keys())
        self.hand = []
        self.open_card = self.hit()
        self.hand = [self.open_card]
        self.get_sum()
        self.initial_sum = self.sum
        self.secret_card = self.hit()

    def get_secret_card(self):
        self.hand.append(self.secret_card)
        self.get_sum()

    def get_cards(self):
        self.get_secret_card()
        while self.sum<17:
            new_card = self.hit()
            self.hand.append(new_card)
            self.get_sum()

    def print_status(self, player, initial_state=False):
        print(f">> Dealer's hand: {' | '.join(self.hand)}")
        print(f"   Dealer's sum: {self.sum}")
        print()
        if initial_state:
            print(f"   Player's hand: {' | '.join(player.hand)}")
            print(f"   Player's sum: {player.sum}")
            print()
        else:
            action = player.action+"s"
            print(f"   Player {action}: {' | '.join(player.hand)}")
            print(f"   Player's sum: {player.sum}")
            print()


class Player():
    def __init__(self, alpha=0.05):
        self.alpha = alpha
        self.states = self.get_states()
        self.actions = ["stay", "hit"]
        self.hand = []
        self.busted = False
        self.done = False
        self.state_history = []
        self.actions_history = []

    def get_states(self):
        player_sum_cards = np.arange(32)
        dealer_initial_card = np.arange(2, 12, 1)
        return [(p, d, ace) for d in dealer_initial_card for p in player_sum_cards for ace in [0,1]]

    def set_Q(self):
        self.Q = {state:{a:0 for a in self.actions} for state in self.states}

    def initial_hit(self, dealer):
        self.hand.append(dealer.hit())
        self.hand.append(dealer.hit())
        self.update_status()
        self.state_history.append(self.get_state(dealer))

    def get_state(self, dealer):
        return (self.sum, dealer.initial_sum, usable_ace(self.hand))

    def get_action(self, training):
        if training:
            return np.random.choice(self.actions)
        current_state = self.state_history[-1]
        if self.Q[current_state]["hit"]>self.Q[current_state]["stay"]:
            return "hit"
        return "stay"

    def update_Q(self, reward):
        gamma = 0.9
        for a, s in zip(reversed(self.actions_history), reversed(self.state_history[:-1])):
            old_Q = self.Q[s][a] 
            self.Q[s][a] = old_Q + self.alpha*(gamma*reward - old_Q)
            gamma *= 0.9

    def update_status(self):
        self.sum = reduce(lambda a,b: a+b, map(lambda x: CARDS_VALUES[x], self.hand))
        if usable_ace(self.hand):
            self.sum += 10
        if self.sum>21:
            self.busted = True
            self.done = True

    def perform_action(self, dealer, training=False):
        self.action = self.get_action(training)
        self.actions_history.append(self.action)
        if self.action=="hit":
            self.hand.append(dealer.hit())
        else:
            self.done=True
        self.update_status()
        self.state_history.append(self.get_state(dealer))

    def define_policy(self):
        self.policy = {state:sorted(actions.items(), key=lambda x: -x[1])[0][0] for (state, actions) in self.Q.items()}

    def reset_hand(self):
        self.hand = []
        self.busted = False
        self.done = False
        self.sum = 0

def play_game(dealer, player, training=False, verbose=False):
    player.initial_hit(dealer)
    if verbose:
        dealer.print_status(player, initial_state=True)
    while not player.done:
        player.perform_action(dealer, training=training)
        if verbose:
            dealer.print_status(player)

    if player.busted:
        reward = -1
    else:
        dealer.get_cards()
        if dealer.sum>21 or player.sum>dealer.sum:
            reward = 1
        else:
            reward = 0
        if verbose:
            dealer.print_status(player)
    if training:
        player.update_Q(reward)
    dealer.reset_deck()
    player.reset_hand()

def usable_ace(hand):
    return "A" in hand and reduce(lambda a,b: a+b, map(lambda x: CARDS_VALUES[x], hand))+10<=21


if __name__ == "__main__":
    dealer = Dealer()
    player = Player()
    player.set_Q()

    epochs=100000
    print("TRAINING STARTED")
    for i in range(epochs):
        if (i+1)%10000==0:
            print(f"Epoch {i+1}/{epochs}")
        play_game(dealer, player, training=True)
    print("TRAINING FINISHED\n")
    player.define_policy()
    print(f"{player.policy}\n")

    with open("blackjack_player.json", "r") as file:
        json.dump(player.policy, file)

    answer = input("Do you want to watch a round? [y/n] ")
    while answer=="y":
        play_game(dealer, player, verbose=True)
        answer = input("Another round? [y/n] ")