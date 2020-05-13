import numpy as np

class Environment():
    def __init__(self, length=3):
        self.length = length
        self.board = np.zeros((self.length, self.length))
        self.x = -1
        self.o = 1
        self.winner = None
        self.ended = False
        self.num_states = 3**(self.length*self.length)

    def draw_board(self):
        for i in range(self.length):
            print("-------------")
            for j in range(self.length):
                print(" ", end="")
                if self.board[i,j]==self.x:
                    print("x |", end="")
                elif self.board[i,j]==self.o:
                    print("o |", end="")
                else:
                    print("  |", end="")
            print("")
        print("-------------")

    def get_state(self):
        k = 0
        h = 0
        for i in range(self.length):
            for j in range(self.length):
                if self.board[i,j]==0:
                    v=0
                elif self.board[i,j]==self.x:
                    v=1
                elif self.board[i,j]==self.o:
                    v=2
                h+=(3**k)*v
                k+=1
        return h

    def game_over(self, force_recalculate=False):
        if not force_recalculate and self.ended:
            return self.ended
        
        for i in range(self.length):
            for player in (self.x, self.o):
                if self.board[i].sum()==player*self.length:
                    self.winner = player
                    self.ended = True
                    return True

        for j in range(self.length):
            for player in (self.x, self.o):
                if self.board[:,j].sum()==player*self.length:
                    self.winner = player
                    self.ended = True
                    return True

        for player in (self.x, self.o):
            if self.board.trace()==player*self.length:
                self.winner = player
                self.ended = True
                return True
            if np.fliplr(self.board).trace()==player*self.length:
                self.winner = player
                self.ended = True
                return True

        if np.all((self.board==0) == False):
            self.winner = None
            self.ended = True
            return True

        self.winner = None
        return False



    def reward(self, symbol):
        if not self.game_over():
            return 0
        return 1 if self.winner==symbol else 0

    def is_empty(self, i, j):
        return self.board[i,j]==0


class Player():
    def __init__(self, eps=0.1, alpha=0.5):
        self.eps = eps
        self.alpha = alpha
        self.verbose = False
        self.state_history = []

    def set_V(self, V):
        self.V = V

    def set_symbol(self, symbol):
        self.symbol = symbol

    def set_verbose(self, v):
        self.verbose = v

    def reset_history(self):
        self.state_history = []

    def take_action(self, env, training):
        r = np.random.rand()
        if training:
            epsilon = self.eps
        else:
            epsilon = 0.0
        if r<epsilon:
            if self.verbose:
                print("Taking a random action...")

            possible_moves = []
            for i in range(env.length):
                for j in range(env.length):
                    if env.is_empty(i,j):
                        possible_moves.append((i,j))
            idx = np.random.choice(len(possible_moves))
            next_move = possible_moves[idx]
        else:
            pos2value = {}
            next_move = None
            best_value = -1
            for i in range(env.length):
                for j in range(env.length):
                    if env.is_empty(i,j):
                        env.board[i,j] = self.symbol
                        state = env.get_state()
                        env.board[i,j] = 0
                        pos2value[(i,j)] = self.V[state]
                        if self.V[state] > best_value:
                            best_value = self.V[state]
                            next_move = (i,j)

            if self.verbose:
                print("Taking a greedy action...")
                for i in range(env.length):
                    print("-----------------")
                    for j in range(env.length):
                        if env.is_empty(i,j):
                            print(f" {pos2value[(i,j)]:.2f}|", end="")
                        else:
                            print("  ", end="")
                            if env.board[i,j]==env.x:
                                print("x  |", end="")
                            elif env.board[i,j]==env.o:
                                print("o  |", end="") 
                            else:
                                print("   |", end="")
                    print("")
                print("-----------------")

        env.board[next_move[0], next_move[1]] = self.symbol


    def update(self, env):
        reward = env.reward(self.symbol)
        target = reward
        for prev in reversed(self.state_history):
            value = self.V[prev] + self.alpha*(target-self.V[prev])
            self.V[prev] = value
            target = value
        self.reset_history()

    def update_state_history(self, state):
        self.state_history.append(state)


class Human():
    def __init__(self):
        pass

    def set_symbol(self, symbol):
        self.symbol = symbol

    def take_action(self, env, training=False):
        while True:
            move = input("Enter coordinates for your next move (i,j=0,1,2): ")
            i, j = move.split(",")
            i = int(i)
            j = int(j)
            if env.is_empty(i,j):
                env.board[i,j]=self.symbol
                break

    def update(self, env):
        pass

    def update_state_history(self, state):
        pass


def play_game(p1, p2, env, draw=False, training=False):

    current_player = None
    while not env.game_over():

        if current_player==p1:
            current_player = p2
        else:
            current_player = p1

        if draw:
            if draw==1 and current_player==p1:
                env.draw_board()
            if draw==2 and current_player==p2:
                env.draw_board()

        current_player.take_action(env, training)

        state = env.get_state()
        p1.update_state_history(state)
        p2.update_state_history(state)
        
    if draw:
        env.draw_board()

    p1.update(env)
    p2.update(env)

def get_state_hash_and_winner(env, i=0, j=0):
    results = []
    for v in (0, env.x, env.o):
        env.board[i,j] = v
        if j==env.length-1:
            if i==env.length-1:
                state = env.get_state()
                ended = env.game_over(force_recalculate=True)
                winner = env.winner
                results.append((state, winner, ended))
            else:
                results += get_state_hash_and_winner(env, i+1, 0)
        else:
            results += get_state_hash_and_winner(env, i, j+1)

    return results

def initialV_x(env, state_winner_triples):
    V = np.zeros(env.num_states)
    for state, winner, ended in state_winner_triples:
        if ended:
            if winner==env.x:
                v=1
            else:
                v=0
        else:
            v=0.5
        V[state]=v

    return V

def initialV_o(env, state_winner_triples):
    V = np.zeros(env.num_states)
    for state, winner, ended in state_winner_triples:
        if ended:
            if winner==env.o:
                v=1
            else:
                v=0
        else:
            v=0.5
        V[state]=v

    return V


if __name__ == "__main__":
    p1 = Player()
    p2 = Player()

    env = Environment(length=3)
    state_winner_triples = get_state_hash_and_winner(env)

    Vx = initialV_x(env, state_winner_triples)
    p1.set_V(Vx)
    Vo = initialV_o(env, state_winner_triples)
    p2.set_V(Vo)

    p1.set_symbol(env.x)
    p2.set_symbol(env.o)

    env.draw_board()
    T = 10000
    print("TRAINING TIC-TAC-TOE PLAYER...")
    for t in range(T):
        if (t+1)%200:
            print(f"{t+1}/{T}")
        play_game(p1, p2, Environment(length=3), training=True)


    print("MACHINE IS READY TO DEFEAT HUMAN")
    human = Human()
    human.set_symbol(env.o)
    while True:
        p1.set_verbose(True)
        play_game(p1, human, Environment(length=3), draw=2)
        answer = input("Play again [y/n]: ")
        if answer and answer.lower()=="n":
            break