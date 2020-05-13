import numpy as np
from sklearn.preprocessing import PolynomialFeatures

class GridWorld():
    def __init__(self, width:int, height:int, start:tuple):
        self.width = width
        self.height = height
        self.start = start
        self.i = start[0]
        self.j = start[1]

    def set(self, rewards:dict, actions=["up", "down", "right", "left"], wall=None):
        self.actions = actions
        self.wall = wall
        self.terminal_states = list(rewards.keys())
        self.possible_actions = {(i,j):[] for i in range(self.height) for j in range(self.width) if (i,j)!=self.wall and (i,j) not in self.terminal_states}
        for (i, j) in self.possible_actions.keys():
            if "up" in self.actions and i-1>=0 and (i-1, j)!=self.wall:
                self.possible_actions[(i, j)].append("up")
            if "left" in self.actions and j-1>=0 and (i, j-1)!=self.wall:
                self.possible_actions[(i, j)].append("left")
            if "down" in self.actions and i+1<self.height and (i+1, j)!=self.wall:
                self.possible_actions[(i, j)].append("down")
            if "right" in self.actions and j+1<self.width and (i, j+1)!=self.wall:
                self.possible_actions[(i, j)].append("right")

        if self.game_over():
            raise ValueError("You cannot assign start state as terminal state.")
        self.rewards = {}
        for i in range(self.height):
            for j in range(self.width):
                if (i, j) in rewards.keys():
                    self.rewards[(i, j)] = rewards[(i,j)]
                else:
                    self.rewards[(i, j)] = -0.01

    def draw_grid(self):
        print()
        print("*---"*self.width, end="*\n")
        for i in range(self.height):
            for j in range(self.width):
                if (i,j)==(self.i, self.j):
                    print("| x ", end="")
                elif (i,j)==self.wall:
                    print("|///", end="")
                else:
                    print("|   ", end="")
            print("|")
            print("*---"*self.width, end="*\n")

    def current_state(self):
        return (self.i, self.j)

    def reset_game(self):
        self.i, self.j = self.start

    def game_over(self):
        return (self.i, self.j) in self.terminal_states


class Robot():
    def __init__(self, alpha, probability_action=1):
        self.probability_action = probability_action
        self.alpha = alpha
        self.state_history = []

    def set_theta(self):
        self.theta = np.random.normal(size=(6,1))

    def get_next_state(self, action, state, env, probability=1):
        if probability==1:
            if action=="up":
                return (state[0]-1, state[1])
            elif action=="down":
                return (state[0]+1, state[1])
            elif action=="right":
                return (state[0], state[1]+1)
            elif action=="left":
                return (state[0], state[1]-1)
        else:
            r = np.random.rand()
            residual = 1-probability
            if r<residual:
                if action=="up" or action=="down":
                    actions=["right", "left"]
                elif action=="right" or action=="left":
                    actions=["up", "down"]
                action = np.random.choice(actions)

            i, j = state
            if action=="up":
                if i-1<0 or (i-1,j)==env.wall:
                    return (i,j)
                return (i-1, j)
            elif action=="down":
                if i+1>=env.height or (i+1,j)==env.wall:
                    return (i,j)
                return (i+1, j)
            elif action=="right":
                if j+1>=env.width or (i,j+1)==env.wall:
                    return (i,j)
                return (i, j+1)
            elif action=="left":
                if j-1<0 or (i,j-1)==env.wall:
                    return (i,j)
                return (i, j-1)

        
    def take_action(self, env):
        current_state = env.current_state()
        possible_actions = env.possible_actions[current_state]
        maxV = -np.inf
        action = None
        for next_action in possible_actions:
            next_state = self.get_next_state(next_action, current_state, env)
            nextV = self.theta.T@PolynomialFeatures().fit_transform(np.array(next_state).reshape(1,-1)).reshape(-1,1)
            if nextV[0,0]>maxV:
                maxV = nextV[0,0]
                action = next_action
        next_state = self.get_next_state(action, current_state, env, self.probability_action)
        env.i, env.j = next_state

    def go_through_episode(self, env):
        current_position = env.current_state()
        self.state_history.append(current_position)
        while not env.game_over():
            possible_actions = env.possible_actions[current_position]
            action = np.random.choice(possible_actions)
            env.i, env.j = self.get_next_state(action, current_position, env, self.probability_action)
            current_position = env.current_state()
            self.state_history.append(current_position)
        self.update_theta(env)
        self.reset_state_history()
        env.reset_game()

    def update_theta(self, env):
        gamma = 0.9
        target = None
        new_v = 0
        for s in reversed(self.state_history):
            if target:
                feats = PolynomialFeatures().fit_transform(np.array(target).reshape(1,-1)).reshape(-1,1)
                new_v = env.rewards[target] + gamma*self.theta.T@feats
            feats = PolynomialFeatures().fit_transform(np.array(s).reshape(1,-1)).reshape(-1,1)
            self.theta += self.alpha*(new_v-self.theta.T@feats)*feats
            target = s

    def reset_state_history(self):
        self.state_history = []

    def print_learning(self, env):
        print()
        print("*--------"*env.width, end="*\n")
        for i in range(env.height):
            for j in range(env.width):
                if (i,j)==env.wall:
                    print("|////////", end="")
                else:
                    v = self.theta.T@PolynomialFeatures().fit_transform(np.array((i,j)).reshape(1,-1)).reshape(-1,1)
                    if v<0:
                        print(f"| {v[0,0]:.3f} ", end="")
                    else:
                        print(f"| +{v[0,0]:.3f} ", end="")
            print("|")
            print("*--------"*env.width, end="*\n")


    def final_policy(self, env):
        policy = {}
        for i in range(env.height):
            for j in range(env.width):
                s = (i,j)
                if s in env.terminal_states:
                    best_action = "$"
                elif s!=env.wall:
                    possible_actions = env.possible_actions[s]
                    maxV = -np.inf
                    for action in possible_actions:
                        next_s = self.get_next_state(action, s, env)
                        nextV = self.theta.T@PolynomialFeatures().fit_transform(np.array(next_s).reshape(1,-1)).reshape(-1,1)
                        if nextV[0,0]>maxV:
                            maxV = nextV[0,0]
                            best_action = action[0].upper()
                policy[s] = best_action
        print()
        print("*---"*env.width, end="*\n")
        for i in range(env.height):
            for j in range(env.width):
                if (i,j)==env.wall:
                    print("|///", end="")
                else:
                    print(f"| {policy[(i,j)]} ", end="")
            print("|")
            print("*---"*env.width, end="*\n")


def play_game(robot, env, draw=False):
    env.reset_game()
    while not env.game_over():
        if draw:
            env.draw_grid()
        
        robot.take_action(env)
    
    if draw:
        env.draw_grid()


if __name__ == "__main__":
    env = GridWorld(width=4, height=3, start=(2,0))
    rewards = {
        (0,3): 1,
        (1,3): -1
        }
    actions = ["up", "down", "right", "left"]
    env.set(rewards, actions, wall=(1,1))

    robot = Robot(alpha=0.001, probability_action=0.8)
    robot.set_theta()

    print("\nStarting learning process...")
    episodes = 10000
    for i in range(episodes):
        if (i+1)%100==0:
            print(f">> {i+1}/{episodes}")
        robot.go_through_episode(env)
    print("\nLearning process is finished...")

    robot.print_learning(env)

    print("\nNow let's see the algorithm in action...")
    play_game(robot, env, draw=True)

    wins = 0
    for i in range(episodes):
        play_game(robot, env)
        wins += (env.rewards[env.current_state()]==1)*1

    print(f"\nFINAL RESULT: {100*(wins/episodes):.2f}% of wins.")

    print("\nFinal Policy:")
    robot.final_policy(env)

