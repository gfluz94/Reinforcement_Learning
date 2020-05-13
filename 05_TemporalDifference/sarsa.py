import numpy as np

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
        self.possible_actions = {(i,j):[] for i in range(self.height) for j in range(self.width) if (i,j)!=self.wall}
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
                    self.rewards[(i, j)] = -0.1

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
    def __init__(self, probability_action=1):
        self.probability_action = probability_action
        self.state_history = []

    def set_Q(self, env):
        self.Q = {(i,j):{} for i in range(env.height) for j in range(env.width)}
        self.N = {(i,j):{} for i in range(env.height) for j in range(env.width)}
        for key in self.Q.keys():
            if key==env.wall:
                continue
            for action in env.possible_actions[key]:
                self.Q[key][action] = 0
                self.N[key][action] = 1
        

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

        
    def next_action(self, env, state, epsilon=-1):
        current_state = state
        possible_actions = env.possible_actions[current_state]
        if np.random.rand()<epsilon:
            action = np.random.choice(possible_actions)
        else:            
            maxQ = -np.inf
            action = None
            for next_action in possible_actions:
                next_state = self.get_next_state(next_action, current_state, env)
                for next_next_action in env.possible_actions[next_state]:
                    if self.Q[next_state][next_next_action]>maxQ:
                        maxQ = self.Q[next_state][next_next_action]
                        action = next_action
        return action

    def go_through_episode(self, env, epsilon, alpha):
        current_position = env.start
        self.state_history.append(current_position)
        action = self.next_action(env, current_position, epsilon)
        gamma = 0.9
        while not env.game_over():
            next_state = self.get_next_state(action, current_position, env, self.probability_action)
            r = env.rewards[next_state]
            next_action = self.next_action(env, next_state, epsilon)
            new_alpha = alpha/self.N[current_position][action]
            self.Q[current_position][action] += new_alpha*(r+gamma*self.Q[next_state][next_action]-self.Q[current_position][action])
            env.i, env.j = next_state
            current_position = next_state
            action = next_action
        env.reset_game()

    def take_action(self, env):
        s = env.current_state()
        action = robot.next_action(env, s)
        env.i, env.j = robot.get_next_state(action, s, env, self.probability_action)

    def reset_state_history(self):
        self.state_history = []

    def print_learning(self, env):
        policy = {}
        for i in range(env.height):
            for j in range(env.width):
                s = (i,j)
                if s in env.terminal_states:
                    best_action = "$"
                elif s!=env.wall:
                    possible_actions = env.possible_actions[s]
                    maxQ = -np.inf
                    for action in possible_actions:
                        next_s = self.get_next_state(action, s, env)
                        for next_next_action in env.possible_actions[next_s]:
                            if self.Q[next_s][next_next_action]>maxQ:
                                maxQ = self.Q[next_s][next_next_action]
                                best_action = action
                policy[s] = best_action
        print()
        print("*--------"*env.width, end="*\n")
        self.V = {}
        for i in range(env.height):
            for j in range(env.width):
                if (i,j)==env.wall:
                    print("|////////", end="")
                else:
                    if policy[(i,j)]!="$":
                        self.V[(i,j)] = self.Q[(i,j)][policy[(i,j)]]
                        if self.Q[(i,j)][policy[(i,j)]]>0:
                            print(f"| +{self.Q[(i,j)][policy[(i,j)]]:.3f} ", end="")
                        else:
                            print(f"| {self.Q[(i,j)][policy[(i,j)]]:.3f} ", end="")
                    else:
                        self.V[(i,j)] = 0
                        print(f"| +0.000 ", end="")
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
                        if self.V[next_s]>maxV:
                            maxV = self.V[next_s]
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

    robot = Robot(probability_action=1)
    robot.set_Q(env)

    print("\nStarting learning process...")
    episodes = 5000
    epsilon = 0.5
    for i in range(episodes):
        epsilon /= (i+1)
        alpha = 0.5
        if (i+1)%100==0:
            print(f">> {i+1}/{episodes}")
        robot.go_through_episode(env, epsilon, alpha)
    print("\nLearning process is finished...")



    print("\nLearning:")
    robot.print_learning(env)
    print("\nFinal Policy:")
    robot.final_policy(env)

