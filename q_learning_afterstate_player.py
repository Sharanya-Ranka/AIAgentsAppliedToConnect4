import numpy as np
import random
import pandas as pd
import re
import copy
import simplejson as json


class Afterstate:
    """
    Representation of an afterstate (State after an action is immediately taken)
    Several state-action pairs can map to the same afterstate
    Will also contain value estimate of the afterstate
    """

    def __init__(self, state, initial_q_value):
        self.state = state
        self.q_value = initial_q_value
        self.most_recent_action = None

    def __eq__(self, other):
        return other and self.state == other.state

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.state)

    def __str__(self):
        return f"State={self.state}\nMost recent action={self.most_recent_action}\nQvalue={self.q_value}"

    def __repr__(self):
        return f"State={self.state}\nMost recent action={self.most_recent_action}\nQvalue={self.q_value}"


class AfterstateDatabase:
    """
    Contains a collection of Afterstate objects in a set.
    Optionally should be able to load Afterstate values from a file (TODO)
    If agent requests an afterstate and its not there, initialise a new one and return that (also put it in the dict)
    """

    def __init__(self, initial_q_value):
        self.Afs_dict = {}
        self.Afs_num = len(self.Afs_dict)
        self.initial_q_value = initial_q_value

    # def base_n_to_base_10(self, n, nums):
    #     return np.dot(np.power(n, np.arange(0, len(nums), dtype=np.int64)), nums)

    # def compress_state(self, state):
    #     # Convert to 0-728 representation
    #     t0 = self.base_n_to_base_10(3, state[:, 0])
    #     t1 = self.base_n_to_base_10(3, state[:, 1])
    #     t2 = self.base_n_to_base_10(3, state[:, 2])
    #     t3 = self.base_n_to_base_10(3, state[:, 3])
    #     t4 = self.base_n_to_base_10(3, state[:, 4])
    #     # t0 = int(''.join(map(str, list(state[:,0]))), 3)
    #     # t1 = int(''.join(map(str, list(state[:,1]))), 3)
    #     # t2 = int(''.join(map(str, list(state[:,2]))), 3)
    #     # t3 = int(''.join(map(str, list(state[:,3]))), 3)
    #     # t4 = int(''.join(map(str, list(state[:,4]))), 3)

    #     # Convert state to a single number
    #     final_state_rep = self.base_n_to_base_10(729, [t0, t1, t2, t3, t4])

    #     return final_state_rep

    def compress_state(self, state):
        return np.sum(np.multiply(np.power(3, np.arange(0, np.prod(state.shape), dtype=np.int64).reshape(state.shape)), state))

    def convert_SA_to_Afs(self, state, action):
        """
        Converts a StateAction to an AfterState
        State, after an action represents a new State
        Afs is the representation of that Afterstate 
        """
        # print(f"Converting SA to Afs")
        # print(f"{state}")
        # In order to do this, we need to implement part of the game-logic
        # And then use the compression technique to convert the new state to an integer
        # State is the game_board representation

        afterstate = state.copy()
        afterstate.next_move(action)

        # zero_rows = np.where(state[:, action] == 0)[0]
        # # print(f"zero rows={zero_rows}")
        # # Action is guaranteed to be legal, hence we can avoid checks
        # afterstate = np.copy(state)
        # afterstate[zero_rows[0], action] = player_num
        # # print(f"Afterstate = {afterstate}")

        afterstate_rep = self.compress_state(afterstate.game_board)

        status = "IN-PLAY"

        if afterstate.game_over:
            if afterstate.player_won[state.next_player - 1]:
                status = "WON"
            elif afterstate.player_won[2 - state.next_player]:
                status = "LOST"
            else:
                status = "DRAW"

        return afterstate_rep, afterstate, status

    def get_Afs(self, state, action):
        # print(f"State={state}, Action={action}, player_num={player_num}")
        # Status can be "WON", "DRAW", "LOST", "IN-PLAY"
        afterstate_rep, afterstate, status = self.convert_SA_to_Afs(state, action)
        # print(f"Afterstate repr is {afterstate_rep}")

        if afterstate_rep in self.Afs_dict:
            # print(f"Afs found in dict")
            # Change the most recent action
            self.Afs_dict[afterstate_rep].most_recent_action = action
            return self.Afs_dict[afterstate_rep]

        else:
            # print(f"Couldn't find Afs in dict")
            given_q_val = self.initial_q_value
            if status == "WON":
                given_q_val = 1000
            elif status == "LOST":
                given_q_val = -1000
            elif status == "DRAW":
                given_q_val = -100

            temp = Afterstate(afterstate.game_board, given_q_val)
            self.Afs_dict[afterstate_rep] = temp
            self.Afs_dict[afterstate_rep].most_recent_action = action
            return temp

    def save_Afs_to_file(self, filepath):
        dict_to_store = {"state": [], "qval": []}
        for afs, afs_obj in self.Afs_dict.items():
            dict_to_store["state"].append(afs)
            dict_to_store["qval"].append(afs_obj.q_value)

        df = pd.DataFrame.from_dict(dict_to_store)
        df['state'] = df['state'].astype(np.int64)
        df.to_csv(filepath, index=False)

    def load_Afs_from_file(self, filepath):
        df = pd.read_csv(filepath)
        for index, row in df.iterrows():
            state = row['state']
            afs = Afterstate(state, row['qval'])
            # print(sa)

            self.Afs_dict[state] = afs


class ConstantEpsilonGreedyPolicy:
    """
    Implements the constant epsilon-greedy policy function
    Takes in a set of StateAction q_values and returns the action to take
    """

    def __init__(self, epsilon):
        self.epsilon = epsilon

    def get_policy_afs(self, afs_set):
        qvals = np.array([afs.q_value for afs in afs_set])
        if random.random() < self.epsilon:
            action_ind = random.choice(range(len(afs_set)))
            return afs_set[action_ind]
        else:
            action_ind = random.choice(np.where(qvals == np.amax(qvals))[0])
            return afs_set[action_ind]


class GreedyPolicy:
    """
    Implements the greedy policy function (required as a target policy for q-learning)
    Takes in a set of StateAction q_values and returns the action to take
    """

    def __init__(self):
        pass

    def get_policy_afs(self, afs_set):
        qvals = np.array([afs.q_value for afs in afs_set])

        action_ind = random.choice(np.where(qvals == np.amax(qvals))[0])
        return afs_set[action_ind]


class Q_Learning_Agent:
    """
    Represents the agent. Eventually the agent must make "intelligent" decisions
    The agent will play many games and form its estimates for "goodness" of state-action pairs (q_value)
    The agent will use a 'database' of state-action pairs
    """

    def __init__(self, alpha, gamma, behaviour_policy, target_policy):
        self.alpha = alpha
        self.gamma = gamma
        self.afs_db = AfterstateDatabase(initial_q_value=0)
        self.behaviour_policy = behaviour_policy
        self.target_policy = target_policy

        self.previous_afs = None
        self.current_afs_set = None

        self.num_games_played = 0
        self.num_games_won = 0
        self.num_games_drawn = 0
        self.value_updates = []
        # self.statistics = {'alpha':self.alpha, 'gamma':self.gamma, 'epsilon':self.behaviour_policy.epsilon, 'games':{}}

    def populate_current_afs_set(self, game_state):
        self.current_afs_set = [self.afs_db.get_Afs(game_state, action) for action in game_state.next_player_actions]

    def update_q_value(self, cur_reward, game_state):
        # print(f"Q-LEARNING Updating q-value")

        if self.previous_afs is not None:
            # print(f"Previous afs is {self.previous_afs}")

            # Handling wins, losses, draws (cur_state = None)
            if game_state is not None:

                # print(f"Current afs set={current_afs_set}\nQvals = {[a.q_value for a in current_afs_set]}")
                # print(f"Chosen q-val (for gamma term)={self.target_policy.get_policy_afs(current_afs_set).q_value}")

                gamma_term = self.gamma * self.target_policy.get_policy_afs(self.current_afs_set).q_value
            else:
                gamma_term = 0

            alpha_term = self.alpha * (cur_reward + gamma_term - self.previous_afs.q_value)
            # print(f"Changing q-value of \n{self.previous_afs}\n from {self.previous_afs.q_value} to ", end="")
            self.previous_afs.q_value += alpha_term
            # if self.num_games_played not in self.statistics['games']:
            #     self.statistics['games'][self.num_games_played] = {}
            #     # self.statistics['alpha'] = self.alpha
            #     self.statistics['games'][self.num_games_played]['update_values'] = []

            self.value_updates.append(alpha_term)

            # print(f"{self.previous_afs.q_value}")

    def get_next_afs(self, game_state):
        # print(f"Q-LEARNING Getting the next move")

        # current_afs_set = [self.afs_db.get_Afs(game_state, action) for action in game_state.next_player_actions]
        # print(f"Q values are {[a.q_value for a in current_afs_set]}")
        afs_to_take = self.behaviour_policy.get_policy_afs(self.current_afs_set)
        # print(f"Afs taken=\n{afs_to_take}")

        return afs_to_take

    def update_previous_afs(self, next_afs):
        self.previous_afs = next_afs

    def change_epsilon(self, new_epsilon_value):
        self.behaviour_policy.epsilon = new_epsilon_value
        self.statistics['epsilon'] = new_epsilon_value

    def change_alpha(self, new_alpha_value):
        self.alpha = new_alpha_value
        self.statistics['alpha'] = new_alpha_value


class Q_Learning_Interface:

    def __init__(self, agent, rewards, update_num=100):
        self.agent = agent
        self.game_not_over_reward = rewards['game_not_over']
        self.won_reward = rewards['won']
        self.lost_reward = rewards['lost']
        self.draw_reward = rewards['draw']

        self.update_num = update_num
        self.stat_update_num = 1
        self.track_stats = False
        self.statistics = {
            "alpha": self.agent.alpha,
            "gamma": self.agent.gamma,
            "epsilon": self.agent.behaviour_policy.epsilon,
            "rewards": rewards,
            "games_info": {}
        }

        pass

    def base_n_to_base_10(self, n, nums):
        return np.dot(np.power(n, np.arange(0, 5, dtype=np.int64)), nums)

    def compress_state(self, state):
        # Convert to 0-728 representation
        t0 = int(''.join(map(str, list(state[:, 0]))), 3)
        t1 = int(''.join(map(str, list(state[:, 1]))), 3)
        t2 = int(''.join(map(str, list(state[:, 2]))), 3)
        t3 = int(''.join(map(str, list(state[:, 3]))), 3)
        t4 = int(''.join(map(str, list(state[:, 4]))), 3)

        # Convert state to a single number
        final_state_rep = self.base_n_to_base_10(729, [t0, t1, t2, t3, t4])

        return final_state_rep

    def get_next_move(self, game_state):
        # This will only be called if the game is not over

        # cur_state = game_state.game_board
        # cur_actions = game_state.next_player_actions
        # player_num = game_state.next_player
        cur_reward = self.game_not_over_reward

        self.agent.populate_current_afs_set(game_state)
        self.agent.update_q_value(cur_reward, game_state)
        next_afs = self.agent.get_next_afs(game_state)
        self.agent.update_previous_afs(next_afs)

        return next_afs.most_recent_action

    def end_of_episode(self):
        self.agent.previous_afs = None
        if self.track_stats and self.agent.num_games_played % self.stat_update_num == 0:
            self.update_stats()
        self.print_important_info()
        # self.agent.statistics['games'][self.agent.num_games_played-1]['unique_afterstates'] = len(self.agent.afs_db.Afs_dict)

    def update_stats(self):
        cur_game_info = {
            "game_num": self.agent.num_games_played,
            "value_updates": np.mean(np.abs(self.agent.value_updates)),
            "games_won_till_now": self.agent.num_games_won,
            "games_drawn_till_now": self.agent.num_games_drawn,
            "games_lost_till_now": self.agent.num_games_played - self.agent.num_games_won - self.agent.num_games_drawn,
            "num_unique_afs": len(self.agent.afs_db.Afs_dict),
        }

        self.statistics['games_info'][self.agent.num_games_played] = cur_game_info

        self.agent.value_updates = []

    def print_important_info(self):
        if self.agent.num_games_played % self.update_num == 0:
            print(f"Q-Learning Afterstates")
            print(f"Games played : {self.agent.num_games_played}")
            print(f"Games won : {self.agent.num_games_won}")
            print(f"Games drawn : {self.agent.num_games_drawn}")
            print(f"Number of unique Afs : {len(self.agent.afs_db.Afs_dict)}")

            # self.agent.num_games_played = 0
            # self.agent.num_games_won = 0
            # self.agent.num_games_drawn = 0

    def refresh_wins(self):
        self.agent.num_games_played = 0
        self.agent.num_games_won = 0
        self.agent.num_games_drawn = 0

        self.agent.value_updates = []

    def player_won(self, game_state):
        self.agent.update_q_value(self.won_reward, None)
        self.agent.num_games_won += 1
        self.agent.num_games_played += 1
        self.end_of_episode()

    def player_lost(self, game_state):
        self.agent.update_q_value(self.lost_reward, None)
        self.agent.num_games_played += 1
        self.end_of_episode()

    def player_drawn(self, game_state):
        self.agent.update_q_value(self.draw_reward, None)
        self.agent.num_games_drawn += 1
        self.agent.num_games_played += 1
        self.end_of_episode()

    def save_stats(self, filepath):
        with open(filepath, 'w') as outfile:
            outfile.write(json.dumps(self.statistics, indent=4))
