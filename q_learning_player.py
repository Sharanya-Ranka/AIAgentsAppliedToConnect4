import numpy as np
import random
import pandas as pd
import re

class StateAction:
    """
    Representation of a state-action pair
    Current state, Action taken in current state
    Will also contain value estimate of the state-action pair
    """
    def __init__(self, state, action, initial_q_value):
        self.state = state
        self.action = action
        self.q_value = initial_q_value

    def __eq__(self, other):
        return other and self.state == other.state and self.action == other.action

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.state, self.action))

    def __str__(self):
        return f"State={self.state}\nAction={self.action}\nQvalue={self.q_value}"

    def __rep__(self):
        return f"State={self.state}\nAction={self.action}\nQvalue={self.q_value}"





class StateActionDatabase:
    """
    Contains a collection of StateAction objects in a set.
    Optionally should be able to load StateAction values from a file (TODO)
    If agent requests a stateaction and its not there, initialise a new one and return that (also put it in the set)
    """

    def __init__(self, initial_q_value):
        self.SA_dict = {}
        self.SA_num = len(self.SA_dict)
        self.initial_q_value = initial_q_value

    def get_SA(self, state, action):
        if tuple([state, action]) in self.SA_dict:
            return self.SA_dict[tuple([state, action])]

        else:
            temp = StateAction(state, action, self.initial_q_value)
            self.SA_dict[tuple([state, action])] = temp
            return temp

    def save_SAs_to_file(self, filepath):
        dict_to_store = {"state":[], "action":[], "qval":[]}
        for sa, sa_obj in self.SA_dict.items():
            dict_to_store["state"].append(sa[0])
            dict_to_store["action"].append(sa[1])
            dict_to_store["qval"].append(sa_obj.q_value)

        df = pd.DataFrame.from_dict(dict_to_store).astype(np.int64)
        df.to_csv(filepath, index=False)

    def load_SAs_from_file(self, filepath):
        df = pd.read_csv(filepath)
        for index, row in df.iterrows():
            state = row['state']
            action = row['action']
            sa = StateAction(state, action, row['qval'])
            # print(sa)

            self.SA_dict[tuple([state, action])] = sa








class ConstantEpsilonGreedyPolicy:
    """
    Implements the constant epsilon-greedy policy function
    Takes in a set of StateAction q_values and returns the action to take
    """
    def __init__(self, epsilon):
        self.epsilon = epsilon


    def get_policy_sa(self, sa_set):
        qvals = np.array([sa.q_value for sa in sa_set])
        if random.random() < self.epsilon:
            action_ind = random.choice(range(len(sa_set)))
            return sa_set[action_ind]
        else:
            action_ind = random.choice(np.where(qvals == np.amax(qvals))[0])
            return sa_set[action_ind]



class GreedyPolicy:
    """
    Implements the greedy policy function (required as a target policy for q-learning)
    Takes in a set of StateAction q_values and returns the action to take
    """
    def __init__(self):
        pass


    def get_policy_sa(self, sa_set):
        qvals = np.array([sa.q_value for sa in sa_set])
        
        action_ind = random.choice(np.where(qvals == np.amax(qvals))[0])
        return sa_set[action_ind]


class Q_Learning_Agent:
    """
    Represents the agent. Eventually the agent must make "intelligent" decisions
    The agent will play many games and form its estimates for "goodness" of state-action pairs (q_value)
    The agent will use a 'database' of state-action pairs
    """

    def __init__(self, alpha, gamma, behaviour_policy, target_policy):
        self.alpha = alpha
        self.gamma = gamma
        self.sa_db = StateActionDatabase(initial_q_value = 0)
        self.behaviour_policy = behaviour_policy
        self.target_policy = target_policy

        self.previous_sa = None
        self.current_state = None
        self.current_actions = None

        self.num_games_played = 0
        self.num_games_won = 0
        self.num_games_drawn = 0
        self.updation_statistics = {}

    def update_q_value(self, cur_reward, cur_state, cur_actions):
        if self.previous_sa is not None:
            # Handling wins, losses, draws (cur_state = None)
            if cur_state is not None:
                current_sa_set = [self.sa_db.get_SA(cur_state, action) for action in cur_actions]
                gamma_term = self.gamma * self.target_policy.get_policy_sa(current_sa_set).q_value
            else:
                gamma_term = 0

            alpha_term = self.alpha * (cur_reward + gamma_term - self.previous_sa.q_value)
            self.previous_sa.q_value += alpha_term

    def get_next_sa(self, cur_state, cur_actions):

        current_sa_set = [self.sa_db.get_SA(cur_state, action) for action in cur_actions]
        sa_to_take = self.behaviour_policy.get_policy_sa(current_sa_set)

        return sa_to_take

    def update_previous_sa(self, next_sa):
        self.previous_sa = next_sa 


class Q_Learning_Interface:

    def __init__(self, agent, rewards):
        self.agent = agent
        self.game_not_over_reward = rewards['game_not_over']
        self.won_reward = rewards['won']
        self.lost_reward = rewards['lost']
        self.draw_reward = rewards['draw']

        pass

    def base_n_to_base_10(self, n, nums):
        return np.dot(np.power(n, np.arange(0, 5, dtype=np.int64)), nums)

    def compress_state(self, state):
        return np.sum(np.multiply(np.power(3, np.arange(0, 6*5, dtype=np.int64).reshape(6, 5)), state))
        # # Convert to 0-728 representation
        # t0 = self.base_n_to_base_10(3, state[:, 0])
        # t1 = self.base_n_to_base_10(3, state[:, 1])
        # t2 = self.base_n_to_base_10(3, state[:, 2])
        # t3 = self.base_n_to_base_10(3, state[:, 3])
        # t4 = self.base_n_to_base_10(3, state[:, 4])
        # t0 = int(''.join(map(str, list(state[:,0]))), 3)
        # t1 = int(''.join(map(str, list(state[:,1]))), 3)
        # t2 = int(''.join(map(str, list(state[:,2]))), 3)
        # t3 = int(''.join(map(str, list(state[:,3]))), 3)
        # t4 = int(''.join(map(str, list(state[:,4]))), 3)

        # # Convert state to a single number
        # final_state_rep = self.base_n_to_base_10(729, [t0, t1, t2, t3, t4])

        # return final_state_rep

    def get_next_move(self, game_state):
        # This will only be called if the game is not over

        cur_state = self.compress_state(game_state.game_board)
        cur_actions = game_state.next_player_actions
        cur_reward = self.game_not_over_reward

        self.agent.update_q_value(cur_reward, cur_state, cur_actions)
        next_sa = self.agent.get_next_sa(cur_state, cur_actions)
        self.agent.update_previous_sa(next_sa)

        return next_sa.action

    def end_of_episode(self):
        self.agent.previous_sa = None
        self.print_important_info()

    def print_important_info(self):
        print(f"Games played : {self.agent.num_games_played}")
        print(f"Games won : {self.agent.num_games_won}")
        print(f"Number of unique SA : {len(self.agent.sa_db.SA_dict)}")

        
    def player_won(self, game_state):
        self.agent.update_q_value(self.won_reward, None, None)
        self.agent.num_games_won += 1
        self.agent.num_games_played += 1
        self.end_of_episode()

    def player_lost(self, game_state):
        self.agent.update_q_value(self.lost_reward, None, None)
        self.agent.num_games_played += 1
        self.end_of_episode()

    def player_drawn(self, game_state):
        self.agent.update_q_value(self.draw_reward, None, None)
        self.agent.num_games_drawn += 1
        self.agent.num_games_played += 1
        self.end_of_episode()






