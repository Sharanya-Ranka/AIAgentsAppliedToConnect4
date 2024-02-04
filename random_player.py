import random

class Random_Agent:
    """
    Represents the agent. Will make random decisions (Uniformly Random)
    """

    def __init__(self):

        self.num_games_played = 0
        self.num_games_won = 0
        self.num_games_drawn = 0
        self.updation_statistics = {}

    def get_next_action(self, cur_state, cur_actions):

        action_to_take = random.choice(cur_actions)

        return action_to_take


class Random_Interface:

    def __init__(self, agent):
        self.agent = agent

        pass

    def get_next_move(self, game_state):
        # This will only be called if the game is not over
        cur_state = tuple(game_state.game_board.flatten())
        cur_actions = game_state.next_player_actions
        
        next_move = self.agent.get_next_action(cur_state, cur_actions)

        return next_move

    def new_episode(self):
        pass

        
    def player_won(self, game_state):
        self.agent.num_games_won += 1
        self.agent.num_games_played += 1
        self.new_episode()

    def player_lost(self, game_state):
        self.agent.num_games_played += 1
        self.new_episode()

    def player_drawn(self, game_state):
        self.agent.num_games_drawn += 1
        self.agent.num_games_played += 1
        self.new_episode()






