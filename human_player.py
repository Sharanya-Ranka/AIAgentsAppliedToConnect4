import random


class Human_Agent:
    """
    Represents the agent. Will make random decisions (Uniformly Random)
    """

    def __init__(self):

        self.num_games_played = 0
        self.num_games_won = 0
        self.num_games_drawn = 0
        self.updation_statistics = {}

    def validate_input(self, u_input, available_actions):
        if u_input.isnumeric() and int(u_input) in available_actions:
            return True
        else:
            print(f"Input is not correct. Please try again")

    def get_next_action(self, cur_state, cur_actions):
        input_validated = False
        print(f"Game board is")

        for i in reversed(range(0, cur_state.shape[0])):
            print(cur_state[i])

        print(f"Please enter the column you want to put the coin in")
        print(f"Options are {cur_actions}")

        while not input_validated:
            action_input = input().strip()
            input_validated = self.validate_input(action_input, cur_actions)

        return int(action_input)

    def game_over(self, cur_state, status):
        print(f"Game board is")

        for i in reversed(range(0, cur_state.shape[0])):
            print(cur_state[i])

        print(f"Game Over... You {status}")


class Human_Interface:

    def __init__(self, agent):
        self.agent = agent

        pass

    def get_next_move(self, game_state):
        # This will only be called if the game is not over
        cur_state = game_state.game_board
        cur_actions = game_state.next_player_actions

        next_move = self.agent.get_next_action(cur_state, cur_actions)

        return next_move

    def new_episode(self):
        pass

    def player_won(self, game_state):
        self.agent.num_games_won += 1
        self.agent.num_games_played += 1
        self.new_episode()
        self.agent.game_over(game_state.game_board, "WON")

    def player_lost(self, game_state):
        self.agent.num_games_played += 1
        self.new_episode()
        self.agent.game_over(game_state.game_board, "LOST")

    def player_drawn(self, game_state):
        self.agent.num_games_drawn += 1
        self.agent.num_games_played += 1
        self.new_episode()
        self.agent.game_over(game_state.game_board, "DREW")
