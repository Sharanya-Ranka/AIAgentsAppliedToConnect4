import numpy as np
import copy
import time


class GameState:
    def __init__(self, num_rows=6, num_cols=5):
        # (6, 5) representation of the GameBoard
        self.game_board = np.zeros((num_rows, num_cols), dtype=np.int8)
        # Stores the first empty rows for each column
        self.first_empty_slot = np.zeros((1, num_cols), dtype=np.int8)
        self.game_over = False
        self.player_won = [False, False]
        self.next_player = 1
        self.next_player_actions = np.argwhere(self.first_empty_slot < num_rows)[:, 1]

    def next_move(self, drop_column):
        """
        Implement the next move. 
        Change the game_board.
        Determine if a player has won.
        Change next_player and the next_player_actions
        """
        if self.game_over or drop_column not in self.next_player_actions:
            print("SAME STATE")
            return

        coin_coords = self.next_move_game_board(drop_column)
        # print(f"Coin coords are {coin_coords}")
        self.check_winner(coin_coords)
        self.set_next_player_and_actions()
        # self.print_state()

    def next_move_game_board(self, drop_column):
        """
        Change the game_board to reflect the state after move
        """
        # print(f"Drop column is {drop_column}")
        coin_coords = np.array([self.first_empty_slot[0][drop_column], drop_column])
        self.game_board[tuple(coin_coords)] = self.next_player
        self.first_empty_slot[0][drop_column] += 1

        return coin_coords

    def check_winner(self, coin_coords):
        """
        Checks if the game_board corresponds to a player winning.
        Sets player_won accordingly.
        """
        # print(f"Game board is\n{self.game_board}")
        # print(f"Coords = {coin_coords}")

        player_coin = self.game_board[tuple(coin_coords)]
        # print(f"Player coin={player_coin}")
        ones = np.array([1, 1, 1, 1], dtype=np.int32)

        horizontal = self.game_board[coin_coords[0], :]
        vertical = self.game_board[:, coin_coords[1]]
        tl_br_diag = self.game_board.diagonal(coin_coords[1] - coin_coords[0])
        tr_bl_diag = np.fliplr(self.game_board).diagonal(self.game_board.shape[1] - 1 - coin_coords[1] - coin_coords[0])

        # print(f"Horizontal={horizontal} shape={horizontal.shape}\nVertical={vertical}\nTL_BR_Diag={tl_br_diag}\nTR_BL_Diag={tr_bl_diag}")

        if np.any(np.convolve(horizontal == player_coin, ones, mode="valid") == 4) or \
                np.any(np.convolve(vertical == player_coin, ones, mode="valid") == 4) or \
                np.any(np.convolve(tl_br_diag == player_coin, ones, mode="valid") == 4) or \
                np.any(np.convolve(tr_bl_diag == player_coin, ones, mode="valid") == 4):

            self.player_won[self.next_player - 1] = True
            self.game_over = True

            # # runs = [bottom, top, right, left, bottom-right, top-left, top-right, bottom-left]
            # runs = [
            #     self.get_run(coin_coords, np.array([0, 1])),
            #     self.get_run(coin_coords, np.array([0, -1])),
            #     self.get_run(coin_coords, np.array([1, 0])),
            #     self.get_run(coin_coords, np.array([-1, 0])),
            #     self.get_run(coin_coords, np.array([1, 1])),
            #     self.get_run(coin_coords, np.array([-1, -1])),
            #     self.get_run(coin_coords, np.array([1, -1])),
            #     self.get_run(coin_coords, np.array([-1, 1]))
            # ]

            # if runs[0] + runs[1] >= 3 or runs[2] + runs[3] >= 3 or runs[4] + runs[5] >= 3 or runs[6] + runs[7] >= 3:
            #     self.player_won[self.next_player - 1] = True
            #     self.game_over = True

    def get_run(self, coin_coords, direction):
        # print(f"Direction={direction}")
        i = 0
        new_coords = coin_coords + direction
        coin = self.game_board[coin_coords[0], coin_coords[1]]

        # print(f"New coords={new_coords}")

        # print(self.game_board[new_coords[0], new_coords[1]])

        while np.all(new_coords >= np.array([0, 0])) and np.all(new_coords < np.array([6, 5])):
            if self.game_board[new_coords[0], new_coords[1]] != coin:
                break

            i += 1
            new_coords += direction

        return i

    def set_next_player_and_actions(self):
        """
        Sets the next player (if someone has won, sets it to 0)
        Also sets the actions available to the next player
        """

        if self.game_over:
            # No one should play next
            self.next_player = 0
            self.next_player_actions = np.zeros(0, dtype=np.int64)

        else:
            # Changes player (1->2 and 2->1)
            self.next_player = 3 - self.next_player
            self.next_player_actions = np.argwhere(self.first_empty_slot < self.game_board.shape[0])[:, 1]
            if len(self.next_player_actions) == 0:
                self.game_over = True

    def print_state(self):
        print(f"Game Board is")
        for i in reversed(range(0, self.game_board.shape[0])):
            print(self.game_board[i])

        print(f"Player won is {self.player_won}")
        print(f"Empty slots are {self.first_empty_slot}")
        print(f"Next player is {self.next_player}")
        print(f"Next player actions are {self.next_player_actions}")
        print()

    def copy(self):
        copy_node = GameState()

        copy_node.game_board = np.copy(self.game_board)
        copy_node.first_empty_slot = np.copy(self.first_empty_slot)
        copy_node.game_over = np.copy(self.game_over)
        copy_node.player_won = self.player_won.copy()
        copy_node.next_player = self.next_player
        copy_node.next_player_actions = np.copy(self.next_player_actions)

        return copy_node


class Game:
    def __init__(self, player1_int, player2_int, num_rows=6, num_cols=5):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.game_state = GameState(num_rows=num_rows, num_cols=num_cols)
        self.player1_interface = player1_int
        self.player2_interface = player2_int

    def play_game(self):
        """
        Conducts the game of connect4 between 2 player interfaces.
        Will give the player interfaces the complete game-state (copy)
        Expects a move from the player interface. Player Interface must make sure that the move is legal
        """
        while not self.game_state.game_over:

            if self.game_state.next_player == 1:
                next_move = self.player1_interface.get_next_move(copy.deepcopy(self.game_state))

            else:
                next_move = self.player2_interface.get_next_move(copy.deepcopy(self.game_state))

            self.game_state.next_move(next_move)
            # self.game_state.print_state()

        if self.game_state.player_won[0] == True:
            # print(f"Player 1 won the game!")
            self.player1_interface.player_won(copy.deepcopy(self.game_state))
            self.player2_interface.player_lost(copy.deepcopy(self.game_state))

        elif self.game_state.player_won[1] == True:
            # print(f"Player 2 won the game!")
            self.player1_interface.player_lost(copy.deepcopy(self.game_state))
            self.player2_interface.player_won(copy.deepcopy(self.game_state))

        else:
            # print(f"Its a Draw!")
            # self.game_state.print_state()
            self.player1_interface.player_drawn(copy.deepcopy(self.game_state))
            self.player2_interface.player_drawn(copy.deepcopy(self.game_state))

    def play_game_time_moves(self):
        """
        Conducts the game of connect4 between 2 player interfaces. Times each player's moves and returns the sum of times taken
        Will give the player interfaces the complete game-state (copy)
        Expects a move from the player interface. Player Interface must make sure that the move is legal
        """
        moves_times = {1: [0, 0], 2: [0, 0]}

        while not self.game_state.game_over:

            if self.game_state.next_player == 1:
                start_time = time.time()
                next_move = self.player1_interface.get_next_move(copy.deepcopy(self.game_state))
                end_time = time.time()
                moves_times[1][0] += 1
                moves_times[1][1] += end_time - start_time

            else:
                start_time = time.time()
                next_move = self.player2_interface.get_next_move(copy.deepcopy(self.game_state))
                end_time = time.time()
                moves_times[2][0] += 1
                moves_times[2][1] += end_time - start_time

            self.game_state.next_move(next_move)

            self.game_state.print_state()

        if self.game_state.player_won[0] == True:
            # print(f"Player 1 won the game!")
            self.player1_interface.player_won(copy.deepcopy(self.game_state))
            self.player2_interface.player_lost(copy.deepcopy(self.game_state))

        elif self.game_state.player_won[1] == True:
            # print(f"Player 2 won the game!")
            self.player1_interface.player_lost(copy.deepcopy(self.game_state))
            self.player2_interface.player_won(copy.deepcopy(self.game_state))

        else:
            # print(f"Its a Draw!")
            # self.game_state.print_state()
            self.player1_interface.player_drawn(copy.deepcopy(self.game_state))
            self.player2_interface.player_drawn(copy.deepcopy(self.game_state))

        return moves_times

    def play_games(self, count):
        for _ in range(count):
            self.play_game()
            self.game_state = GameState(num_rows=self.num_rows, num_cols=self.num_cols)

    def play_games_time_moves(self, count):
        results = {1: [0, 0], 2: [0, 0]}

        for _ in range(count):
            self.game_state = GameState(num_rows=self.num_rows, num_cols=self.num_cols)
            cur_result = self.play_game_time_moves()
            # Adding to cumulative time

            results[1][0] += cur_result[1][0]
            results[1][1] += cur_result[1][1]

            results[2][0] += cur_result[2][0]
            results[2][1] += cur_result[2][1]

        return results


class TestGameState:
    def __init__(self):
        self.game_state = GameState()

        # moves = [3, 3, 2, 2, 1, 4, 1, 2, 2, 0, 0, 1, 0, 1] Given Example
        # moves = [4, 3, 3, 2, 1, 2, 3, 1, 0, 1, 1, 0, 2] Diagonals
        # moves = [1, 1, 1, 1, 1 ,1, 2, 2, 2, 2, 2, 2, ] Testing restricted actions
        for move in moves:
            self.game_state.next_move(move)


# s = TestGameState()
