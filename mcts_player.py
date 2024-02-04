import random
import numpy as np
import copy
import time
import simplejson as json


class GameState:
    def __init__(self):
        # (6, 5) representation of the GameBoard
        self.game_board = np.zeros((6, 5), dtype=np.int64)
        # Stores the first empty rows for each column
        self.first_empty_slot = np.zeros((1, 5), dtype=np.int64)
        self.game_over = False
        self.player_won = [False, False]
        self.next_player = 1
        self.next_player_actions = np.argwhere(self.first_empty_slot < 6)[:, 1]
        self.child_states = {}
        self.parent_state = None
        # self.is_leaf = True

        # This information is used to implement UCT
        self.games_won = 0
        self.games_played = 0

    def next_move(self, drop_column):
        """
        Implement the next move. 
        Change the game_board.
        Determine if a player has won.
        Change next_player and the next_player_actions
        """
        if self.game_over or drop_column not in self.next_player_actions:
            print("SAME STATE mcts")
            self.print_state()
            print(f'Action tried={drop_column}')
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
        self.game_board[coin_coords[0], coin_coords[1]] = self.next_player
        self.first_empty_slot[0][drop_column] += 1

        return coin_coords

    def check_winner(self, coin_coords):
        """
        Checks if the game_board corresponds to a player winning.
        Sets player_won accordingly.
        """
        # print(f"Game board is\n{self.game_board}")
        # print(f"Coords = {coin_coords}")

        player_coin = self.game_board[coin_coords[0], coin_coords[1]]
        # print(f"Player coin={player_coin}")
        ones = np.array([1, 1, 1, 1], dtype=np.int32)

        horizontal = self.game_board[coin_coords[0], :]
        vertical = self.game_board[:, coin_coords[1]]
        tl_br_diag = self.game_board.diagonal(coin_coords[1] - coin_coords[0])
        tr_bl_diag = np.fliplr(self.game_board).diagonal(4 - coin_coords[1] - coin_coords[0])

        # print(f"Horizontal={horizontal} shape={horizontal.shape}\nVertical={vertical}\nTL_BR_Diag={tl_br_diag}\nTR_BL_Diag={tr_bl_diag}")

        if np.any(np.convolve(horizontal == player_coin, ones, mode="valid") == 4) or \
                np.any(np.convolve(vertical == player_coin, ones, mode="valid") == 4) or \
                np.any(np.convolve(tl_br_diag == player_coin, ones, mode="valid") == 4) or \
                np.any(np.convolve(tr_bl_diag == player_coin, ones, mode="valid") == 4):

            self.player_won[self.next_player - 1] = True
            self.game_over = True

        # player_coin = self.game_board[coin_coords]
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
            self.next_player_actions = np.argwhere(self.first_empty_slot < 6)[:, 1]
            if len(self.next_player_actions) == 0:
                self.game_over = True

    def set_child_states(self, child_states):
        self.child_states = child_states

    def get_uct_score(self, c):
        """
        Returns UCT score for the GameState. We will never request for UCT score for root (no self.parent_state)
        """
        if self.games_played == 0:
            return np.inf

        else:
            exploitation_factor = self.games_won / self.games_played
            exploration_factor = c * np.sqrt(np.log(self.parent_state.games_played) / self.games_played)

            return exploitation_factor + exploration_factor

    def print_state(self):
        print(f"Game Board is")
        for i in reversed(range(0, 6)):
            print(self.game_board[i])

        # print(f"Player won is {self.player_won}")
        # print(f"Empty slots are {self.first_empty_slot}")
        # print(f"Next player is {self.next_player}")
        # print(f"Next player actions are {self.next_player_actions}")
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


class GameStateDB:
    def __init__(self):
        self.game_state_dict = {}
        self.total_requests = 0
        self.successful_requests = 0

    def compress_state(self, state):
        return np.sum(np.multiply(np.power(3, np.arange(0, 6 * 5, dtype=np.int64).reshape(6, 5)), state))

    def get_next_state(self, state, action):
        self.total_requests += 1

        coin_coords = state.next_move_game_board(action)
        compressed_st = self.compress_state(state.game_board)

        if compressed_st in self.game_state_dict:
            self.successful_requests += 1
            return self.game_state_dict[compressed_st].copy()

        else:
            state.check_winner(coin_coords)
            state.set_next_player_and_actions()
            self.game_state_dict[compressed_st] = state.copy()

            return state


class GameTree:
    def __init__(self, uct_c_coeff):
        self.game_root = GameState()
        self.current_game_state = self.game_root
        self.uct_c_coeff = uct_c_coeff

    def selection(self, node):
        """
        This function corresponds to the selection step of the MCTS algorithm.
        Given a GameState (node), the GameTree will perform selection till it reaches a leaf (based on UCT scores)
        It returns the leaf-node and probably info on whether it is a decisive state or not
        """
        # print(f"MCTS: SELECTION")
        # print(f"Node is ")
        # node.print_state()
        cur_node = node

        # print(f"Num of children = {len(cur_node.child_states)}")

        while len(cur_node.child_states) > 0:  # (not cur_node.is_leaf) and
            # Get actions and uct_scores
            actions = list(cur_node.child_states.keys())
            uct_scores = np.array([c.get_uct_score(self.uct_c_coeff) for c in list(cur_node.child_states.values())])

            # Get max uct_score, ties are broken randomly
            ind_chosen = np.random.choice(np.flatnonzero(uct_scores == uct_scores.max()))
            cur_node = list(cur_node.child_states.values())[ind_chosen]
            # print(f"Child chosen=")
            # cur_node.print_state()

        return cur_node

    def get_copy_node(self, node):
        copy_node = GameState()

        copy_node.game_board = np.copy(node.game_board)
        copy_node.first_empty_slot = np.copy(node.first_empty_slot)
        copy_node.game_over = np.copy(node.game_over)
        copy_node.player_won = node.player_won.copy()
        copy_node.next_player = node.next_player
        copy_node.next_player_actions = np.copy(node.next_player_actions)

        return copy_node

    def expansion(self, node):
        """
        This corresponds to the Expansion step of the MCTS algorithm.
        Given a GameState, we get all child GameStates and set original GameState's children
        """
        # print(f"MCTS: EXPANSION")
        if node.game_over:
            return

        # # Leaf state given to us. It may already have children from a previous exploration, or not
        # # If it already has children, simply initialise their games_won and games_played to 0
        # if len(node.child_states) > 0:
        #     print(f"Child states already existing for ")
        #     node.print_state()
        #     node.is_leaf = False

        #     for child in node.

        actions = node.next_player_actions
        # print(f"Expansion on node")
        # node.print_state()

        for action in actions:
            child_node = self.get_copy_node(node)
            child_node.next_move(action)
            # print(f"Created child")
            # child_node.print_state()
            node.child_states[action] = child_node
            child_node.parent_state = node

        # print(f"Node has children={len(node.child_states)}")

    def simulation(self, node, player_num, game_state_db):
        """
        This function corresponds to the simulation step of the MCTS algorithm.
        For 1 simulation-
        Random moves are made until a terminal state is reached. The result is recorded.

        After all simulations are done, we perform the backpropagation step using win-lose-draw stats collected
        """
        # print(f"MCTS: SIMULATION")
        copy_node = self.get_copy_node(node)
        # print(f"Copy node is")
        # copy_node.print_state()

        while not copy_node.game_over:
            # copy_node = game_state_db.get_next_state(copy_node, random.choice(copy_node.next_player_actions))
            copy_node.next_move(random.choice(copy_node.next_player_actions))

        # print(f"Final Copy Node is")
        # copy_node.print_state()

        if copy_node.player_won[player_num - 1]:
            # print(f"WON")
            return "WON"
        elif copy_node.player_won[2 - player_num]:
            # print(f"LOST")
            return "LOST"
        else:
            # print(f"DRAWN")
            return "DRAWN"

    def backpropagation(self, node, player_num, player_outcome):
        """
        This function corresponds to the backpropagation step of the MCTS algorithm
        Given the node from which simulations are done, updates are made to the uct score components (wins and number of games played)
        If the nodes from root are P1->P2->P1->P2 and  P2 won 1 game, we will increase win count for P2 only and game count for all nodes
        """
        # print(f"MCTS: BACKPROPAGATION")
        # print(f"Player outcome={player_outcome}")

        us_player_num_update = 1 if player_outcome == "WON" else (0.5 if player_outcome == "DRAWN" else 0)
        other_player_num_update = 1 - us_player_num_update
        cur_node = node

        while cur_node is not None:
            # If the previous action was taken by us, then the current nodes should tell us how well off we are choosing them
            # We are using current action to see which update to perform
            # print(f"Cur_node is\n")
            # cur_node.print_state()

            if cur_node.next_player == player_num:
                cur_node.games_won += other_player_num_update
                cur_node.games_played += 1

            else:
                cur_node.games_won += us_player_num_update
                cur_node.games_played += 1

            # print(f"WON={cur_node.games_won} PLAYED={cur_node.games_played}")

            cur_node = cur_node.parent_state


class MCTS_Agent:
    """
    Represents the MCTS agent. This agent will perform the steps of selection, expansion, simulation and backpropagation
    and then will choose which next move to play (based on UCT score)
    The MCTS agent keeps track of the Game Tree and playout scores for each node explored.
    Selection - From the root node R (current game state), select child nodes until you encounter a leaf-node (L)
    Expansion - If leaf node is not a decisive node, create its children and choose 1 child (C) for next step. If decisive ??
    Simulation - Simulate n playouts from C till the game terminates. Here n is settable (but constant throughout gameplay) and will be an agent atribute
    Backpropagation - Using wins/draws/losses information from playouts, update nodes (C->R). (Win=+1, Draw=+0.5, Loss=0)
    """

    def __init__(self, num_playouts, uct_c_coeff):
        self.num_playouts = num_playouts
        self.game_tree = GameTree(uct_c_coeff)
        self.current_game_player_num = 1
        self.uct_c_coeff = uct_c_coeff
        self.game_state_db = GameStateDB()

        self.num_games_played = 0
        self.num_games_won = 0
        self.num_games_drawn = 0
        self.statistics = {}

    def refresh_game_tree(self):
        self.game_tree = GameTree(self.uct_c_coeff)
        # self.game_tree.current_game_state = self.game_tree.game_root

    def perform_mcts_steps(self):
        """
        Perform selection, expansion, simulation and backpropagation steps num_playouts times 
        """
        # print(f"MCTS: Performing MCTS steps with current game state")
        # self.game_tree.current_game_state.print_state()
        for sim in range(self.num_playouts):
            leaf_node = self.game_tree.selection(self.game_tree.current_game_state)
            self.game_tree.expansion(leaf_node)
            # print(f"Leaf node is")
            # leaf_node.print_state()
            new_leaf_choices = list(leaf_node.child_states.values())

            if new_leaf_choices is None or len(new_leaf_choices) == 0:
                new_leaf = leaf_node
            else:
                new_leaf = random.choice(new_leaf_choices)

            player_outcome = self.game_tree.simulation(new_leaf, self.current_game_player_num, self.game_state_db)
            self.game_tree.backpropagation(new_leaf, self.current_game_player_num, player_outcome)

        # print(f"MCTS: After Performing MCTS CHILD STATES ARE")
        # for action, child in self.game_tree.current_game_state.child_states.items():
        #     print(f"Action={action} State=\n")
        #     child.print_state()
        #     print(f"Games won={child.games_won}")
        #     print(f"Games played={child.games_played}")

    def get_next_action(self, game_state):
        """
        For the MCTS agent, this function handles getting the next action.
        It takes in the current state, and sets the game-tree's current state to it in order to perform MCTS steps
        It first performs the MCTS steps of selection, expansion, simulation and backpropagation
        Next, it chooses an action based on some condition (best UCT score?) and returns that action
        """
        # Currently, we are at the move our agent chose (or root).
        # We go 1 level down and check all children to see what the opponent chose

        # print(f"MCTS: Getting next action, game_state=\n")
        # game_state.print_state()

        # print(f"{len(self.game_tree.current_game_state.child_states)}  {np.all(self.game_tree.current_game_state.game_board == game_state.game_board)}")

        if len(self.game_tree.current_game_state.child_states) == 0 and not np.all(self.game_tree.current_game_state.game_board == game_state.game_board):
            # No children present for node game-tree, perfrom 1 expansion step and check among children
            # print(f"Expanding the node to check for opponent move")
            self.game_tree.expansion(self.game_tree.current_game_state)

        # Children are present in game_tree
        for action, child in self.game_tree.current_game_state.child_states.items():
            # print(f"Action={action}\nState=")
            # child.print_state()
            if np.all(child.game_board == game_state.game_board):
                self.game_tree.current_game_state = child
                # print(f"Found child state")
                break

        # At this point, we should have current_state with opponent last move

        self.perform_mcts_steps()

        best_uct_score = 0
        action_to_take = None

        # print(f"Num of child states={len(self.game_tree.current_game_state.child_states)}")
        for action, child in self.game_tree.current_game_state.child_states.items():
            # print(f"Action={action} Child uct ={child.get_uct_score(self.uct_c_coeff)}\n")
            # child.print_state()

            if child.get_uct_score(0) >= best_uct_score:
                # print(f"Changing best uct score to {child.get_uct_score(self.uct_c_coeff)}. Action={action}")
                # child.print_state()
                best_uct_score = child.get_uct_score(0)
                action_to_take = action

        # We will be taking action_to_take. Update GameTree's current state as per action taken
        self.game_tree.current_game_state = self.game_tree.current_game_state.child_states[action_to_take]
        # print(f"Changed current_game_state to")
        # self.game_tree.current_game_state.print_state()

        # action_to_take = random.choice(cur_actions)

        return action_to_take


class MCTS_Interface:

    def __init__(self, agent, update_num = 100):
        self.agent = agent
        self.update_num = update_num
        self.info_dict = {}

    def get_next_move(self, game_state):
        # This will only be called if the game is not over
        self.agent.current_game_player_num = game_state.next_player
        next_move = self.agent.get_next_action(game_state)

        return next_move

    def new_episode(self):

        self.agent.refresh_game_tree()
        self.print_important_info()

    def print_important_info(self):
        if self.agent.num_games_played % self.update_num == 0:
            print(f"MCTS - {self.agent.num_playouts}")
            print(f"Games played : {self.agent.num_games_played}")
            print(f"Games won : {self.agent.num_games_won}")
            print(f"Games drawn : {self.agent.num_games_drawn}")
            # print(f"States in GameStateDB = {len(self.agent.game_state_db.game_state_dict)}")
            # print(f"Total requests = {self.agent.game_state_db.total_requests} Successful = {self.agent.game_state_db.successful_requests}")

            # self.agent.num_games_played = 0
            # self.agent.num_games_won = 0
            # self.agent.num_games_drawn = 0

    def refresh_wins(self):
        self.agent.num_games_played = 0
        self.agent.num_games_won = 0
        self.agent.num_games_drawn = 0

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

    def add_question_1_info(self):
        cur_info_dict = {
            "c":self.agent.uct_c_coeff,
            "wins":self.agent.num_games_won,
            "draws":self.agent.num_games_drawn,
            "total_games":self.agent.num_games_played,
            "player_num":self.agent.current_game_player_num
        }

        self.info_dict[self.agent.uct_c_coeff] = cur_info_dict

    def save_question_1_info(self, filepath):

        with open(filepath, 'w') as outfile:
            outfile.write(json.dumps(self.info_dict, indent=4))


# MCTS Agent
# mcts_agent = MCTS_Agent(60)
# mcts_int = MCTS_Interface(mcts_agent)

# temp_game = GameState()

# while not temp_game.game_over:
#     act = mcts_int.get_next_move(copy.deepcopy(temp_game))
#     print(act)
#     temp_game.next_move(act)
#     temp_game.next_move(random.choice(temp_game.next_player_actions))
