import random
import numpy as np
import copy
import time
import simplejson as json
import game_implementation


class MCTSGameState(game_implementation.GameState):
    def __init__(self, num_rows=6, num_cols=5):
        game_implementation.GameState.__init__(self, num_rows, num_cols)
        self.child_states = {}
        self.parent_state = None
        # self.is_leaf = True

        # This information is used to implement UCT
        self.games_won = 0
        self.games_played = 0

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

    def copy(self):
        copy_node = MCTSGameState()

        copy_node.game_board = np.copy(self.game_board)
        copy_node.first_empty_slot = np.copy(self.first_empty_slot)
        copy_node.game_over = np.copy(self.game_over)
        copy_node.player_won = self.player_won.copy()
        copy_node.next_player = self.next_player
        copy_node.next_player_actions = np.copy(self.next_player_actions)

        return copy_node


# NOT USED!!
class GameStateDB:
    def __init__(self):
        self.game_state_dict = {}
        self.total_requests = 0
        self.successful_requests = 0

    def compress_state(self, state):
        return np.sum(np.multiply(np.power(3, np.arange(0, np.prod(state.shape), dtype=np.int64).reshape(state.shape)), state))

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
    def __init__(self, uct_c_coeff, num_rows, num_cols):
        self.game_root = MCTSGameState(num_rows=num_rows, num_cols=num_cols)
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

    # def get_copy_node(self, node):
    #     copy_node = GameState()

    #     copy_node.game_board = np.copy(node.game_board)
    #     copy_node.first_empty_slot = np.copy(node.first_empty_slot)
    #     copy_node.game_over = np.copy(node.game_over)
    #     copy_node.player_won = node.player_won.copy()
    #     copy_node.next_player = node.next_player
    #     copy_node.next_player_actions = np.copy(node.next_player_actions)

    #     return copy_node

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
        # print(f"Next player actions={node.next_player_actions}")

        for action in actions:
            child_node = node.copy()
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
        copy_node = node.copy()
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

    def __init__(self, num_playouts, uct_c_coeff, num_rows=6, num_cols=5):
        self.num_playouts = num_playouts
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.game_tree = GameTree(uct_c_coeff, num_rows, num_cols)
        self.current_game_player_num = 1
        self.uct_c_coeff = uct_c_coeff
        self.game_state_db = GameStateDB()

        self.num_games_played = 0
        self.num_games_won = 0
        self.num_games_drawn = 0
        self.statistics = {}

    def refresh_game_tree(self):
        self.game_tree = GameTree(self.uct_c_coeff, num_rows=self.num_rows, num_cols=self.num_cols)
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
        # We set the current game state
        for action, child in self.game_tree.current_game_state.child_states.items():
            # print(f"Action={action}\nState=")
            # child.print_state()
            if np.all(child.game_board == game_state.game_board):
                self.game_tree.current_game_state = child
                # print(f"Found child state")
                break

        # At this point, we should have current_state with opponent last move

        self.perform_mcts_steps()

        # Criteria for choosing next action - Choose most ROBUST child (i.e. most explored child - num_games_played)
        # all_children_num_games_played = np.array([c.games_played for c in self.game_tree.current_game_state.child_states.values()])
        # Criteria for choosing next action - Choose BEST child (i.e. child wth best uct score)
        all_children_uct_scores = np.array([c.get_uct_score(0) for c in self.game_tree.current_game_state.child_states.values()])
        all_children_actions = np.array([a for a in self.game_tree.current_game_state.child_states.keys()])
        # print(f"Actions allowed={game_state.next_player_actions}")
        # print(f"All children num_games_played={all_children_num_games_played}\nActions={all_children_actions}")

        chosen_ind = np.random.choice(np.flatnonzero(all_children_uct_scores == all_children_uct_scores.max()))
        action_to_take = all_children_actions[chosen_ind]

        # best_uct_score = 0
        # action_to_take = None

        # # print(f"Num of child states={len(self.game_tree.current_game_state.child_states)}")
        # for action, child in self.game_tree.current_game_state.child_states.items():
        #     # print(f"Action={action} Child uct ={child.get_uct_score(self.uct_c_coeff)}\n")
        #     # child.print_state()

        #     if child.get_uct_score(0) >= best_uct_score:
        #         # print(f"Changing best uct score to {child.get_uct_score(self.uct_c_coeff)}. Action={action}")
        #         # child.print_state()
        #         best_uct_score = child.get_uct_score(0)
        #         action_to_take = action

        # We will be taking action_to_take. Update GameTree's current state as per action taken
        self.game_tree.current_game_state = self.game_tree.current_game_state.child_states[action_to_take]
        # print(f"Changed current_game_state to")
        # self.game_tree.current_game_state.print_state()

        # # action_to_take = random.choice(cur_actions)

        return action_to_take


class MCTS_Interface:

    def __init__(self, agent, update_num=100):
        self.agent = agent
        self.update_num = update_num
        self.info_dict = {}
        self.statistics = {
            "uct_c": self.agent.uct_c_coeff,
            "num_playouts": self.agent.num_playouts,
            "games_info": {}
        }
        self.stat_update_num = 1
        self.track_stats = False

    def get_next_move(self, game_state):
        # This will only be called if the game is not over
        self.agent.current_game_player_num = game_state.next_player
        next_move = self.agent.get_next_action(game_state)

        return next_move

    def new_episode(self):
        self.agent.refresh_game_tree()
        self.print_important_info()
        if self.track_stats and self.agent.num_games_played % self.stat_update_num == 0:
            self.update_stats()

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
            "c": self.agent.uct_c_coeff,
            "wins": self.agent.num_games_won,
            "draws": self.agent.num_games_drawn,
            "total_games": self.agent.num_games_played,
            "player_num": self.agent.current_game_player_num
        }

        self.info_dict[self.agent.uct_c_coeff] = cur_info_dict

    def save_question_1_info(self, filepath):

        with open(filepath, 'w') as outfile:
            outfile.write(json.dumps(self.info_dict, indent=4))

    def update_stats(self):
        cur_game_info = {
            "game_num": self.agent.num_games_played,
            "games_won_till_now": self.agent.num_games_won,
            "games_drawn_till_now": self.agent.num_games_drawn,
            "games_lost_till_now": self.agent.num_games_played - self.agent.num_games_won - self.agent.num_games_drawn,
        }

        self.statistics['games_info'][self.agent.num_games_played] = cur_game_info

    def save_stats(self, filepath):
        with open(filepath, 'w') as outfile:
            outfile.write(json.dumps(self.statistics, indent=4))


# MCTS Agent
# mcts_agent = MCTS_Agent(60)
# mcts_int = MCTS_Interface(mcts_agent)

# temp_game = GameState()

# while not temp_game.game_over:
#     act = mcts_int.get_next_move(copy.deepcopy(temp_game))
#     print(act)
#     temp_game.next_move(act)
#     temp_game.next_move(random.choice(temp_game.next_player_actions))
