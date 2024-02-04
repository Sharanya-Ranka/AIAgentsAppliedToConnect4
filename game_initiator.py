import game_implementation
import q_learning_player
import q_learning_afterstate_player as qlap
import mcts_inherit as mcts_player
import random_player
import human_player
import numpy as np
import simplejson as json


def q_learning_afterstate_initiator(info, *args):
    alpha = info['alpha']
    gamma = info['gamma']
    epsilon = info['epsilon']
    rewards = info['rewards']  # {"game_not_over": -5, "won": 1000, "lost": -1000, "draw": -100}

    bev_policy = qlap.ConstantEpsilonGreedyPolicy(epsilon)
    tar_policy = qlap.GreedyPolicy()
    q_learning_agent = qlap.Q_Learning_Agent(alpha, gamma, bev_policy, tar_policy)
    # Use previous estimates
    if info['load_file'] != "":
        q_learning_agent.afs_db.load_Afs_from_file(info['load_file'])

    q_learning_int = qlap.Q_Learning_Interface(q_learning_agent, rewards)
    q_learning_int.update_num = info['update_num']
    q_learning_int.stat_update_num = info['stat_update_num']
    q_learning_int.track_stats = False if info['stats_file'] == "" else True

    print(f"Initialised Q-Learning Afterstates with")
    print(f"alpha={alpha} gamma={gamma}, epsilon={epsilon}")
    print(f"rewards={rewards}")
    print(f"load from={info['load_file']}   save_to={info['save_file']}  stats={info['stats_file']}")
    print(f"update num={info['update_num']} stat_update_num={info['stat_update_num']}\n")

    return q_learning_int


def q_learning_afterstate_terminator(q_learning_int, info):
    agent = q_learning_int.agent

    if info['stats_file'] != "":
        q_learning_int.save_stats(info['stats_file'])

    if info['save_file'] != "":
        agent.afs_db.save_Afs_to_file(info['save_file'])


def mcts_terminator(mcts_int, info):

    if info['stats_file'] != "":
        mcts_int.save_stats(info['stats_file'])


def mcts_initiator(info, num_rows, num_cols):
    # MCTS Agent
    mcts_agent = mcts_player.MCTS_Agent(info['num_playouts'], info['uct_c'], num_rows, num_cols)
    mcts_int = mcts_player.MCTS_Interface(mcts_agent)
    mcts_int.update_num = info['update_num']

    mcts_int.stat_update_num = info['stat_update_num']
    mcts_int.track_stats = False if info['stats_file'] == "" else True

    print(f"Initialised MCTS with")
    print(f"num_playouts={info['num_playouts']} uct_c={info['uct_c']}")
    print(f"stats={info['stats_file']}")
    print(f"update num={info['update_num']} stat_update_num={info['stat_update_num']}\n")

    return mcts_int


def random_initiator(info):
    pass


def human_initiator(info):
    pass


initiators = {
    "q_learning_afterstates": q_learning_afterstate_initiator,
    "mcts": mcts_initiator,
    "random": random_initiator,
    "human": human_initiator
}

terminators = {
    "q_learning_afterstates": q_learning_afterstate_terminator,
    "mcts": mcts_terminator,
    "random": None,
    "human": None
}


def initiate_games():
    with open("game_config.json", 'r') as infile:
        game_config = json.load(infile)

    games_info = game_config['games_info']
    num_rows = games_info['num_rows']
    num_cols = games_info['num_cols']

    print(f"Initiating games with")
    print(f"Rows={num_rows} Columns={num_cols} Plays={games_info['num_games']}\n")

    # Calling intiator on player1
    pl1_info = game_config['player1']
    int_player1 = initiators[pl1_info['type']](pl1_info, num_rows, num_cols)

    # Calling initiator on player2
    pl2_info = game_config['player2']
    int_player2 = initiators[pl2_info['type']](pl2_info, num_rows, num_cols)

    # Playing games
    game = game_implementation.Game(int_player1, int_player2, num_rows, num_cols)
    game.play_games(games_info['num_games'])

    # Calling terminator on player1
    pl1_info = game_config['player1']
    int_player1 = terminators[pl1_info['type']](int_player1, pl1_info)

    # Calling terminator on player2
    pl2_info = game_config['player2']
    int_player2 = terminators[pl2_info['type']](int_player2, pl2_info)


if __name__ == "__main__":
    initiate_games()
