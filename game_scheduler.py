import simplejson as json
import time
import game_initiator as gi
import numpy as np

with open("config_template.json", 'r') as template_file:
    template = json.load(template_file)

game_info = template['games_info']
mcts_player = template['mcts_player']
q_player = template['q_learning_afterstates_player']


def set_config(game_info, player1, player2):
    config_dict = {
        "games_info": game_info,
        "player1": player1,
        "player2": player2
    }

    with open("game_config.json", 'w') as outfile:
        outfile.write(json.dumps(config_dict, indent=4))

    time.sleep(1)


# First experiment is to run MC40 against MC200 for different values of uct_c
game_info['num_rows'] = 6
game_info['num_cols'] = 5
game_info['num_games'] = 50

mc40 = mcts_player.copy()
mc200 = mcts_player.copy()

mc40['num_playouts'] = 40
mc40['update_num'] = 50

mc200['num_playouts'] = 200
mc200['update_num'] = 50
mc200['stat_update_num'] = 10

for uct_c in np.arange(0, 1.6, 0.1):

    mc40['uct_c'] = uct_c
    mc200['uct_c'] = uct_c

    for pl_num in range(2):
        mc200['stats_file'] = "MC40_vs_MC200/mc_stats_best_uct" + str(uct_c) + "pl_" + str(pl_num) + ".json"
        if pl_num == 0:
            set_config(game_info, mc200, mc40)
        else:
            set_config(game_info, mc40, mc200)

        gi.initiate_games()

# def clean_filenames(pl_info):
#     if "stats_file" in pl_info:
#         pl_info['stats_file'] = ""

#     if "save_file" in pl_info:
#         pl_info['save_file'] = ""

#     if "load_file" in pl_info:
#         pl_info['load_file'] = ""


# # Second experiment is to train Q-Learning agent against MCn (0 <= n <= 25)
# game_info['num_rows'] = 3
# game_info['num_cols'] = 5


# qlearner = q_player.copy()
# mc = mcts_player.copy()

# mc['num_playouts'] = 25
# mc['update_num'] = 1000
# mc['uct_c'] = 1.5

# qlearner['update_num'] = 1000
# qlearner['stat_update_num'] = 100

# epsilon = 0.1
# alpha = 0.2

# # [0, 5, 10, 15, 20, 25]

# # Training
# game_info['num_games'] = 20000
# qlearner['epsilon'] = epsilon
# qlearner['alpha'] = alpha
# qlearner['stats_file'] = "QLearning_vs_MCn/" + str(game_info['num_rows']) + "/train_final" + ".json"
# qlearner['save_file'] = "QLearning_vs_MCn/" + str(game_info['num_rows']) + "/weights_final" + ".dat"
# set_config(game_info, mc, qlearner)
# clean_filenames(qlearner)
# gi.initiate_games()

# # Testing EPSILON NONZERO
# game_info['num_games'] = 100
# # qlearner['load_file'] = "QLearning_vs_MCn/" + str(game_info['num_rows']) + "/weights_alpha_" + str(alpha_ch) + ".dat"
# # qlearner['stats_file'] = "QLearning_vs_MCn/" + str(game_info['num_rows']) + "/test_alpha_" + str(alpha_ch) + "ep_nonzero.json"
# # set_config(game_info, mc, qlearner)
# # clean_filenames(qlearner)
# # gi.initiate_games()

# for mc_n in [10, 25, 50, 100, 200]:
#     new_mc = mcts_player.copy()
#     new_mc['num_playouts'] = mc_n
#     # Testing EPSILON ZERO
#     qlearner['load_file'] = "QLearning_vs_MCn/" + str(game_info['num_rows']) + "/weights_final" + ".dat"
#     qlearner['stats_file'] = "QLearning_vs_MCn/" + str(game_info['num_rows']) + "/test_final_mc" + str(mc_n) + "ep_zero.json"
#     qlearner['epsilon'] = 0
#     set_config(game_info, new_mc, qlearner)
#     clean_filenames(qlearner)
#     gi.initiate_games()

# alpha = qlearner['alpha']

# # # [0, 5, 10, 15, 20, 25]

# for epsilon_ch in np.arange(0.1, 0.3, 0.1):
#     # Training
#     game_info['num_games'] = 10000
#     qlearner['epsilon'] = epsilon_ch
#     qlearner['alpha'] = alpha
#     qlearner['stats_file'] = "QLearning_vs_MCn/" + str(game_info['num_rows']) + "/train_epsilon_" + str(epsilon_ch) + ".json"
#     qlearner['save_file'] = "QLearning_vs_MCn/" + str(game_info['num_rows']) + "/weights_epsilon_" + str(epsilon_ch) + ".dat"
#     set_config(game_info, mc, qlearner)
#     clean_filenames(qlearner)
#     gi.initiate_games()

#     # Testing EPSILON NONZERO
#     game_info['num_games'] = 100
#     # qlearner['load_file'] = "QLearning_vs_MCn/" + str(game_info['num_rows']) + "/weights_epsilon_" + str(epsilon_ch) + ".dat"
#     # qlearner['stats_file'] = "QLearning_vs_MCn/" + str(game_info['num_rows']) + "/test_epsilon_" + str(epsilon_ch) + "ep_nonzero.json"
#     # set_config(game_info, mc, qlearner)
#     # clean_filenames(qlearner)
#     # gi.initiate_games()

#     # Testing EPSILON ZERO
#     qlearner['load_file'] = "QLearning_vs_MCn/" + str(game_info['num_rows']) + "/weights_epsilon_" + str(epsilon_ch) + ".dat"
#     qlearner['stats_file'] = "QLearning_vs_MCn/" + str(game_info['num_rows']) + "/test_epsilon_" + str(epsilon_ch) + "ep_zero.json"
#     qlearner['epsilon'] = 0
#     set_config(game_info, mc, qlearner)
#     clean_filenames(qlearner)
#     gi.initiate_games()


# Experiment three train Q-Learning against Q-Learning, then try using them against MCn
# game_info['num_rows'] = 3
# game_info['num_cols'] = 5

# qlearner1 = q_player.copy()
# qlearner2 = q_player.copy()

# qlearner1['update_num'] = 1000
# qlearner1['stat_update_num'] = 500

# qlearner2['update_num'] = 1000
# qlearner2['stat_update_num'] = 500

# for num_epochs in range(10):
#     # Training
#     game_info['num_games'] = 50000
#     qlearner1['epsilon'] = 0.1
#     qlearner1['alpha'] = 0.2

#     qlearner2['epsilon'] = 0.1
#     qlearner2['alpha'] = 0.2
#     qlearner2['stat_update_num'] = 500

#     if num_epochs != 0:
#         qlearner1['load_file'] = "QLearning_vs_QLearning/" + str(game_info['num_rows']) + "/pl1_weights_epoch_" + str(num_epochs - 1) + ".dat"
#         qlearner2['load_file'] = "QLearning_vs_QLearning/" + str(game_info['num_rows']) + "/pl2_weights_epoch_" + str(num_epochs - 1) + ".dat"

#     qlearner2['stats_file'] = "QLearning_vs_QLearning/" + str(game_info['num_rows']) + "/pl2_train_epoch_" + str(num_epochs) + ".json"
#     qlearner2['save_file'] = "QLearning_vs_QLearning/" + str(game_info['num_rows']) + "/pl2_weights_epoch_" + str(num_epochs) + ".dat"
#     qlearner1['save_file'] = "QLearning_vs_QLearning/" + str(game_info['num_rows']) + "/pl1_weights_epoch_" + str(num_epochs) + ".dat"

#     set_config(game_info, qlearner1, qlearner2)
#     clean_filenames(qlearner2)
#     gi.initiate_games()

#     # Testing EPSILON ZERO
#     game_info['num_games'] = 100
#     for mc_n in [10, 25, 50, 100, 200]:
#         mc = mcts_player.copy()
#         mc['num_playouts'] = mc_n

#         qlearner2['stat_update_num'] = 100
#         qlearner2['load_file'] = "QLearning_vs_QLearning/" + str(game_info['num_rows']) + "/pl2_weights_epoch_" + str(num_epochs) + ".dat"
#         qlearner2['stats_file'] = "QLearning_vs_QLearning/" + str(game_info['num_rows']) + "/pl2_test_epoch_" + str(num_epochs) + "mcn" + str(mc_n) + ".json"
#         qlearner2['epsilon'] = 0
#         set_config(game_info, mc, qlearner2)
#         clean_filenames(qlearner2)
#         gi.initiate_games()

    # qlearner['load_file'] = "QLearning_vs_MCn/" + str(game_info['num_rows']) + "/weights_epsilon_" + str(epsilon_ch) + ".dat"
    # qlearner['stats_file'] = "QLearning_vs_MCn/" + str(game_info['num_rows']) + "/test_epsilon_" + str(epsilon_ch) + "ep_nonzero.json"
    # set_config(game_info, mc, qlearner)
    # clean_filenames(qlearner)
    # gi.initiate_games()

    # Testing EPSILON ZERO
    # qlearner['load_file'] = "QLearning_vs_MCn/" + str(game_info['num_rows']) + "/weights_epsilon_" + str(epsilon_ch) + ".dat"
    # qlearner['stats_file'] = "QLearning_vs_MCn/" + str(game_info['num_rows']) + "/test_epsilon_" + str(epsilon_ch) + "ep_zero.json"
    # qlearner['epsilon'] = 0
    # set_config(game_info, mc, qlearner)
    # clean_filenames(qlearner)
    # gi.initiate_games()
