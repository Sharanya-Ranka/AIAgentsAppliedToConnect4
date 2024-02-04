import game_implementation
import q_learning_player
import q_learning_afterstate_player as qlap
import mcts_inherit as mcts_player
import random_player
import human_player
import numpy as np
import simplejson as json

# ## Q LEARNING STATE_ACTION AGENT
# # Initialising the Q-Learning agent and interface
# alpha = 0.2
# gamma = 1
# epsilon = 0.1
# rewards = {"game_not_over":-5, "won":100, "lost":-100, "draw":-20}


# bev_policy = q_learning_player.ConstantEpsilonGreedyPolicy(epsilon)
# tar_policy = q_learning_player.GreedyPolicy()
# q_learning_agent = q_learning_player.Q_Learning_Agent(alpha, gamma, bev_policy, tar_policy)
# # Use previous estimates
# # q_learning_agent.sa_db.load_SAs_from_file("SA_data.dat")
# q_learning_int = q_learning_player.Q_Learning_Interface(q_learning_agent, rewards)


# Q LEARNING AFTERSTATE AGENT
# Initialising the Q - Learning agent and interface
alpha = 0.2
gamma = 1
epsilon = 0.1
rewards = {"game_not_over": -5, "won": 1000, "lost": -1000, "draw": -100}


bev_policy = qlap.ConstantEpsilonGreedyPolicy(epsilon)
tar_policy = qlap.GreedyPolicy()
q_learning_agent = qlap.Q_Learning_Agent(alpha, gamma, bev_policy, tar_policy)
# Use previous estimates
q_learning_agent.afs_db.load_Afs_from_file("Q_Q_Pl1_35.dat")
q_learning_int = qlap.Q_Learning_Interface(q_learning_agent, rewards)


# Initialising the Q-Learning agent and interface
alpha1 = 0.2
gamma1 = 1
epsilon1 = 0
rewards1 = {"game_not_over": -5, "won": 1000, "lost": -1000, "draw": -100}


bev_policy1 = qlap.ConstantEpsilonGreedyPolicy(epsilon1)
tar_policy1 = qlap.GreedyPolicy()
q_learning_agent1 = qlap.Q_Learning_Agent(alpha1, gamma1, bev_policy1, tar_policy1)
# Use previous estimates
q_learning_agent1.afs_db.load_Afs_from_file("Q_Q_Pl2_35.dat")
q_learning_int1 = qlap.Q_Learning_Interface(q_learning_agent1, rewards1)


# MCTS Agent
mcts_agent1 = mcts_player.MCTS_Agent(25, 0.5, 3, 5)
mcts_int1 = mcts_player.MCTS_Interface(mcts_agent1)

# MCTS Agent
mcts_agent2 = mcts_player.MCTS_Agent(200, 0.5)
mcts_int2 = mcts_player.MCTS_Interface(mcts_agent2)


# RANDOM AGENT
# Initialising the Random agent and Interface
random_agent = random_player.Random_Agent()
random_int = random_player.Random_Interface(random_agent)


# HUMAN AGENT
# Initialise Human Agent
human_agent = human_player.Human_Agent()
human_int = human_player.Human_Interface(human_agent)


game = game_implementation.Game(mcts_int1, q_learning_int1, 3, 5)
game.play_games(500)

# Saving learnt info to file
# q_learning_agent.afs_db.save_Afs_to_file("Q_Q_Pl1_35.dat")
# q_learning_agent1.afs_db.save_Afs_to_file("Q_Q_Pl2_35.dat")

# # Answer to question 1
# mcts_int1.update_num = 50
# mcts_int2.update_num = 50

# for uct_c in np.arange(0, 2.1, 0.15):
#     mcts_agent1.uct_c_coeff = uct_c
#     mcts_agent2.uct_c_coeff = uct_c

#     game = game_implementation.Game(mcts_int1, mcts_int2)
#     game.play_games(50)
#     game = game_implementation.Game(mcts_int2, mcts_int1)
#     game.play_games(50)
#     print()

#     mcts_int1.add_question_1_info()
#     mcts_int2.add_question_1_info()

#     mcts_int1.refresh_wins()
#     mcts_int2.refresh_wins()

# mcts_int1.save_question_1_info("mcts_40_info.json")
# mcts_int2.save_question_1_info("mcts_200_info.json")

# # Move times
# agent_times = {}

# agent_times['Q-Learning'] = [0, 0]
# game = game_implementation.Game(random_int, q_learning_int)
# results = game.play_games_time_moves(100)
# agent_times['Q-Learning'][0] += results[2][0]
# agent_times['Q-Learning'][1] += results[2][1]


# for mcts_n in range(25, 401, 25):
#     agent_times[f'MCTS - {mcts_n}'] = [0, 0]
#     mcts_agent1.num_playouts = mcts_n
#     mcts_int1.update_num = 20
#     game = game_implementation.Game(random_int, mcts_int1)
#     results = game.play_games_time_moves(20)
#     agent_times[f'MCTS - {mcts_n}'][0] += results[2][0]
#     agent_times[f'MCTS - {mcts_n}'][1] += results[2][1]

# print(agent_times)

# with open('agent_times.json','w') as outfile:
#     outfile.write(json.dumps(agent_times, indent=4))


# # Answer to question 2
# game = game_implementation.Game(mcts_int1, q_learning_int)
# game.play_games(10000)
# q_learning_int.save_statistics("mcts_10_q_0p2_0p1.json")


# Code for question 3 Q-Learning vs Q-Learning approach

# def enter_wins(q_learning_int, mcts_lvl, position, trained, put_in):
#     if trained not in put_in:
#         put_in[trained] = {'first':{}, 'second':{}}

#     put_in[trained][position][mcts_lvl] = q_learning_int.agent.num_games_won

# q_q_stats = {}
# plays_per_loop = 10000
# plays_with_mcts = 20
# # FORMAT: {0:{'first':{25:wins_against_25, 50:wins_against_50, ...}, 'second':{25:wins_against_25, 50:wins_against_50, ...}}, 10000:{...}, ...}

# q_learning_int.update_num = plays_with_mcts
# q_learning_int1.update_num = plays_with_mcts
# mcts_int1.update_num = plays_with_mcts

# for mcts_sims in [20, 50, 100, 200, 400]:
#     mcts_agent1.num_playouts = mcts_sims
#     test_game = game_implementation.Game(q_learning_int, mcts_int1)
#     test_game.play_games(plays_with_mcts)
#     enter_wins(q_learning_int, mcts_sims, 'first', 0, q_q_stats)
#     q_learning_int.refresh_wins()
#     mcts_int1.refresh_wins()
#     print()
#     test_game = game_implementation.Game(mcts_int1, q_learning_int1)
#     test_game.play_games(plays_with_mcts)
#     enter_wins(q_learning_int1, mcts_sims, 'second', 0, q_q_stats)
#     q_learning_int1.refresh_wins()
#     mcts_int1.refresh_wins()
#     print()
#     print()

# print()

# for i in range(1, 11):
#     q_learning_int.update_num = 1000
#     q_learning_int1.update_num = 1000

#     game = game_implementation.Game(q_learning_int, q_learning_int1)
#     game.play_games(plays_per_loop)

#     q_learning_int.refresh_wins()
#     q_learning_int1.refresh_wins()

#     q_learning_int.update_num = plays_with_mcts
#     q_learning_int1.update_num = plays_with_mcts
#     mcts_int1.update_num = plays_with_mcts

#     print(f"\n\n\n")
#     for mcts_sims in [20, 50, 100, 200, 400]:
#         mcts_agent1.num_playouts = mcts_sims
#         test_game = game_implementation.Game(q_learning_int, mcts_int1)
#         test_game.play_games(plays_with_mcts)
#         enter_wins(q_learning_int, mcts_sims, 'first', i*plays_per_loop, q_q_stats)
#         q_learning_int.refresh_wins()
#         mcts_int1.refresh_wins()
#         print()
#         test_game = game_implementation.Game(mcts_int1, q_learning_int1)
#         test_game.play_games(plays_with_mcts)
#         enter_wins(q_learning_int1, mcts_sims, 'second', i*plays_per_loop, q_q_stats)
#         q_learning_int1.refresh_wins()
#         mcts_int1.refresh_wins()
#         print()
#         print()
#     print(f"\n\n\n")


# with open('q_vs_q_learning.json', 'w') as outfile:
#     outfile.write(json.dumps(q_q_stats, indent=4))


# # Code for question 3 Q-Learning vs increadingly strong mcts approach
# mcts_plays_dict = {2:3000, 4:3000, 6:2000, 8:2000, 10:1000, 12:1000, 14:500, 16:500, 18:500, 20:500}
# # t_mcts_plays_dict = {2:3, 4:3, 6:2, 8:2}

# stats = {'training':[], 'testing':{}}
# training_games_interval = 100
# plays_with_mcts = 20

# total_plays = 0

# for mcts_sims, plays in mcts_plays_dict.items():
#     q_learning_int.update_num = plays
#     mcts_agent1.num_playouts = mcts_sims
#     games_trained = 0
#     while games_trained < plays:
#         game = game_implementation.Game(mcts_int1, q_learning_int)
#         game.play_games(training_games_interval)

#         stats['training'].append(q_learning_int.agent.num_games_won / training_games_interval)

#         q_learning_int.refresh_wins()
#         mcts_int1.refresh_wins()
#         games_trained += training_games_interval

#     total_plays += plays
#     q_learning_int.update_num = plays_with_mcts
#     mcts_int1.update_num = plays_with_mcts

#     print(f"\n\n\n")
#     for mcts_sims in [20, 50, 100, 200, 400]:
#         mcts_agent1.num_playouts = mcts_sims

#         test_game = game_implementation.Game(mcts_int1, q_learning_int)
#         test_game.play_games(plays_with_mcts)
#         enter_wins(q_learning_int, mcts_sims, 'second', total_plays, stats['testing'])
#         q_learning_int.refresh_wins()
#         mcts_int1.refresh_wins()

#         print()
#         print()
#     print(f"\n\n\n")

# with open('mcts_inc_vs_q.json', 'w') as outfile:
#     outfile.write(json.dumps(stats, indent=4))

# Q-Learning against mcts200 (Checking for good parameters for q-learning)
# Checking good alpha value
def enter_stats(q_learning_int, alpha_or_epsilon, put_in):
    if alpha_or_epsilon not in put_in:
        put_in[alpha_or_epsilon] = {"wins": [], "afterstates": [], "updates": []}

    put_in[alpha_or_epsilon]["wins"].append(q_learning_int.agent.num_games_won)
    put_in[alpha_or_epsilon]["afterstates"].append(len(q_learning_int.agent.afs_db.Afs_dict))
    put_in[alpha_or_epsilon]["updates"].append(np.mean(np.abs(q_learning_int.agent.value_updates)))


# alpha_val_stats = {}
# plays_with_mcts = 2000
# plays_per_update = 100

# for alpha_val in [0.1, 0.2, 0.5, 0.8]:#, 0.2, 0.5,
#     # Initialising the Q-Learning agent and interface
#     print(f"Alpha is {alpha_val}")
#     alpha1 = alpha_val
#     gamma1 = 1
#     epsilon1 = 0.1
#     rewards1 = {"game_not_over": -5, "won": 1000, "lost": -1000, "draw": -100}


#     bev_policy1 = qlap.ConstantEpsilonGreedyPolicy(epsilon1)
#     tar_policy1 = qlap.GreedyPolicy()
#     q_learning_agent1 = qlap.Q_Learning_Agent(alpha1, gamma1, bev_policy1, tar_policy1)
#     # Use previous estimates
#     # q_learning_agent.afs_db.load_Afs_from_file("Afs_data.dat")
#     q_learning_int1 = qlap.Q_Learning_Interface(q_learning_agent1, rewards1)

#     q_learning_int1.update_num = plays_per_update
#     mcts_int1.update_num = plays_per_update

#     test_game = game_implementation.Game(mcts_int1, q_learning_int1)
#     plays_till_now = 0
#     while plays_till_now < plays_with_mcts:
#         test_game.play_games(plays_per_update)
#         enter_stats(q_learning_int1, alpha_val, alpha_val_stats)
#         q_learning_int1.refresh_wins()
#         mcts_int1.refresh_wins()
#         plays_till_now += plays_per_update

#     # alpha_val_stats[alpha_val]["other_info"] = q_learning_int1.agent.statistics

# with open('alpha_value_info.json', 'w') as outfile:
#     outfile.write(json.dumps(alpha_val_stats, indent=4))


# # Checking good epsilon value

# epsilon_val_stats = {}
# plays_with_mcts = 2000
# plays_per_update = 100

# for epsilon_val in [0.1, 0.2, 0.3, 0.5]:#, 0.2, 0.5,
#     # Initialising the Q-Learning agent and interface
#     alpha1 = 0.2
#     gamma1 = 1
#     epsilon1 = epsilon_val
#     rewards1 = {"game_not_over": -5, "won": 1000, "lost": -1000, "draw": -100}


#     bev_policy1 = qlap.ConstantEpsilonGreedyPolicy(epsilon1)
#     tar_policy1 = qlap.GreedyPolicy()
#     q_learning_agent1 = qlap.Q_Learning_Agent(alpha1, gamma1, bev_policy1, tar_policy1)
#     # Use previous estimates
#     # q_learning_agent.afs_db.load_Afs_from_file("Afs_data.dat")
#     q_learning_int1 = qlap.Q_Learning_Interface(q_learning_agent1, rewards1)

#     q_learning_int1.update_num = plays_per_update
#     mcts_int1.update_num = plays_per_update

#     test_game = game_implementation.Game(mcts_int1, q_learning_int1)
#     plays_till_now = 0
#     while plays_till_now < plays_with_mcts:
#         test_game.play_games(plays_per_update)
#         enter_stats(q_learning_int1, epsilon_val, epsilon_val_stats)
#         q_learning_int1.refresh_wins()
#         mcts_int1.refresh_wins()
#         plays_till_now += plays_per_update

#     # alpha_val_stats[alpha_val]["other_info"] = q_learning_int1.agent.statistics

# with open('epsilon_value_info.json', 'w') as outfile:
#     outfile.write(json.dumps(epsilon_val_stats, indent=4))

# Saving learnt info to file
# q_learning_agent.sa_db.save_SAs_to_file("SA_data.dat")
# q_learning_agent.afs_db.save_Afs_to_file("MCTS_Q.dat")
