from sequence_classifier import classifiers
import dict_reader

num_dialogs = 500
all_together = False
max_iter = 50

data_path = "/Users/alankar/Documents/cmu/code/social_user_simulator/src/data/"
# All slots
slot_set_path = data_path + "slot_set"
slot_set = dict_reader.text_to_dict(slot_set_path)

# All phases
phase_set_path = data_path + "phase_set"
phase_set = dict_reader.text_to_dict(phase_set_path)

# List of system acts
sys_act_set_path = data_path + "sys_act_set"
sys_act_set = dict_reader.text_to_dict(sys_act_set_path)

# List of user acts
user_act_set_path = data_path + "user_act_set"
user_act_set = dict_reader.text_to_dict(user_act_set_path)

# Dialog rewards
reward = {}
pass
reward['send_msg_tlink'] = 5

# Count tracking
count_slots = ['session', 'person', 'food']

# Boolean slots
bool_slots = {'primary_goal': 0, 'goal_session': 1, 'goal_person': 2, 'goal_food': 3,
              'feedback': 4, 'send_msg_tlink': 5, 'another_reco': 6}

rapp_max = 6
rapp_min = 2

# User Goal slots
user_goal_slots = {}
pass
user_goal_slots['interest'] = "X"  # Just a placeholder
user_goal_slots['goal_food'] = [True, False]  # Possible values of this slot
user_goal_slots['goal_person'] = [True, False]
user_goal_slots['goal_session'] = [True, False]

# User Type slots
user_type_slots = {}
pass
user_type_slots['first_time'] = [True, False]
user_type_slots['met_before'] = [True, False]
user_type_slots['num_reco'] = [1, 2, 3, 4, 5, 6]
user_type_slots['num_reco_person'] = [0, 1, 2, 3, 4, 5, 6]
user_type_slots['num_reco_session'] = [0, 1, 2, 3, 4, 5, 6]
user_type_slots['primary_goal'] = ['goal_session', 'goal_person', 'goal_food']

# Less Time (LT): 1-2 recos, Enough Time (ET): 3-4 recos, More Time (MT): 5-6 recos
user_type_slots['time'] = ['LT', 'ET', 'MT']
# Whether or not the user cares about building rapport
user_type_slots['rapport_care'] = [True, False]

# Probability dict (for random user type initialization)
prob_user_type = {}
pass
prob_user_type['first_time'] = 0.5
prob_user_type['met_before'] = 0.0
prob_user_type['rapport_care'] = 0.35

# Probability distribution for total number of recos (1 to 6)
prob_user_type['num_reco'] = [[0.13, 0.17, 0.26, 0.29, 0.11, 0.04], [0.22, 0.22, 0.17, 0.30, 0.09, 0], [0.18, 0.21, 0.28, 0.17, 0.12, 0.03]]

# Probability distribution for total number of person recos (0 to num_reco)
prob_user_type['num_reco_1'] = [[0.71, 0.29], [0.2, 0.8], [0.5, 0.5]]
prob_user_type['num_reco_2'] = [[0.22, 0.22, 0.56], [0, 0, 1], [0.14, 0.14, 0.72]]
prob_user_type['num_reco_3'] = [[0.13, 0.29, 0.29, 0.29], [0.25, 0.5, 0.25, 0], [0.17, 0.33, 0.28, 0.22]]
prob_user_type['num_reco_4'] = [[0, 0, 0.5, 0.25, 0.25], [0, 0.14, 0.57, 0.29, 0], [0, 0.09, 0.55, 0.27, 0.09]]
prob_user_type['num_reco_5'] = [[0, 0, 0.2, 0.6, 0.2, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0.125, 0.75, 0.125, 0]]
prob_user_type['num_reco_6'] = [[0, 0, 0, 0, 0, 1, 0], [], [0, 0, 0, 0, 0, 1, 0]]

prob_user_type['acceptance_1'] = [0.64, 0.6, 0.63]
prob_user_type['acceptance_2'] = [0.72, 0.56, 0.67]
prob_user_type['acceptance_3'] = [0.78, 0.36, 0.65]
prob_user_type['acceptance_4'] = [0.57, 0.8, 0.66]
prob_user_type['acceptance_5'] = [0.56, 0.8, 0.65]

# Decision points
decision_points = ['feedback', 'send_msg_tlink', 'another_reco']

# Probability functions for decision points
prob_funcs = {}
pass
prob_funcs['send_msg_tlink'] = classifiers.get_prob_accept_msg
prob_funcs['another_reco'] = classifiers.get_prob_another_reco

# Penalty per turn
small_penalty = -0.25
large_penalty = -1
