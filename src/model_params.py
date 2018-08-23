import os

parent_path = os.path.abspath('../')
data_path = parent_path + '/src/data/'

max_window = 2
num_user_cs = 6
num_agent_cs = 8
num_agent_ts = 26

all_clusters = [0, 1, 'all']
all_model_types = ['sr', 're']
all_participants = ['agent', 'user']
all_feature_types = ['cs_only', 'cs + rapport', 'cs + rapport + ts']
all_window_types = [1, 2, 'linear_combination']
all_hidden_dim = [8, 16]

cs_types = {}
pass
cs_types[all_participants[0]] = ['ASN', 'ACK', 'SD', 'QESD', 'PR', 'HE', 'VSN', 'NONE']
cs_types[all_participants[1]] = ['SD', 'QESD', 'PR', 'HE', 'VSN', 'NONE']

intention_types = {}
pass
intention_types[all_participants[0]] = ['ack()', 'request(met_before)', 'take_selfie()',
                                        'give_feedback()', 'tired()', 'request(selfie)',
                                        'request(send_msg_tlink)', 'request(another_reco)', 'greeting()',
                                        'request(first_time)', 'no_worries()', 'do()',
                                        'request(interest)', 'thank()', 'bye()', 'request(goal)',
                                        'send_msg()', 'request(feedback)', 'introduce()',
                                        'glad()', 'sorry()', 'request(anything_else)',
                                        'request(primary_goal)', 'other()', 'inform(info)', 'you()']

# Best model feature types
feature_type = {}
pass
feature_type[all_model_types[0]] = [all_feature_types[1], all_feature_types[2], all_feature_types[1]]
feature_type[all_model_types[1]] = [all_feature_types[1], all_feature_types[1], all_feature_types[1]]
feature_type[all_participants[0]] = [all_feature_types[2]]

# Best model window types
window_type = {}
pass
window_type[all_model_types[0]] = [all_window_types[2], all_window_types[2], all_window_types[0]]
window_type[all_model_types[1]] = [all_window_types[0], all_window_types[2], all_window_types[2]]
window_type[all_participants[0]] = [all_window_types[0]]

# Model file names
model_fname = {}
pass
model_fname[all_model_types[0]] = []
for cluster_id in all_clusters:
    model_fname[all_model_types[0]].append('weights_sr_user_' + str(cluster_id) + '.t7')
model_fname[all_model_types[1]] = ['weights_re_0.t7', 'weights_re_1.t7', 'weights_re_all.t7']
model_fname[all_participants[0]] = ['weights_sr_agent.t7']

# Best hidden sizes
hidden_dim = {}
pass
hidden_dim[all_model_types[0]] = [all_hidden_dim[1], all_hidden_dim[1], all_hidden_dim[0]]
hidden_dim[all_model_types[1]] = [all_hidden_dim[1], all_hidden_dim[1], all_hidden_dim[1]]
hidden_dim[all_participants[0]] = [all_hidden_dim[1]]

# Best hidden sizes
leaky_slope = {}
pass
leaky_slope[all_model_types[0]] = [0.25, 0.25, 0.05]
leaky_slope[all_model_types[1]] = [0.05, 0.05, 0.20]
leaky_slope[all_participants[0]] = [0.20]

# Best hidden sizes
thresh = {}
pass
thresh[all_model_types[0]] = [0.40, 0.35, 0.40]
thresh[all_model_types[1]] = [0.40, 0.40, 0.40]
thresh[all_participants[0]] = [0.35]


def get_input_size(feature_type, window_type):
    if feature_type == all_feature_types[0]:
        input_size = num_user_cs + num_agent_cs
    elif feature_type == all_feature_types[1]:
        input_size = num_user_cs + num_agent_cs + 1
    else:
        input_size = num_user_cs + num_agent_cs + num_agent_ts + 1

    if window_type == all_window_types[1]:
        input_size *= 2

    return input_size


def get_output_size(model_type, sr_type):
    if model_type == all_model_types[1]:
        output_size = 1
    else:
        if sr_type == all_participants[1]:
            output_size = num_user_cs
        else:
            output_size = num_agent_cs
    return output_size
