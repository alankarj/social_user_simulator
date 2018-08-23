import dialog_config
import model_params
from model import JointEstimator, CategoricalEncoder
import torch
import numpy as np
import common_func

intention_str = 'intention'


class RuleBasedAgent:
    def __init__(self):
        self.phase = None
        self.current_action = None
        self.prev_action = None
        self.max = None

        self.sr = None
        self.re = None

        self.prev_cs_agent = None
        self.prev_cs_user = None
        self.prev_ti_agent = None
        self.prev_rapp_val = None

        self.thresh = None

    def initialize(self):
        index = 0
        sr_type = model_params.all_participants[0]

        # Social Reasoner
        model_type = model_params.all_participants[0]
        self.sr = self.get_model(sr_type, model_type, index)

        # Rapport Estimator
        model_type = model_params.all_model_types[1]
        index = 2
        self.re = self.get_model(sr_type, model_type, index)

        null_list = ["NULL"]
        self.prev_cs_agent = [null_list, null_list]
        self.prev_cs_user = [null_list, null_list]
        self.prev_ti_agent = [null_list, null_list]
        self.prev_rapp_val = [0, 0]

        cs_types = model_params.cs_types
        intention_types = model_params.intention_types

        self.enc = {}
        for p in model_params.all_participants:
            self.enc[p] = CategoricalEncoder(cs_types[p])
        self.enc[model_params.all_participants[0] + '_' + intention_str] = \
            CategoricalEncoder(intention_types[model_params.all_participants[0]])

        self.max = {}
        pass
        self.max['session'] = 5
        self.max['person'] = 5

        # Current values
        self.phase = 'greetings'

        # First agent action
        agent_action = {}
        pass
        agent_action['act'] = 'greeting'
        agent_action['phase'] = self.phase
        agent_action['request_slots'] = ''
        agent_action['inform_slots'] = {}

        self.current_action = agent_action

        agent_intention = common_func.get_agent_intention(agent_action)
        agent_action['CS'] = self.generate_conv_strat_data_driven(null_list, agent_intention, print_info=False)

        return agent_action, self.prev_rapp_val[0]

    def get_model(self, sr_type, model_type, index, print_info=False):
        feature_type = model_params.feature_type[model_type][index]
        window_type = model_params.window_type[model_type][index]
        input_size = model_params.get_input_size(feature_type, window_type)
        hidden_dim = model_params.hidden_dim[model_type][index]
        leaky_slope = model_params.leaky_slope[model_type][index]
        output_size = model_params.get_output_size(model_type, sr_type)
        model_fname = model_params.model_fname[model_type][index]
        self.thresh = model_params.thresh[model_type][index]

        if print_info:
            print("Feature type: ", feature_type)
            print("Window type: ", window_type)
            print("Input size: ", input_size)
            print("Hidden dimension: ", hidden_dim)
            print("Leaky slope: ", leaky_slope)
            print("Output size: ", output_size)

        model = JointEstimator(input_size, hidden_dim, output_size, leaky_slope, window_type,
                               feature_type, model_type)
        model.load_state_dict(torch.load(dialog_config.data_path + model_fname, map_location='cpu'))
        return model

    def next(self, state):
        user_action = state['user_action']
        self.prev_action = self.current_action

        user_act = user_action['act']
        inform_slots = {}
        request_slots = ''
        agent_action = {}
        phase = self.phase
        user_inform_slots = user_action['inform_slots']

        if self.phase == 'greetings':
            if user_act == 'greeting':
                phase = 'introductions'
                act = 'introduce'
                agent_action = self.construct_action(act, phase, inform_slots,
                                                     request_slots)
        elif self.phase == 'introductions':
            if user_act == 'null':
                if state['first_time'] == '':
                    act = 'request'
                    request_slots = 'first_time'
                    agent_action = self.construct_action(act, phase,
                                                         inform_slots,
                                                         request_slots)

                else:
                    act = 'request'
                    phase = 'met_before'
                    request_slots = 'met_before'
                    agent_action = self.construct_action(act, phase,
                                                         inform_slots,
                                                         request_slots)
            if user_act == 'inform':
                act = 'give_feedback'
                agent_action = self.construct_action(act, phase,
                                                     inform_slots,
                                                     request_slots)

        elif self.phase == 'met_before':
            if self.prev_action['act'] == 'give_feedback':
                phase = 'goal_elicitation'
                act = 'request'
                request_slots = 'primary_goal'
                agent_action = self.construct_action(act, phase,
                                                     inform_slots,
                                                     request_slots)

            else:
                act = 'give_feedback'
                agent_action = self.construct_action(act, phase, inform_slots,
                                                     request_slots)

        elif self.phase == 'goal_elicitation':
            if self.prev_action['act'] == 'give_feedback':
                phase = 'interest_elicitation'
                act = 'request'
                request_slots = 'interest'
                agent_action = self.construct_action(act, phase,
                                                     inform_slots,
                                                     request_slots)

            else:
                act = 'give_feedback'
                agent_action = self.construct_action(act, phase, inform_slots,
                                                     request_slots)

        elif self.phase == 'interest_elicitation':
            if self.prev_action['act'] == 'give_feedback':
                phase = 'session_recommendation'
                act = 'request'
                request_slots = 'goal_session'
                agent_action = self.construct_action(act, phase,
                                                     inform_slots,
                                                     request_slots)

            else:
                act = 'give_feedback'
                agent_action = self.construct_action(act, phase, inform_slots,
                                                     request_slots)

        elif self.phase == 'session_recommendation':
            reco_type = "session"
            alt_reco_type = "person"
            agent_action, phase = self.process_recommendation(reco_type,
                                                              alt_reco_type,
                                                              state,
                                                              user_action)

        elif self.phase == 'person_recommendation':
            reco_type = "person"
            alt_reco_type = "food"
            agent_action, phase = self.process_recommendation(reco_type,
                                                              alt_reco_type,
                                                              state,
                                                              user_action)

        elif self.phase == 'food_recommendation':
            if not state['goal_food']:
                phase = 'selfie'
                act = 'request'
                request_slots = 'selfie'
                agent_action = self.construct_action(act, phase,
                                                     inform_slots,
                                                     request_slots)

            elif self.prev_action['act'] == 'inform':
                act = 'request'
                request_slots = 'feedback'
                agent_action = self.construct_action(act, phase,
                                                     inform_slots,
                                                     request_slots)

            elif user_act == 'inform':
                if 'feedback' in user_inform_slots.keys():
                    if user_inform_slots['feedback']:
                        act = 'glad'
                        agent_action = self.construct_action(act, phase,
                                                             inform_slots,
                                                             request_slots)
                    else:
                        act = 'sorry'
                        agent_action = self.construct_action(act, phase,
                                                             inform_slots,
                                                             request_slots)
                else:
                    act = 'inform'
                    inform_slots = {'food': 'info_food'}
                    agent_action = self.construct_action(act, phase,
                                                     inform_slots,
                                                     request_slots)
            else:
                phase = 'selfie'
                act = 'request'
                request_slots = 'selfie'
                agent_action = self.construct_action(act, phase,
                                                     inform_slots,
                                                     request_slots)

        elif self.phase == 'selfie':
            if user_act == 'null':
                phase = 'farewell'
                act = 'bye'
                agent_action = self.construct_action(act, phase,
                                                     inform_slots,
                                                     request_slots)

            else:
                act = 'take_selfie'
                agent_action = self.construct_action(act, phase,
                                                     inform_slots,
                                                     request_slots)

        self.current_action = agent_action
        self.phase = phase

        # Social Reasoner
        cs = user_action['CS'][1]
        agent_intention = common_func.get_agent_intention(agent_action)
        agent_action['CS'] = self.generate_conv_strat_data_driven(cs, agent_intention, print_info=False)

        return agent_action, self.prev_rapp_val[0]

    def process_recommendation(self, reco_type, alt_reco_type, state, user_action):
        inform_slots = {}
        request_slots = ''
        user_act = user_action['act']
        user_inform_slots = user_action['inform_slots']
        phase = self.phase

        if not state['goal_' + reco_type]:
            phase = alt_reco_type + '_recommendation'
            act = 'request'
            request_slots = 'goal_' + alt_reco_type
            agent_action = self.construct_action(act, phase,
                                                 inform_slots,
                                                 request_slots)

        elif user_act == 'inform':
            if 'feedback' in user_inform_slots.keys():
                if user_inform_slots['feedback']:
                    act = 'glad'
                    agent_action = self.construct_action(act, phase,
                                                         inform_slots,
                                                         request_slots)
                else:
                    act = 'sorry'
                    agent_action = self.construct_action(act, phase,
                                                         inform_slots,
                                                         request_slots)

            elif 'send_msg_tlink' in user_inform_slots.keys():
                if user_inform_slots['send_msg_tlink']:
                    act = 'send_msg'
                    agent_action = self.construct_action(act, phase,
                                                         inform_slots,
                                                         request_slots)
                else:
                    act = 'no_worries'
                    agent_action = self.construct_action(act, phase,
                                                         inform_slots,
                                                         request_slots)

            else:
                if state[reco_type] == 0 or user_inform_slots['another_reco']:
                    act = 'inform'
                    inform_slots = {reco_type: "info_" + reco_type}
                    agent_action = self.construct_action(act, phase,
                                                         inform_slots,
                                                         request_slots)

                else:
                    phase = alt_reco_type + '_recommendation'
                    act = 'request'
                    request_slots = 'goal_' + alt_reco_type
                    agent_action = self.construct_action(act, phase,
                                                         inform_slots,
                                                         request_slots)

        else:
            if self.prev_action['act'] == 'inform':
                act = 'request'
                request_slots = 'feedback'
                agent_action = self.construct_action(act, phase,
                                                     inform_slots,
                                                     request_slots)

            elif self.prev_action['act'] == 'glad':
                act = 'request'
                request_slots = 'send_msg_tlink'
                agent_action = self.construct_action(act, phase,
                                                     inform_slots,
                                                     request_slots)

            elif state[reco_type] < self.max[reco_type]:
                act = 'request'
                request_slots = 'another_reco'
                agent_action = self.construct_action(act, phase,
                                                     inform_slots,
                                                     request_slots)

            else:
                phase = alt_reco_type + '_recommendation'
                act = 'request'
                request_slots = 'goal_' + alt_reco_type
                agent_action = self.construct_action(act, phase,
                                                     inform_slots,
                                                     request_slots)

        return agent_action, phase

    @staticmethod
    def construct_action(act, phase, inform_slots, request_slots):
        agent_action = {}
        pass
        agent_action['act'] = act
        agent_action['phase'] = phase
        agent_action['inform_slots'] = inform_slots
        agent_action['request_slots'] = request_slots
        return agent_action

    def generate_conv_strat_data_driven(self, user_cs, agent_ti, print_info=False):
        if print_info:
            print("User CS (Agent): ", user_cs)
            print("CS user: ", self.prev_cs_user)
            print("CS agent: ", self.prev_cs_agent)
            print("TS agent: ", self.prev_ti_agent)
            print("Rapport: ", self.prev_rapp_val)

        self.prev_cs_user.pop()
        if type(user_cs) != list:
            user_cs = [user_cs]
        user_cs = [ucs.upper() for ucs in user_cs]
        self.prev_cs_user.insert(0, user_cs)

        self.prev_ti_agent.pop()
        if type(agent_ti) != list:
            agent_ti = [agent_ti]
        agent_ti = [ati.lower() for ati in agent_ti]
        self.prev_ti_agent.insert(0, agent_ti)

        R = torch.Tensor(np.array(self.prev_rapp_val))[None, :]
        U = torch.Tensor(np.array([self.enc['user'].fit(u) for u in self.prev_cs_user])).transpose(0, 1)[None, :, :]
        A = torch.Tensor(np.array([self.enc['agent'].fit(a) for a in self.prev_cs_agent])).transpose(0, 1)[None, :, :]
        AT = torch.Tensor(np.array([self.enc['agent_intention'].fit(at) for at in self.prev_ti_agent])).transpose(0, 1)[None, :, :]

        rapp = self.re(U, A, R, AT)
        self.prev_rapp_val.pop()
        self.prev_rapp_val.insert(0, common_func.clip(rapp[0][0]))

        prob_pred = self.sr(U, A, R, AT)
        y_pred, agent_cs = common_func.get_cs(self.enc['agent'], self.thresh, prob_pred.data.cpu().numpy())

        self.prev_cs_agent.pop()
        self.prev_cs_agent.insert(0, agent_cs)

        if print_info:
            print("Agent CS (Agent): ", agent_cs)
            print("###########################################################")

        return y_pred, agent_cs
