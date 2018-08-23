import numpy as np
import model_params


class StateTracker:
    def __init__(self):
        self.state = None
        self.param = None
        self.dialog_over = None
        self.rapport = None

    def initialize(self, slot_set, param):
        self.state = {k: '' for k in slot_set.keys()}
        self.param = param

        self.state['turn'] = 0

        count_slots = self.param['count_slots']
        reward_slots = self.param['reward_slots']
        for c_slot in count_slots:
            self.state[c_slot] = 0
        self.state['num_accepted'] = {}
        for c_slot in count_slots:
            self.state['num_accepted'][c_slot] = {}
            for r_slot in reward_slots:
                self.state['num_accepted'][c_slot][r_slot] = 0

        self.dialog_over = False

        self.state['cs_dist_user'] = np.zeros((1, model_params.num_user_cs))
        self.state['cs_dist_agent'] = np.zeros((1, model_params.num_agent_cs))

        self.state['rapp'] = {}
        self.state['rapp'][0] = []
        self.state['rapp'][1] = []
        return self.dialog_over, self.state

    def update(self, agent_action=None, user_action=None, rapport=None):
        count_slots = self.param['count_slots']
        reward_slots = self.param['reward_slots']

        self.state['turn'] += 1

        if agent_action is not None:
            self.rapport = rapport
            self.state['phase'] = agent_action['phase']

            y_pred, cs = agent_action['CS']
            self.state['cs_dist_agent'] += y_pred
            self.state['agent_action'] = agent_action

            act = agent_action['act']
            inform_slots = agent_action['inform_slots'].keys()

            if act == 'inform':
                for slot in count_slots:
                    if slot in inform_slots:
                        self.state[slot] += 1

        if user_action is not None:
            y_pred, cs = user_action['CS']
            self.state['cs_dist_user'] += y_pred
            self.state['user_action'] = user_action

            act = user_action['act']
            inform_slots = user_action['inform_slots'].keys()

            if act == 'bye':
                self.dialog_over = True

            elif act == 'inform':
                for slot in self.state.keys():
                    if slot in inform_slots:
                        self.state[slot] = user_action['inform_slots'][slot]

                for r_slot in reward_slots:
                    if r_slot in inform_slots:
                        val = user_action['inform_slots'][r_slot]
                        if val:
                            self.state['rapp'][1].append(self.rapport)
                        else:
                            self.state['rapp'][0].append(self.rapport)
                        if val:
                            for c_slot in count_slots:
                                if self.state['phase'] == c_slot + "_recommendation":
                                    self.state['num_accepted'][c_slot][r_slot] += 1

        return self.dialog_over, self.state
