from src import dialog_config


class StateTracker:
    def __init__(self):
        self.phase = None
        self.turn = None

        self.count_reco = None
        self.num_accepted = None

        self.reward = None
        self.dialog_over = None

        self.count_slots = None
        self.reward_slots = None
        self.slot_set = None

    def initialize(self, slot_set):
        self.slot_set = slot_set
        self.turn = 1
        self.slot_set['turn'] = {}
        self.slot_set['turn'] = self.turn
        self.count_slots = dialog_config.count_slots
        self.count_reco = {}
        for c_slot in self.count_slots:
            self.count_reco[c_slot] = 0
            self.slot_set[c_slot] = self.count_reco[c_slot]

        self.reward_slots = dialog_config.reward.keys()
        self.num_accepted = {}
        self.slot_set['num_accepted'] = {}
        for c_slot in self.count_slots:
            self.num_accepted[c_slot] = {}
            self.slot_set['num_accepted'][c_slot] = {}
            for r_slot in self.reward_slots:
                self.num_accepted[c_slot][r_slot] = 0
                self.slot_set['num_accepted'][c_slot][r_slot] = \
                self.num_accepted[c_slot][r_slot]

        self.reward = 0
        self.dialog_over = False

    def update(self, agent_action=None, user_action=None):
        self.turn += 1
        self.slot_set['turn'] = self.turn

        if agent_action:
            if agent_action['phase'] != self.phase:
                self.phase = agent_action['phase']
                print("-" * 50, end='')
                print(self.phase, end='')
                print("-" * (50 - len(self.phase)))

            act = agent_action['act']
            inform_slots = agent_action['inform_slots'].keys()

            if act == 'inform':
                for slot in self.slot_set.keys():
                    if slot in inform_slots:
                        self.slot_set[slot] = agent_action['inform_slots'][slot]

                for slot in self.count_slots:
                    if slot in inform_slots:
                        self.count_reco[slot] += 1
                        self.slot_set[slot] = self.count_reco[slot]

        elif user_action:
            act = user_action['act']
            inform_slots = user_action['inform_slots'].keys()

            if act == 'bye':
                self.dialog_over = True

            elif act == 'inform':
                for slot in self.slot_set.keys():
                    if slot in inform_slots:
                        self.slot_set[slot] = user_action['inform_slots'][slot]

                for r_slot in self.reward_slots:
                    if r_slot in inform_slots:
                        if user_action['inform_slots'][r_slot]:
                            self.reward += dialog_config.reward[r_slot]
                            for c_slot in self.count_slots:
                                if self.phase == c_slot + "_recommendation":
                                    self.num_accepted[c_slot][r_slot] += 1
                                    self.slot_set['num_accepted'][c_slot][
                                        r_slot] = self.num_accepted[c_slot][
                                        r_slot]

        # print(self.reward)
        # print(self.num_accepted)
        # print(self.num_accepted['person'])

        return self.reward, self.dialog_over, self.slot_set


if __name__ == "__main__":
    state_tracker = StateTracker()
    state_tracker.initialize()

    agent_action = {}
    pass
    agent_action['act'] = 'inform'
    agent_action['phase'] = 'session'
    agent_action['inform_slots'] = {'send_msg_tlink': True}
    agent_action['request_slots'] = ''
    state_tracker.update(user_action=agent_action)
