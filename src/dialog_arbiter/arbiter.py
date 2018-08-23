from dialog_arbiter.state_tracker import StateTracker


class DialogArbiter:
    def __init__(self, user, agent, slot_set, param_state_tracker):
        self.user = user
        self.agent = agent
        self.slot_set = slot_set
        self.param_state_tracker = param_state_tracker

        self.state_tracker = StateTracker()
        self.user_action = None
        self.agent_action = None
        self.rapport = None
        self.dialog_over = False
        self.reward = None
        self.s_t = None

    def initialize(self):
        self.dialog_over, self.s_t = self.state_tracker.initialize(self.slot_set, self.param_state_tracker)
        self.user.initialize()
        self.agent_action, self.rapport = self.agent.initialize()
        self.reward = 0

    def next(self, print_info=False):
        if print_info:
            self.print_info(agent_action=self.agent_action)
        self.state_tracker.update(agent_action=self.agent_action, rapport=self.rapport)

        self.user_action, r_t = self.user.next(self.agent_action, self.rapport)
        self.reward += r_t

        self.dialog_over, self.s_t = self.state_tracker.update(user_action=self.user_action)

        if print_info:
            self.print_info(user_action=self.user_action)
            print("Reward: ", r_t)

        if not self.dialog_over:
            self.agent_action, self.rapport = self.agent.next(self.s_t)

        return self.reward, self.dialog_over, self.s_t

    @staticmethod
    def print_info(agent_action=None, user_action=None):
        if agent_action is not None:
            print("Agent: ", end="")
            action = agent_action
        if user_action is not None:
            print("User: ", end="")
            action = user_action

        act = action['act']

        if act == 'inform':
            act_slots = action[act + '_slots']
            print(act, end='')
            for slot in act_slots:
                print('(' + str(slot) + '=' + str(act_slots[slot]), end='')
            print(')', end='')

        elif act == 'request':
            act_slots = action[act + '_slots']
            print(act, end='')
            print('(' + act_slots, end='')
            print(')', end='')

        else:
            print(act + '()', end='')

        print(', CS: ', action['CS'][1])
