import json
from src.dialog_arbiter.state_tracker import StateTracker
from src.user_simulator.my_rule_based_user_sim import RuleBasedUserSimulator
from src.agent.rule_based_agent import RuleBasedAgent
from src import dialog_config


class DialogArbiter:
    def __init__(self, user, agent):
        self.agent = agent
        self.user = user
        self.dialog_over = False
        self.reward = 0
        self.state_tracker = StateTracker()
        self.agent_action = None
        self.user_action = None

    def initialize(self, slot_set):
        self.state_tracker.initialize(slot_set)
        self.agent_action = self.agent.initialize()
        self.user.initialize()
        print("New dialog, user goal, user type:")
        print(json.dumps(self.user.goal, indent=2))
        print(json.dumps(self.user.type, indent=2))

        reward, dialog_over, state = self.state_tracker.update(
            agent_action=self.agent_action)
        # print("Agent: ", end="")
        # dialog_config.print_info(self.agent_action)

    def next(self):
        reward, dialog_over, state = self.state_tracker.update(
            agent_action=self.agent_action)
        print("Agent: ", end="")
        #print(self.agent_action)
        dialog_config.print_info(self.agent_action)

        user_action = self.user.next(self.agent_action)
        reward, dialog_over, state = self.state_tracker.update(
            user_action=user_action)
        self.user_action = user_action
        print("User: ", end="")
        dialog_config.print_info(self.user_action)

        if not dialog_over:
            agent_action = self.agent.next(self.user_action, state)
            self.agent_action = agent_action

        return reward, dialog_over, state


if __name__ == "__main__":
    goal_type = "random"
    user = RuleBasedUserSimulator(goal_type)
    agent = RuleBasedAgent()
    slot_set = dialog_config.slot_set
    # print(slot_set)
    arbiter = DialogArbiter(user, agent)

    arbiter.initialize(slot_set)
    max_turns = 40
    dialog_over = False
    while not dialog_over:
        reward, dialog_over, state = arbiter.next()

    print("Reward:")
    print(reward)
    print("")

    count_slots = dialog_config.count_slots
    reward_slots = dialog_config.reward.keys()

    for c_slot in count_slots:
        print("Counts for " + c_slot + ":")
        print(state[c_slot])
        for r_slot in reward_slots:
            print("Num-accepted for " + c_slot + ", " + r_slot + ":")
            print(state['num_accepted'][c_slot][r_slot])
