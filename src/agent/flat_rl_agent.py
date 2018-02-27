from src import dialog_config
import json

feasible_actions = dialog_config.feasible_actions
total_actions = len(feasible_actions)
print(total_actions)
print(json.dumps(dialog_config.feasible_actions, indent=2))
print(feasible_actions[5])