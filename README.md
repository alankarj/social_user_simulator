# Datasets and models
The user simulator RE and SR models have been trained using prelim-analysis repo. In the following, <cluster_id> can take values in {0, 1, 'all'}, where 0 is P-Type, 1 is I-Type and 'all' is full dataset.
- **weights_re_<cluster_id>.t7**: Weights for rapport estimator.
- **weights_sr_user_<cluster_id>.t7**: Weights for the user social reasoner.
- **weights_sr_agent.t7**: Weights for the agent social reasoner.
- **all_rapps_gold_full.pkl**: Ground truth rapport values before agent's recommendation feedback task strategy.

In addition, the following files contain details of the domain ontology: **slot_set**: list of slots, **phase_set**: list of phases in the interaction, **sys_act_set**: list of agent (system) task strategy strings (acts), **user_act_set**: list of user acts.

# Commands for reproduction of results
`python src/run.py`
Modify the dialog_config file: (a) all_together = True for unimodal, False for bimodal, (b) num_dialogs are simulated max_iter times to obtain estimates (mean, std. dev.) of KL and CvM divergences.
