from argparse import ArgumentParser
from Env import Environment

from policy_iteration import PolicyIteration

# from value_iteration import ValueIteration


# ================================== VARIABLES ================================== #

# GLOBAL VARIABLES
STATES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
ACTIONS = ["keep", "replace"]
START_STATE = 0

COST = 500
PROFIT = 150

PROBA = 0.9

# SPECIFIC POLICY ITERATION
V0_VAL = 0  # Initial value of the value functions for each state
GAMMA = 0.9  # Discount factor
THETA = 0.01  # threshold parameter that defines when the change in the value function is negligible (i.e. when we can stop process of Policy Evaluation)
SEED = 42  # seed (for matter of reproducible results)


# ================================== ENVIRONMENT INITIALIZATION ================================== #

env = Environment(
    actions=ACTIONS,
    states=STATES,
    transition_proba_p=PROBA,
    positive_reward=PROFIT,
    negative_reward=COST,
    start_state=START_STATE,
)

# ================================== MAIN FUNCTION ================================== #

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--strat",
        dest="type_of_strategy",
        default="policy_iter",
        type=str,
        help="Choice of strategy.",
    )

    # Nb of possible states
    n = len(STATES)

    args = parser.parse_args()

    if args.type_of_strategy == "policy_iter":
        agent = PolicyIteration(
            env=env,
            v0_val=V0_VAL,
            gamma=GAMMA,
            theta=THETA,
            seed=SEED,
        )
        agent.policy_iteration()

    elif args.type_of_strategy == "value_iter":
        # agent = Value_Iteration()
        # agent.play()
        pass
