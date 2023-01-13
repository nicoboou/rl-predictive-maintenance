class Environment:
    # A class to represent the entire Reinforcement Learning environment

    def __init__(
        self,
        actions,
        states,
        transition_proba_p,
        positive_reward,
        negative_reward,
        start_state=0,
    ):
        # Intialize environment.

        self.actions = actions
        self.states = states
        self.start_state = start_state
        self.current_state = start_state
        self.positive_reward = positive_reward
        self.negative_reward = negative_reward
        self.transition_proba_p = transition_proba_p

    def action_function(current_state):
        if current_state < 10:
            actions = ["keep", "replace"]
            return actions  # Action 'Keep

        elif current_state == 10:
            actions = ["replace"]
            return actions  # Action 'Replace'

    # Reward Function #
    def reward_function(state, action, s_next, positive_reward, negative_reward):
        if action == "keep":
            return (8 + state - 0.15 * state**2) * positive_reward  # Action 'Keep

        elif action == "replace":
            return (
                (8 + state - 0.15 * state**2) * positive_reward
            ) - negative_reward  # Action 'Replace'

    # State Transition Function #
    def state_transition_function(current_state, action, probability):
        """
        Return all non-zero probability transitions for this action
            from this state, as a list of (state, probability) pairs

        => Pr[s'|s,a] = Pr[S_{t+1} = s_prime | S_t = current_state, A_t = action]
        """

        if (action == "keep") and (current_state <= 8):
            return [
                (current_state + 1, probability),
                (current_state + 2, 1 - probability),
            ]

        if (action == "keep") and (current_state == 9):
            return [(10, 1)]

        if (action == "keep") and (current_state == 10):
            return [(0, 0)]

        if action == "replace":
            return [(1, 1)]
