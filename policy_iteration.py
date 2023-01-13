"""
Policy Iteration algorithm for Gridworld problem.

We are assuming that:
    - we start with the equiprobable policy;
    - when the action send us to a cell outside the grid, we will stay in the same cell.
"""

import time
import numpy as np


class PolicyIteration:
    def __init__(self, env, v0_val, gamma, theta, seed):
        """
        Initialize our PolicyIteration class.

        Args:
            env (Environment): Environment class
            v0_val (int): initial value for the value function
            gamma (float): gamma parameter (between 0 and 1)
            theta (float): threshold parameter that defines when the change in the value function is negligible (i.e. when we can stop process)
            seed (int): seed (for matter of reproducible results)
        """
        self.env = env
        self.v0_val = v0_val
        self.gamma = gamma
        self.theta = theta
        self.seed = seed

        self.v = []
        self.pi = []
        self.optimal_actions = []

    def policy_iteration(self):
        """
        Runs the Policy Iteration algorithm:
            - Policy Evaluation
            - Policy Improvement

        Args:
            env (Environment): Environment class
            v0_val (int): initial value for the value function
            gamma (float): gamma parameter (between 0 and 1)
            theta (float): threshold parameter that defines when the change in the value function is negligible (i.e. when we can stop process)
            seed (int): seed (for matter of reproducible results)
        """
        # Data storage initialization
        return_mem = []
        timestep_mem = []
        greedy_return_mem = []
        timesteps = 0

        # Generate initial value function and policy
        self.v = self.get_init_v(self.v0_val)
        self.pi, self.optimal_actions = self.get_initial_policy()

        # Initialize policy as a NOT STABLE one
        policy_stable = False
        # Handle events

        while True:
            if not policy_stable:
                timesteps += 1
                print(f"\nIteration {timesteps} of Policy Iteration algorithm")

                # ============== Policy Evaluation Step ============== #
                self.policy_evaluation(
                    self.env, self.v, self.pi, self.gamma, self.theta
                )

                # ============== Policy Improvement Step ============== #
                policy_stable = self.policy_improvement(
                    self.env, self.v, self.pi, self.gamma
                )

                print(
                    f"\The whole Policy Iteration (eval -> improvement -> eval -> ...) algorithm converged after {timesteps} steps"
                )

            else:
                time.sleep(2)

    def policy_evaluation(self, environment, v, pi, gamma, theta):
        """
        Applies the policy evaluation algorithm.

        Args:
            env (Environment): environment
            v (array): numpy array representing the value function
            pi (array): numpy array representing the policy
            gamma (float): gamma parameter (between 0 and 1)
            theta (float): threshold parameter that defines when the change in the value function is negligible
        """

        delta = theta + 1
        iter = 0

        while delta >= theta:
            old_v = v.copy()
            delta = 0

            # Iterate all states
            for state in environment.states:  # [1,...,10]
                # Run one iteration of the Bellman update rule for the value function
                self.bellman_update(environment, v, old_v, state, pi, gamma)
                # Compute difference for EACH STATE, and take the maximum difference
                delta = max(delta, abs(old_v[state] - v[state]))
            iter += 1

        print(
            f"\nValue function updated: the Policy Evaluation algorithm converged after {iter} sweeps"
        )

    def policy_improvement(self, environment, v, pi, gamma):
        """
        Applies the Policy Improvement step.

        Args:
            board (Environment): gridworld environment
            v (array): numpy array representing the value function
            pi (array): numpy array representing the policy
            gamma (float): gamma parameter (between 0 and 1)
        """
        policy_stable = True

        # Iterate all states
        for state in environment.states:  # [1,...,10]
            old_pi = pi[state, :].copy()

            # Instanciate best actions list & best action val
            best_actions = []
            max_action_val = None

            ############ COMPUTE the ACTION-value function Q_𝜋(s,a) for each action ############
            for a in environment.actions:
                # Get next state
                s_prime = self.get_next_state(state, a, environment.transition_proba_p)

                # Get ACTION value
                curr_action_value = (
                    environment.reward_function(s_prime, a) + gamma * v[s_prime]
                )

                if max_action_val is None:  # If no best action, add this one
                    max_action_val = curr_action_value
                    best_actions.append(a)
                elif (
                    curr_action_value > max_action_val
                ):  # If better than precedent action, replace
                    max_action_val = curr_action_value
                    best_actions = [a]
                elif (
                    curr_action_value == max_action_val
                ):  # If the Action-value for this specific actions equals another Action-value of another action, both deserve to be taken
                    best_actions.append(a)

            # Define new policy π(a|s) with the following variables
            # - pi: current policy, will be updated by new_policy
            # - current_state: state for which the policy should be updated
            # - new_policy: best actions to take for this specific state
            # - actions: all set of actions
            self.pi[state] = self.improve_policy(
                pi, state, best_actions, environment.actions
            )

            # Check whether the policy has changed
            if (old_pi == self.pi[state, :]).all():
                policy_stable = True

        if not policy_stable:
            print(f"\nPolicy improved for all states.")
        else:
            # Refresh the display
            print(f"\nPolicy is now STABLE !")

        return policy_stable

    def bellman_update(self, env, v, old_v, state, pi, gamma):
        """
        Applies the Bellman update rule to the value function.

        Args:
            - env (Environment): environment
                - env.actions
                - env.states
                - env.reward_list
                - env.transition_function
            - v (array): numpy array representing the value function
            - old_v (array): numpy array representing the value function on the last iteration
            - current_state (int):  current state
            - gamma (float): gamma parameter (between 0 and 1)
        """

        # The value function on the terminal state always has value 0
        # if state == goal_state:
        #     return None

        total = 0

        for action in env.actions:

            # Get next state
            s_prime = self.get_next_state(state, action, env.transition_proba_p)

            # Compute reward for next state
            reward = self.env.reward_function(s_prime, action)

            # Compute V value for specific state
            total += pi[state, action] * (reward + (gamma * old_v[s_prime]))

        # UPDATE OF the VALUE function
        v[state] = total

    def improve_policy(self, pi, current_state, best_actions, actions, probability):
        """
        Defines a new policy π(a|s) given the new best actions (computed by the Policy improvement)

        Args:
            pi (array): numpy array representing the policy
            current_state (int): Current state s_t of agent
            best_actions (list): list with best actions
            actions (list): list of every possible action (given by board.actions)
        """

        possible_actions = self.env.action_function(current_state)
        prob = self.env.state_transition_function(
            self, current_state, action, next_state, probability
        )

        for a in actions:
            pi[current_state, a] = prob if a in best_actions else 0

        return pi[current_state]

    def get_init_v(self, v0):
        """
        Defines initial value function v_0.

        Args:
            env: Environment
            v0 (float): initial FLOAT value for the value function (equal for every state)
        Returns:
            v0 (array): initial value ARRAY
        """

        # Init
        v0 = v0 * np.ones(len(self.environment.states))

        return v0

    def get_initial_policy(self):
        """
        Defines the initial policy.

        - Policy is a matrix s.t. pi[s, a] = Pr[A = a | S = current_state]

        - Args:
            board_height (int): height of the grid
            board_width (int): width of the grid
            actions (array): Array of actions delivered by board.actions

        - Returns:
            pi (array): numpy array representing the equiprobably policy
        """

        # One policy per action, for each state => p[x,y] = [0.25,0.25,0.25,0.25] & p[x,y,a] = 0.25
        pi = {
            1: {"keep": 0.5, "replace": 0.5},
            2: {"keep": 0.5, "replace": 0.5},
            3: {"keep": 0.5, "replace": 0.5},
            4: {"keep": 0.5, "replace": 0.5},
            5: {"keep": 0.5, "replace": 0.5},
            6: {"keep": 0.5, "replace": 0.5},
            7: {"keep": 0.5, "replace": 0.5},
            8: {"keep": 0.5, "replace": 0.5},
            9: {"keep": 0.5, "replace": 0.5},
            10: {"keep": 0, "replace": 1},
        }
        opt_act_temp = []

        for i in range(self.environment.states):
            opt_act_temp.append("Keep or Replace")
            i += 1

        self.optimal_actions = np.array(self.optimal_actions)

        return pi, self.optimal_actions

    def get_next_state(self, current_state, action, probability):
        """Computes next state from current state and action.
        Args:
            current_state (int): between [1,10]
            a (int): action
        Returns:
            s_prime (int): value of the next state
        """
        roll = np.random.rand()

        # Compute next state according to the action

        if (action == "keep") and (current_state <= 8) and roll <= probability:
            s_prime = current_state + 1
            return s_prime

        elif (action == "keep") and (current_state <= 8) and roll <= 1 - probability:
            s_prime = current_state + 2
            return s_prime

        elif (action == "keep") and (current_state == 9):
            s_prime = 10
            return s_prime

        elif action == "replace":
            s_prime = 1
            return s_prime
