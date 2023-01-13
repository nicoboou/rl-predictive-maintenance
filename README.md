# rl-predictive-maintenance

## Problem statement

At the end of each production cycle (e.g. seasonal) a candy production line must decide
whether to keep a machinery again or replace it with a new one. A machinery at cycle t has
a corresponding efficiency state $s_t \in S = {1, 2, ..., 10}$. We know the machinery’s state at the
first cycle $s_1 = 1$, and it has probability $p = 0.9$ to go to efficiency state $s_{t+1} = min{s_{t+1}, 10}$
and probability $1 − p$ to go to efficiency state $s_{t+1} = min{s_{t + 2}, 10} if not replaced by a
new one. At each efficiency state $s$, it produces $y(s) = 8 + s − 0.15s^2$ tons of candy over the
corresponding production cycle. We assume a machinery must be replaced upon completion
of the production cycle $s_t = 10$ since it becomes too unproductive. The net cost of replacing
a machine is $c = 500$ k€ and the profit contribution of candy is $m = 150$ k€ per ton.

## Problem formulation

1. Write down the **_Bellman optimality equation_** of the value function.
2. Find the replacement policy that maximizes the _expected long term cumulative_ profits using **Value Iteration** to solve the problem (Policy Iteration and Linear Programming methods can be considered too)
3. Test the sensitivity of the optimal policies to different problem parameters
