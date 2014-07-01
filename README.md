Finite-horizon non-linear controller using Sequential Dynamic Programming
===========

A [Python notebook documents](http://nbviewer.ipython.org/github/a-rahimi/sqp-control/blob/master/Following%20an%20Uncertain%20Path.ipynb) this package.


Steering a car is hard because what you do at any given moment has consequences down the line. At worse, a bad car control strategy can go unstable. More typically, it causes annoying driving. This is why race car drivers memorize the track in every muscle of their bodies, why parallel parking is hard, and why drivers lose control of their cars.

I demonstrate planner on a simplified car model. The car can accelerate and steer wheel, but there are limits on its acceleration, the rate at which it can turn its wheels, and the steering angle.

The goal is to follow a given fuzzy path with high fidelity while maintaining a target speed. The path is fuzzy in that we can't exactly determine the car's distance to it. Instead, the path is given to us as a signed score function, which I'll descrie later.

A controller steers and accelerates the car subject to the car's limits. I briefly describe a starwman greedy controller. At each time step, it proposes a move that results in the best performance at the next time step. When it starts on the road, this controller can track it well. But its approach to the road is too aggressive when it starts away from the road.

With enogh work, a greedy controller can be made to work well, of course, but the ultimate solution is always a planner.

A planner considers the long term consequences of its actions. Before proposing a move, it determines the control signals it might need to generate a few steps later, and proposes the move that results in the best average performance in the near future. At each time step, based on the measurements available at the current time step, the planner hallucinates an entire trajectory of control signals. We use only the first step it proposes and ignore the rest of the hallucinated trajectory. This is called the "model predictive control".
