import pdb
from numpy import *
import scipy.optimize as O
import pylab as P
import cvxpy as CX

import sim
import discrete


def one_step_cost(u, path, s0, u0, target_speed, lambda_speed):
  """
  Evaluate the scalar function

    L(u) = p(x(u)) + lambda_speed * (speed(u)-target_speed)^2

  and its dervivatives wrt u.
  """

  # parse arguments
  x0,y0,alpha0,speed,theta0 = s0
  ddx0,dtheta0 = u0
  ddx,dtheta = u

  # parse s(u)
  s = sim.apply_control(s0,u, derivs={'du'})
  x,y,alpha,speed,theta = s['val']

  p = path(reshape((x,y),(1,2)), derivs={'dx'})

  # L at the new state
  L = p['val']**2 + lambda_speed * (speed - target_speed)**2

  # derivative of L at the new state
  dL = 2*p['val']*dot(p['dx'],s['du'][:2]) + lambda_speed*2*array((speed-target_speed, 0))

  return L,dL.ravel()


def greedy_controller(L, u0, s0, max_dtheta, max_theta, max_ddx):
  """A greedy controller.

  Minimizes a function L. L takes as input the current state s0 and a
  control signal u0. It returns the tuple of its value and its
  derivatives wrt u.

  Find a control signal u that minimizes L(u) subject to the following
  constraints:

         |dtheta| < max_dtheta
         |theta(u)| < max_theta
         |ddx| < max_ddx
  """
  # the bounds are:
  #   -max_dtheta < dtheta < max_dtheta
  #   -max_theta < theta0+dtheta < max_theta
  # which we combine into
  #   max(-max_dtheta, -max_theta-theta0) < theta < min(max_dtheta,max_theta-theta0)
  _,_,_,_,theta0 = s0
  bounds=(((-max_ddx, max_ddx),
          (max(-max_dtheta, -max_theta-theta0),
           min(max_dtheta, max_theta-theta0))))

  res = O.minimize(L, array(u0), method='L-BFGS-B', jac=True, bounds=bounds)
  return res['x']


def test_greedy_controller():
  "generate and draw a random path"
  path = sim.genpath()

  s0 = (.5, -.7, pi/2, .04, -pi/4)
  x,y,alpha,speed,theta = s0
  u0 = 0.1, 0.01
  max_dtheta = pi/20
  max_theta = pi/4
  max_ddx = 0.01
  target_speed = .1
  lambda_speed = 10.

  S = [s0]
  Ls = []
  for i in xrange(80):
    def L(u):
      return one_step_cost(u, path, s0, u0, target_speed, lambda_speed)

    if True:
      sim.deriv_check(L, u0, 1e-2)

    u0 = greedy_controller(L, u0, s0, max_dtheta, max_theta, max_ddx)
    s0 = sim.apply_control(s0, u0)['val']
    S.append( s0 )
    Ls.append( L(u0)[0] )

  sim.show_results(path, S, Ls, animated=0.1)





def min_quad_with_linear_eq(H, a, K, u):
  """minimize a quadratic subject to linear constraints:

      min_x   a'x + 1/2 x' H x
      s.t.    K x = u

  The lagrangian is
      max_l min_x a'x + 1/2 x' H x + l'(Kx-u)
    = max_l -l'u + min_x  a'x + 1/2 x' H x + l'Kx
    = max_l -l'u + min_x  1/2 x' H x + (a+K'l)'x
    = max_l -l'u + min_x  1/2 (x+x_) H (x+x_) -  x_' H x_

  where x_ = H\(a+K'l). The optimizing value of x is -x_. Substituting
  this in reduces the inner minimization to

    max_l -l'u - x_' H x_

  Taking derivatives wrt to l and setting to zero gives

    0 = u + KH\(a+K'l)

    l = -(KH\K') \ (KH\a + u)
    x_ = H\a - H\K' (KH\K')\ (KH\a + u)
  """
  # compute H\K' and H\a from  H\ [K', a]
  H_Ka = linalg.solve(H,vstack((K,a)).T)
  H_K = H_Ka[:,:-1]
  H_a = H_Ka[:,-1]

  x_ = H_a - dot(H_K, linalg.solve(dot(K,H_K), dot(K,H_a)+u))
  return -x_


def min_quad_with_linear_eq0(H, a, K, u):
  """minimize a quadratic subject to linear constraints:

      min_x   a'x + 1/2 x' H x
      s.t.    K x = u

  Parametrize the constraint as set x = x0 + Z y, where the columns of
  Z are orthonormal and span the null space of K, x0 is feasible, and
  y is unconstrained.

  Setting the derivatives of the objective wrt y to 0 gives

     0 = Z'a + Z' H (Z y + x0)

     y = - (Z'HZ)\Z' (a + H x0)
     x = x0 - Z(Z'HZ)\Z' (a + H x0)

  Given the SVD of K=USV', the columns of Z are the rows of V' with
  zero singular values, and x0 is pinv(K) u.
  """
  # K = dot(U*s, V)
  U,s,V = linalg.svd(K)
  # x0 = pinv(H) u
  x0 = dot(V[:len(s),:].T, dot(U.T,u)/s)
  # ZZ'
  Z = V[len(s):,:].T

  return x0 - dot(Z, dot(linalg.solve( dot(Z.T,dot(H,Z)), Z.T),  dot(H,x0) + a))


def test_min_quad_with_linear_eq():
  d = 3
  D = 6

  # a random instance
  H = random.randn(D,D)
  H = dot(H,H.T)

  a = random.randn(D)

  K = random.randn(d,D)
  u = random.randn(d)

  x = min_quad_with_linear_eq(H,a,K,u)

  assert all( abs(dot(K,x)-u) < 1e-10 ), 'Solution is not feasible'

  # an instance where the first 3 elements are constrained to 0.
  K = eye(d,D)
  u = zeros(d)

  x = min_quad_with_linear_eq(H,a,K,u)
  assert all(abs(x[:d])<1e-10) , 'Solution is not feasible'

  x_ = -linalg.solve(H[d:,d:], a[d:])
  assert all(abs(x[d:]-x_) < 1e-10), 'Incorrection solution'

  print 'OK'


def planner_sqp(T, cost, dynamics, s0, max_dtheta, max_theta, max_ddx,
                max_iters=30, show_results=None):
  """
  Find control signals u_1...u_T, u_t=(ddx_t,dtheta_t) and the ensuing
  states s_1...s_T that

  minimize  sum_{t=1}^T  cost(s_t)
  subject to        s_t = f(u_t, s_{t-1})
                    | dtheta_t | < max_dtheta
                    | ddx_t | < max_ddx

  Solves this as a Sequential Quadratic program by approximating L by
  a quadratic, and f by an affine transform.
  """
  # the initial sequence of states
  Sv,_,_ = discrete.continous_plan(T, 0.8, s0, L)
  # recover controls from the state sequence
  Uv = XXX


  for it in xrange(max_iters):
    # the instantaneous costs for this iteration
    L = [inf] * T
    # objective and constraints of the quadratic problem
    objective = 0
    constraints = []
    S = [None] * T
    U = [None] * T
    S[0] = s0

    for t in xrange(1,T):
      # cost(s_t) and its derivatives
      c = cost(Sv[t], {'ds','ds2'})
      # f(u_t, s_{t-1}) and its derivatives
      f = dynamics(Uv[t], Sv[t-1], {'du','ds'})

      L[t] = c['val']

      S[t] = CX.Variable(5, name='s_%d'%t)
      U[t] = CX.Variable(2, name='s_%d'%t)

      objective += c['val'] + sum(c['ds']*S[t]) + 0.5*CX.quad_form(S[t],c['ds2'])
      constraints += [S[t] == f['val'] + sum(f['ds']*S[t-1]) + sum(f['du']*U[t])]

    # solve for S and U
    p = CX.Problem(CX.Minimize(objective), constriants)
    r = p.solve()

    # check convergence
    if all(abs(Unew-U) < 1e-5):
      break
    U = Unew

    if show_results:
      show_results(S, L, it)


def test_planner_sqp():
  path = sim.genpath()

  s0 = (.5, -.7, pi/2, .04, -pi/4)
  x,y,alpha,speed,theta = s0
  u0 = 0.1, 0.01
  max_dtheta = pi/20
  max_theta = pi/4
  max_ddx = 0.01
  target_speed = .1
  lambda_speed = 10.

  def cost(s, derivs=set([])):
    """p(x)**2 + lambda (speed - target_speed)^2 and its derivatives wrt s"""

    x,y,_,speed,_ = s

    p = path(reshape((x,y),(1,2)),
             derivs=set([{'ds':'dx', 'ds2':'dx2'}[d] for d in derivs]) )

    res = {'val':  p['val']**2 + lambda_speed * (speed-target_speed)**2 }

    if 'ds' in derivs:
      ds = zeros(5)
      ds[[0,1,3]] = 2*p['val']*p['dx'] + 2*lambda_speed * (speed-target_speed)
      res['ds'] = ds
    if 'ds2' in derives:
      ds2 = zeros((5,5))
      ds2[ix_([0,1,3],[0,1,3])] = 2*p['val']*p['dx2'] + 2*outer(p['dx'],p['dx'])
      res['ds2'] = ds2
    return res

  def show_results(S, L, it):
    sim.show_results(path, S, L, animated=0)
    P.figure(0).title('iteration %d'%it)
    print 'iteration', it

  S = planner_sqp(15, cost, apply_control, s0, max_dtheta, max_theta, max_ddx,
                max_iters=30, show_results=show_results)


def transition_to_action(s0,s1, dynamics, max_dtheta, max_ddx, max_iter=30,
                         max_line_searches=10):
  """recovers a control signal u that can cause a transition from s0 to s1.

  specifically, minimize || f(s0,u) - s1 ||^2 as a Sequential
  Quadratic Program.
  """
  # the current step size along the search direction recovered by the QP
  step = 1.

  # initial guess
  s0 = array(s0)
  s1 = array(s1)
  u0 = array((0,0))

  def cost(u):
    "|| f(s0,u) - s1 ||^2"
    return sum((array(dynamics(s0,u)['val']) - s1)**2)

  # the value of the initial guess
  best_cost = cost(u0)

  for it in xrange(max_iter):
    f = dynamics(s0, u0, derivs={'du'})

    # linearize || f(s0,u) - s1 ||^2 about u0 and solve as a QP
    u = CX.Variable(len(u0), name='u')
    objective = CX.sum(CX.square(array(f['val']) + vstack(f['du'])*(u-u0) - s1))
    p = CX.Problem(CX.Minimize(objective),
                      [CX.abs(u[0]) <= max_ddx,
                       CX.abs(u[1]) <= max_dtheta])
    r = p.solve()
    unew = array(u.value.flat)

    # line search along unew-u0 from u0
    line_search_success = False
    for line_searches in xrange(max_line_searches):
      new_cost = cost(u0 + step*(unew-u0))
      if new_cost < best_cost:
        # accept the step
        best_cost = new_cost
        u0 = u0 + step*(unew-u0)
        # grow the step for the next iteration
        step *= 1.2
        line_search_success = True
      else:
        # shrink the step size and try again
        step *= 0.5
    if not line_search_success:
      # convergence is when line search fails.
      return u0

  print 'Warning: failed to converge'
  return u0


def test_transition_to_action():
  s0 = (.5, -.7, pi/2, .04, -pi/4)
  max_dtheta = pi/20
  max_ddx = 0.01

  for it in xrange(5):
    # apply a random control
    u = array(((random.rand()-0.5)*max_ddx, (random.rand()-0.5)*max_dtheta))
    s1 = sim.apply_control(s0,u)['val']

    # recover it
    u_ = transition_to_action(s0, s1, sim.apply_control, max_dtheta, max_ddx)

    assert all(abs(u_-u)/abs(u)<1e-2), 'Recovered control %s is not %s'%(u_,u)

  print 'OK'


def test_transition_to_action1():
  # a random problem
  path = sim.genpath()
  s0 = (.5, -.7, pi/2, .04, -pi/4)

def test_transition_to_action1():
  # a random problem
  path = sim.genpath()
  s0 = (.5, -.7, pi/2, .04, -pi/4)
  max_dtheta = pi/20
  max_ddx = 0.01

  # solve it as discrete
  def cost(s):
    return path(reshape(s[:2],(1,2)))['val']**2
  S,Ls,_ = discrete.continuous_plan(15, cost, 0.8, s0)

  # recover control actions and apply to current state
  Scont = [s0]
  for s0,s1 in zip(S,S[1:]):
    u = transition_to_action(s0, s1, sim.apply_control, max_dtheta, max_ddx)
    Scont.append( sim.apply_control(Scont[-1], u)['val'] )

  # show discrete trajectory
  sim.show_results(path, S, Ls, animated=0)
  # show continuous trajectory
  sim.animate_car(P.figure(0).add_subplot(1,1,1), Scont,
                  remove_car=False, sleep=0.,
                  alphas=linspace(0.1,.5,len(S))**2,
                  color='r')
  P.draw()
