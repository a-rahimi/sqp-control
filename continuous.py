from sim import show_results
import scipy.optimize as O

lambda_speed = 10.

def planner_sqp(T, L, s0, max_dtheta, max_theta, max_ddx, max_iters=30):
  """
  Find control signals u_1...u_T, u_t=(ddx_t,dtheta_t) and the ensuing
  states s_1...s_T that

  minimize  sum_{t=1}^T  L(s_t)
  subject to        s_t = f(u_t, s_{t-1})
                    | dtheta_t | < max_dtheta
                    | ddx_t | < max_ddx

  Solves this as a Sequential Quadratic program by approximating L by
  a quadratic, and f by an affine transform.
  """
  # the initial sequence of states
  S = planner_discrete(T, L, u0, s0, max_dtheta, max_theta, max_ddx)

  for it in xrange(max_iters):
    pass


def deriv_check(L, u, rtol, dh=1e-4):
  du = dh * random.randn(len(u))

  numeric = L(u+du)[0] - L(u)[0]
  analytic = dot(L(u)[1], du)

  assert abs(analytic-numeric)/linalg.norm(du) < rtol, \
    'num %s ana %s'%(numeric, analytic)


def one_step_cost(u, path, s0, u0, target_speed, lambda_speed):
  """
  Evaluate the scalar function

    L(u) = p(x(u)) + lambda_speed * (speed(u)-target_speed)^2

  and its dervivatives wrt u.
  """

  # parse arguments
  x0,alpha0,speed,theta0 = s0
  ddx0,dtheta0 = u0
  ddx,dtheta = u

  # parse s(u)
  s = apply_control(s0,u, derivs={'du'})
  x,alpha,speed,theta = s['val']

  p = path(x.reshape(1,2),derivs={'dx'})

  # L at the new state
  L = p['val']**2 + lambda_speed * (speed - target_speed)**2

  # derivative of L at the new state
  dL = 2*p['val']*dot(p['dx'],s['du'][0]) + lambda_speed*2*array((speed-target_speed, 0))

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
  _,_,_,theta0 = s0
  bounds=(((-max_ddx, max_ddx),
          (max(-max_dtheta, -max_theta-theta0),
           min(max_dtheta, max_theta-theta0))))

  res = O.minimize(L, array(u0), method='L-BFGS-B', jac=True, bounds=bounds)
  return res['x']


def test_gradient_controller():
  "generate and draw a random path"
  path = genpath()

  x,y,alpha,speed,theta = (.5, -.7, pi/2, .04, -pi/4)
  s0 = ((x,y), alpha, speed,theta)
  u0 = 0.1, 0.01
  max_dtheta = pi/20
  max_theta = pi/4
  max_ddx = 0.01

  S = [s0]
  Ls = []
  for i in xrange(80):
    def L(u):
      return one_step_cost(u, path, s0, u0, target_speed, lambda_speed)
    if True:
      deriv_check(L, u0, 1e-2)

    u0 = greedy_controller(L, u0, s0, max_dtheta, max_theta, max_ddx)
    s0 = apply_control(s0,u0)['val']
    S.append( s0 )
    Ls.append( L(u0)[0] )

  show_results(path, S, Ls)
