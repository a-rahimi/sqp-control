from numpy import *
import scipy.optimize as O
import sim

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


def test_gradient_controller():
  "generate and draw a random path"
  path = sim.genpath()

  target_speed = .1
  s0 = (.5, -.7, pi/2, .04, -pi/4)
  x,y,alpha,speed,theta = s0
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
    s0 = sim.apply_control(s0, u0)['val']
    S.append( s0 )
    Ls.append( L(u0)[0] )

  sim.show_results(path, S, Ls)


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
