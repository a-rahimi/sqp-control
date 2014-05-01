import pdb
from numpy import *
import pylab as P
import time
import itertools
import scipy.optimize as O

P.ion()

# problem parameters
target_speed = .1
lambda_speed = 10.
car_length = 0.1

def genpath():
  """Generate a random smooth path, represented as an implicit
  function f(x)=0, x \in R^2.

  f(x) takes the form a mixture of two Gaussians with random centers,
  variances, and equal weights.
  """
  # means
  u0 = array([-1., 0])
  u1 = array([1., 0])

  # weights
  w0 = .5+random.rand()
  w1 = -.5-random.rand()

  def circle(X,u):
    "||x-u||"
    return sqrt( sum((X-u)**2,axis=1) )

  def dcircle(X,u):
    """d||x-u|| = (x-u) / ||x-u||"""
    return (X-u) / circle(X,u)

  def path(X,derivs={}):
    """A mixture of two random Gaussians. Closure from genpath"""
    p = w0 * circle(X,u0) + w1 * circle(X,u1)
    res = {'val': p}
    if 'dx' in derivs:
      res['dx'] = w0 * dcircle(X,u0) + w1 * dcircle(X,u1)
    return res

  path.params = {'u0': u0, 'u1': u1,
                 'w0': w0, 'w1': w1}
  print path.params
  return path


def draw_path(ax, path):
  """draw the given the path function on the matplotlib axis"""

  # Evaluate the path on a grid
  X,Y = meshgrid(linspace(-1,1,40), linspace(-1,1,40))
  Z = path(hstack((X.reshape(-1,1), Y.reshape(-1,1))))['val'].reshape(X.shape)

  # representative values of the path function
  v = linspace(Z.min(), Z.max(),10)

  # default curve parameters
  linewidths = zeros(len(v)) + .5
  colors = [(0.5,0.5,0.5)]*len(v)

  # the path value closest to zero
  izero = argmin(abs(v))

  # request a path through path=0, and change its parameters
  v[izero] = 0
  linewidths[izero] = 5
  colors[izero] = 'k'

  # contour plot for the given values
  ax.contour(X,Y,Z,v, colors=colors, linewidths=linewidths)


def test_path():
  "generate and draw a random path"
  path = genpath()
  P.ion()
  P.clf()
  draw_path(P.gca(), path)
  P.draw()



class CarDrawing():
  """drawing of the car and its movement trail"""

  def __init__(self, **kwargs):
    # the trail
    self.trail = P.Line2D([],[], marker='o', **kwargs)
    # the car itself
    self.h = []

  def remove_car(self):
    for h in self.h:
      h.remove()

  def remove(self):
    "remove the car and the trail"
    self.remove_car()
    self.trail.remove()


def draw_car(ax, s, **kwargs):
  # parse the state
  (x,y),alpha,speed,theta = s

  # rotation matrix for the car relative to the current universe
  dx,dy = cos(alpha),sin(alpha)
  R = array([(dx, -dy),
             (dy,  dx)])
  assert linalg.det(R) > 0.

  # scale the coordinate system so that the car's length is 1
  R *= car_length

  # rotation matrix of the wheels relative to the car
  Rwheel = array([(cos(theta),-sin(theta)),
                  (sin(theta), cos(theta))])
  # rotation matrix of the wheels relative to the current universe
  Rwheel = dot(R,Rwheel)

  # body of the car in the curernt universe
  body = dot([(-2,-1),
              ( 2,-1),
              ( 2, 1),
              (-2, 1)], R.T) + (x,y)

  # the front axle, in the current universe
  axle = dot([(2, -1.5),
              (2,  1.5)], R.T) + (x,y)

  # a wheel. a vertical segment in its own coordinate system
  wheel = dot([(-.5, 0),
               ( .5, 0)], Rwheel.T)


  hbody = P.Polygon(body, fill=False, edgecolor=kwargs.get('color','b'))
  haxle = P.Line2D(axle[:,0], axle[:,1], **kwargs)
  hleftwheel = P.Line2D([axle[0,0] + wheel[:,0]],
                        [axle[0,1] + wheel[:,1]],
                        **kwargs)
  hrightwheel = P.Line2D([axle[1,0] + wheel[:,0]],
                         [axle[1,1] + wheel[:,1]],
                        **kwargs)
  hcenter = P.Line2D([x], [y], marker='o', ms=5, mec=None, **kwargs)

  # the artists for the car
  hs =  (hbody, haxle, hleftwheel, hrightwheel, hcenter)

  # add artists to the axis if supplied
  if ax:
    for h in hs:
      ax.add_artist(h)

  return hs


def test_draw_car():
  P.ion()
  P.clf()
  P.axis('scaled')
  P.axis([-1,1,-1,1])

  X = [ ((-.1,-.5), pi/3, 0, -pi/6),
        ((-.1,-.5), pi/3, 0, 0.),
        ((-.1,-.5), pi/3, 0, pi/6),
        ((-.1,-.5), 0, 0, -pi/6),
        ((-.1,-.5), 0, 0, 0),
        ((-.1,-.5), 0, 0, pi/6),
        ]
  hs = []

  for x in X:
    for h in hs:
      h.remove()
    hs = draw_car(P.gca(), x, color='b')
    P.draw()
    #time.sleep(1.)
    P.waitforbuttonpress()


def animate_car(ax, S, drawing=None, remove_car=True, sleep=.1, alphas=None):
  """animate the car along the given state sequence. returns an object
  that represent's the car's shape for subsequent animation.  by default,
  produces an animation. but if remove_car=False, draws a final frame with
  the car ghosted.
  """
  # initialize the drawing if needed
  if not drawing:
    drawing = CarDrawing()
    ax.add_artist(drawing.trail)

  if alphas is None:
    alphas = itertools.repeat(1.)

  for s,alpha in itertools.izip(S,alphas):
    # parse the state
    (x,y),_,_,_  = s
    # append to the trail
    drawing.trail.set_xdata(hstack((drawing.trail.get_xdata(), x)))
    drawing.trail.set_ydata(hstack((drawing.trail.get_ydata(), y)))

    # remove the car if so requested
    if remove_car:
      drawing.remove_car()

    # draw the car
    drawing.h = draw_car(ax, s)
    for h in drawing.h:
      h.set(alpha=alpha)

    if sleep:
      P.draw()
      time.sleep(sleep)

  P.draw()


def test_animate_car():
  "run the car with constant control input"
  # initial state
  x,y,alpha,speed,theta = (.6, .2, pi*.3, .04, -pi/12)
  s0 = (array((x,y)), alpha, speed, theta)

  # apply the controls
  S = [s0]
  for u in [(0.0,0.03)]*20:
    S.append(apply_control(S[-1],u)['val'])

  P.clf()
  P.axis('equal')
  animate_car(P.gca(), S, remove_car=False, sleep=0,
              alphas=linspace(0.1,1.,len(S))**4)



def apply_control(s0,u, derivs={}):
  """s(u). apply controls u to state s0.

  With

    s=(x,alpha,speed,theta)
    u=(ddx,dtheta)

  and eta the function that normalizes its vector argument to unit
  length, s(u) is

    speed(u) = speed0 + ddx
    theta(u) = theta0 + dtheta
    alpha(u) = alpha0 + speed/car_length * tan(theta(u))
    x(u)     = x0 + v(alpha(u)) * speed
  """
  # parse arguments
  x0,alpha0,speed0,theta0 = s0
  ddx,dtheta = u

  speed = speed0 + ddx
  theta = theta0 + dtheta
  alpha = alpha0 + speed/car_length * tan(theta)
  alpha = alpha % (2*pi)

  x = x0 + array((cos(alpha), sin(alpha))) * speed

  # aggregate into a state object

  res = { 'val': (x,alpha,speed,theta) }

  if 'du' in derivs:
    dspeeddu = (1,0)
    dthetadu = (0,1)
    dalphadu = (1/car_length*tan(theta), speed/car_length * (1+tan(theta)**2))
    dxdu = speed*outer((-sin(alpha),cos(alpha)), dalphadu) + outer((cos(alpha),sin(alpha)), (1,0))
    res['du'] = (dxdu, dalphadu, dspeeddu, dthetadu)

  return res


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


def deriv_check(L, u, rtol, dh=1e-4):
  du = dh * random.randn(len(u))

  numeric = L(u+du)[0] - L(u)[0]
  analytic = dot(L(u)[1], du)

  assert abs(analytic-numeric)/linalg.norm(du) < rtol, \
    'num %s ana %s'%(numeric, analytic)



def show_results(path, S, costs):
  P.ion()
  fig1 = P.figure(0); fig1.clear()
  ax = fig1.add_subplot(1,1,1)
  ax.axis('scaled')
  ax.axis([-1,1,-1,1])
  draw_path(ax, path)

  fig2 = P.figure(1); fig2.clear()
  ax2 = fig2.add_subplot(2,1,1)
  ax3 = fig2.add_subplot(2,1,2)

  # show summary statistics
  ax2.plot(cost, label='state cost');
  ax2.set_ylabel('Controller score')
  ax3.plot([speed for dx,alpha,speed,theta in S], label='actual')
  ax3.plot([0, len(S)], [target_speed, target_speed], 'k--', label='target')
  ax3.legend(loc='best')
  ax3.set_ylabel('speed')

  # show the car animation
  #animate_car(ax, S, remove_car=True, sleep=0.1)
  animate_car(ax, S, remove_car=False, sleep=0,
              alphas=linspace(0.1,.5,len(S))**2)

  P.draw()


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



def planner_sqp(T, L, s0, max_dtheta, max_theta, max_ddx, max_iters=30):
  """
  Find control signals u_1...u_T, u_t=(ddx_t,dtheta_t) and the ensuing
  states s_1...s_T that

  minimize  sum_{t=1}^T  L(s_t)
  subject to             s_t = f(u_t, s_{t-1})
                         | dtheta_t | < max_dtheta
                         | ddx_t | < max_ddx

  Solves this as a Sequential Quadratic program by approximating L by
  a quadratic, and f by an affine transform.
  """
  # the initial sequence of states
  S = planner_discrete(T, L, u0, s0, max_dtheta, max_theta, max_ddx)

  for it in xrange(max_iters):
    pass


def planner_discrete(T, n, cost, reachable_set, s0):
  """
  Find states s_1...s_T that

  minimize  sum_{t=1}^T  cost(s_t)
  subject to             s_t = reachable_states(s_{t-1})

  Returns the tuple of the value and the state sequence.
  """
  # fill out the DP table in a backward pass.
  #  V_t(s_{t-1}) = min_s c(s) + V_{t+1}(s)   s.t. s reachable from s0
  #  I_t(s_{t-1}) = argmin_s c(s) + V_{t+1}(s)   s.t. s reachable from s0
  # Takes O(T d^2)
  V = inf+zeros((n, T+1))
  I = zeros((n, T), int)
  # base case: Q_T(s,s_{T-1}) = 0
  V[:,-1] = 0
  for t in xrange(T,0,-1):
    for si in xrange(states.nstates()):
      # the cost of transitioning to each reachable state
      q = [ cost(sj) + DP[t+1][sj]
            for sj in reachable_set(si) ]
      # the best state transision
      sj = argmin( q )
      # populate the table
      I[t][si] = sj
      Q[t][si] = q[sj]

  # find the minimizers starting at V_1(s_0)
  return [s0] + [ I[t][S[-1]] for t in xrange(1,T+1) ]


class DiscreteStates:
  """Functions that map between a continuous vector representation of
  the state to an integer."""
  alpha = None
  speed = target_speed
  minx,maxx = -1.,1.
  miny,maxy = -1.,1.

  ntheta = 8
  nx = int(ceil( (maxx-minx) / speed ))
  ny = int(ceil( (maxy-miny) / speed ))

  def nstates(self):
    return self.nx * self.ny * self.ntheta

  def continuous_state_to_integer(self, s):
    # parse the state
    (x,y), _, _,theta = s

    # quantize the components independently
    ix = round((x-self.minx) / self.speed)
    iy = round((y-self.miny) / self.speed)
    itheta = round(theta/(2*pi) * self.ntheta)

    # bijection from (ix,iy,itheta) to [0...nx*ny*ntheta]
    return (iy* self.nx + ix)* self.ntheta + itheta

  def integer_state_to_continuous(self, si):
    # bijection from [0...nx*ny*ntheta] to (ix,iy,itheta)
    r, itheta = divmod(si, self.ntheta)
    iy, ix = divmod(r, self.nx)

    # map to real
    x = ix * self.speed + self.minx
    y = iy * self.speed + self.miny
    theta = itheta * 2*pi/self.ntheta

    return ((x,y), self.alpha, self.speed, theta)


def test_discrete_states():
  disc = DiscreteStates()

  # inspect every state
  for si in xrange(disc.nstates()):
    s = disc.integer_state_to_continuous(si)

    # ensure it's a valid state
    (x,y), _, _,theta = s
    assert x>=disc.minx, '%s is out of bounds'%x
    assert x<=disc.maxx, '%s is out of bounds'%x
    assert y>=disc.miny, '%s is out of bounds'%x
    assert y<=disc.maxy, '%s is out of bounds'%x
    assert theta>=0, '%s is out of bounds'%x
    assert theta<=2*pi, '%s is out of bounds'%x

    # ensure it convert back to the same integer
    ti = disc.continuous_state_to_integer(s)
    assert ti == si, 'rediscritized state #%d is #%d. continuous is %s'%(si,ti,s)

  print 'OK'


def test_discrete_planner():
  disc = DiscreteStates()

  path = genpath()
  s0 = ((.5, -.7), pi/2, .04, -pi/4)

  def cost(si):
    "the cost of a discrete state s0."
    xy,_,_,_ = disc.integer_state_to_continuous(si)

    return path(xy)['val']

  def reachable_states(si):
    "the set of discrete states reachable from discrete state s0"
    (x,y),alpha,speed,theta = disc.integer_state_to_continuous(s0)

    si = [
      self.continuous_state_to_integer(x + self.speed * cos(theta+dtheta),
                                       y + self.speed * sin(theta+dtheta),
                                       alpha,
                                       speed,
                                       theta + dtheta)
          for dtheta in (-2*pi/self.ntheta, 0, 2*pi/self.ntheta) ]

  Si = planner_discrete(10, disc.nstates(), cost, reachable_states,
                        disc.continuous_state_to_integer(s0))

  # convert to continuous
  S = [si_to_s(si) for si in Si]
