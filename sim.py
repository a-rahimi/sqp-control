import pdb
from numpy import *
import pylab as P
import time
import scipy.optimize as O

P.ion()

car_length = 0.1

def genpath():
  """Generate a random smooth path, represented as an implicit
  function f(x)=0, x \in R^2.

  f(x) takes the form a mixture of two Gaussians with random centers,
  variances, and equal weights.
  """
  # means
  u0 = array([-random.rand(), 0])
  u1 = array([random.rand(), 0])

  # weights
  w0 = random.rand()
  w1 = -random.rand()

  def circle(X,u):
    "||x-u||"
    return sqrt( sum((X-u)**2,axis=1) )
  def dcircle(X,u):
    """d||x-u|| = (x-u) / ||x-u||"""
    return (X-u) / circle(X,u)

  def path(X,deriv=False):
    """A mixture of two random Gaussians. Closure from genpath"""
    p = w0 * circle(X,u0) + w1 * circle(X,u1)
    if not deriv:
      return p
    dp = w0 * dcircle(X,u0) + w1 * dcircle(X,u1)
    return p,dp

  path.params = {'u0': u0, 'u1': u1,
                 'w0': w0, 'w1': w1}
  print path.params
  return path


def draw_path(ax, path):
  """draw the given the path function on the matplotlib axis"""

  # Evaluate the path on a grid
  X,Y = meshgrid(linspace(-1,1,40), linspace(-1,1,40))
  Z = path(hstack((X.reshape(-1,1), Y.reshape(-1,1)))).reshape(X.shape)

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
    self.trail = P.Line2D([],[], **kwargs)
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
  (x,y),(dx,dy),speed,theta = s

  # rotation matrix for the car relative to the current universe
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

  X = [ ((-1.,-.5), (cos(pi/3), sin(pi/3)), -pi/6),
        ((-1.,-.5), (cos(pi/3), sin(pi/3)), 0.),
        ((-1.,-.5), (cos(pi/3), sin(pi/3)), pi/6),
        ((-1.,-.5), (cos(0), sin(0)), -pi/6),
        ((-1.,-.5), (cos(0), sin(0)), 0),
        ((-1.,-.5), (cos(0), sin(0)), pi/6),
        ]
  hs = []

  for x in X:
    for h in hs:
      h.remove()
    hs = draw_car(P.gca(), x, color='b')
    P.draw()
    #time.sleep(1.)
    P.waitforbuttonpress()


def animate_car(ax, S, drawing=None, remove_car=True, sleep=.1, alpha_inc=0):
  """animate the car along the given state sequence. returns an object
  that represent's the car's shape for subsequent animation.  by default,
  produces an animation. but if remove_car=False, draws a final frame with
  the car ghosted.
  """
  # initialize the drawing if needed
  if not drawing:
    drawing = CarDrawing()
    ax.add_artist(drawing.trail)

  for i,s in enumerate(S):
    # parse the state
    (x,y), (dx,dy),speed,theta  = s
    # append to the trail
    drawing.trail.set_xdata(hstack((drawing.trail.get_xdata(), x)))
    drawing.trail.set_ydata(hstack((drawing.trail.get_ydata(), y)))

    # remove the car if so requested
    if remove_car:
      drawing.remove_car()

    # draw the car
    drawing.h = draw_car(ax, s)
    if alpha_inc:
      for h in drawing.h:
        h.set(alpha=alpha_inc+i*alpha_inc)

    if sleep:
      P.draw()
      time.sleep(sleep)

  P.draw()



def apply_control(s0,u):
  "s(u). apply controls u to state s0"
  # parse arguments
  x0,n0,speed0,theta0 = s0
  ddx,dtheta = u

  speed = speed0 + ddx
  theta = theta0 + dtheta

  R = array(((  0,           -tan(theta0)),
             ( tan(theta0), 0          )))
  dn = speed0/car_length * dot(R,n0)
  n = n0 + dn
  n /= sqrt(n[0]**2 + n[1]**2)

  x = x0 + n*speed

  # aggregate into a state object
  s = (x,n,speed,theta)

  return s


def test_animate_car():
  "run the car with constant control input"
  # initial state
  x,y,dx,dy,speed,theta = (.6, .2, 0, 1, .04, -pi/12)
  s0 = (array((x,y)), array((dx,dy)), speed, theta)

  # apply the controls
  S = [s0]
  for u in [(0,0)]*20:
    S.append(apply_control(S[-1],u))

  P.clf()
  animate_car(P.gca(), S, remove_car=False, sleep=0, alpha_inc=1./len(S))
  print S

def one_step_cost(u, path, s0, u0, target_speed, lambda_speed, deriv):
  # parse arguments
  x0,dx0,theta0 = s0
  ddx0,dtheta0 = u0
  ddx,dtheta = u

  # parse s(u)
  x,dx,theta = apply_control(s0,u)

  res = path(x.reshape(1,2),deriv=deriv)
  if deriv:
    p,dp = res
  else:
    p,dp = res,None

  speed_diff = dx[0]**2 + dx[1]**2 - target_speed**2

  # L at the new state
  L = p + lambda_speed * abs(speed_diff)

  if not deriv:
    return L

  # dx/du
  dxdu = array(((cos(theta), -ddx*sin(theta)),
                (sin(theta),  ddx*cos(theta))))
  # dspeed_diff/du
  dspeed_diffdu = 2*dot(dx,dxdu)

  # derivative of L at the new state
  dL = dot(dp,dxdu) + lambda_speed * sign(speed_diff) * dspeed_diffdu

  return L,dL.ravel()


def greedy_controller(path, s0, u0, target_speed, lambda_speed,
                      max_dtheta, max_theta, max_ddx):
  """A greedy controller.

  The target path is given as a signed distance
  function. s0=(x0,dx0,theta0) is the current state of the car and
  u0=(ddx,dtheta) is the current control signal.

  Find a control signal u that minimizes

    L(u) = p(x(u)) + lambda_speed * | ||dx(u)||^2-target_speed^2 |

    s.t. |dtheta| < max_dtheta
         |theta(u)| < max_theta
         |ddx| < max_ddx

  Here, s(u) = (x(u), dx(u), theta(u)) is the updated state after the
  control u has been applied:

    dx(u)    =  dx0 + v(theta) ddx
    x(u)     =  x0 + dx(u)
    theta(u) =  theta0 + dtheta
  """
  def L(u):
    return one_step_cost(u, path, s0, u0, target_speed, lambda_speed,
                         deriv=True)

  x0,dx0,theta0 = s0

  # the bounds are:
  #   -max_dtheta < dtheta < max_dtheta
  #   -max_theta < theta0+dtheta < max_theta
  # which we combine into
  #   max(-max_dtheta, -max_theta-theta0) < theta < min(max_dtheta,max_theta-theta0)
  bounds=(((-max_ddx, max_ddx),
          (max(-max_dtheta, -max_theta-theta0),
           min(max_dtheta, max_theta-theta0))))
  res = O.minimize(L, array(u0), method='L-BFGS-B', jac=True, bounds=bounds)
  return res['x']


def test_gradient_controller():
  "generate and draw a random path"
  path = genpath()

  x,y,dx,dy,theta = (.4,0,0,.1,pi/12)
  s0 = ((x,y), (dx,dy), theta)
  u0 = 0.1, 0.01
  target_speed = 1.
  lambda_speed = 10.
  max_dtheta = pi/10
  max_theta = pi
  max_ddx = 0.01

  # derivative check
  def L(u):
    return one_step_cost(u, path, s0, u0, target_speed, lambda_speed, True)
  du = 1e-4 * random.randn(2)
  numeric = L(u0+du)[0] - L(u0)[0]
  analytic = dot(L(u0)[1], du)
  print 'num:', numeric
  print 'ana:', analytic
  assert abs(analytic-numeric)/linalg.norm(du) < 1e-2
  print 'OK derivative'

  P.ion()
  P.clf()
  P.axis('scaled')
  P.axis([-1,1,-1,1])
  ax = P.gca()
  draw_path(ax, path)

  hcar = draw_car(ax, s0)
  for h in hcar:
    h.set(color='k')


  print 'u0:', u0
  u = greedy_controller(path, s0, u0, target_speed, lambda_speed,
                        max_dtheta, max_theta, max_ddx)
  print 'u:', u
  s = apply_control(s0,u)
  draw_car(ax, s)

  P.draw()
