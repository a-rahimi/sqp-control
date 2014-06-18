import pdb
from numpy import *
import pylab as P
import time
import itertools


# problem parameters
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

  def ddcircle(X,u):
    """d^2/dx^2 ||x-u|| = d (x-u) / ||x-u||
                        = I/||x-u|| - (x-u)(x-u)' / ||x-u||^3
    """
    c = circle(X,u)
    return (eye(2) - outer(X-u,X-u)/c**2) / c

  def path(X,derivs={}):
    """||x-u0|| w0 + ||x-u1|| w1 and its derivatives"""
    p = w0 * circle(X,u0) + w1 * circle(X,u1)
    res = {'val': p}
    if 'dx' in derivs:
      res['dx'] = w0 * dcircle(X,u0) + w1 * dcircle(X,u1)
    if 'dx2' in derivs:
      res['dx2'] = w0 * ddcircle(X,u0) + w1 * ddcircle(X,u1)
    return res

  path.params = {'u0': u0, 'u1': u1,
                 'w0': w0, 'w1': w1}
  return path


def draw_path(ax, path):
  """draw the given the path function on the matplotlib axis"""

  # Evaluate the path on a grid
  X,Y = meshgrid(linspace(-1,1,80), linspace(-1,1,80))
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
  return ax.contour(X,Y,Z,v, colors=colors, linewidths=linewidths)


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
  x,y,alpha,speed,theta = s

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

  X = [ (-.1,-.5, pi/3, 0, -pi/6),
        (-.1,-.5, pi/3, 0, 0.),
        (-.1,-.5, pi/3, 0, pi/6),
        (-.1,-.5, 0, 0, -pi/6),
        (-.1,-.5, 0, 0, 0),
        (-.1,-.5, 0, 0, pi/6),
        ]
  hs = []

  for x in X:
    for h in hs:
      h.remove()
    hs = draw_car(P.gca(), x, color='b')
    P.draw()
    #time.sleep(1.)
    P.waitforbuttonpress()


def animate_car(ax, S, drawing=None, remove_car=True, sleep=.1, alphas=None,
                **kwargs):
  """animate the car along the given state sequence. returns an object
  that represent's the car's shape for subsequent animation.  by default,
  produces an animation. but if remove_car=False, draws a final frame with
  the car ghosted.
  """
  # initialize the drawing if needed
  if not drawing:
    drawing = CarDrawing(**kwargs)
    ax.add_artist(drawing.trail)

  if alphas is None:
    alphas = itertools.repeat(1.)

  for s,alpha in itertools.izip(S,alphas):
    # parse the state
    x,y,_,_,_  = s
    # append to the trail
    drawing.trail.set_xdata(hstack((drawing.trail.get_xdata(), x)))
    drawing.trail.set_ydata(hstack((drawing.trail.get_ydata(), y)))

    # remove the car if so requested
    if remove_car:
      drawing.remove_car()

    # draw the car
    drawing.h = draw_car(ax, s, **kwargs)
    for h in drawing.h:
      h.set(alpha=alpha)

    if sleep:
      P.draw()
      time.sleep(sleep)





def test_animate_car():
  "run the car with constant control input"
  # initial state
  s0 = (.0, .1, pi*.3, .04, -pi/12)

  # apply the controls
  S = [s0]
  for u in [(0.0,0.03)]*20:
    S.append(apply_control(S[-1],u)['val'])

  P.clf()
  P.axis('equal')
  animate_car(P.gca(), S, remove_car=False, sleep=0,
              alphas=linspace(0.1,1.,len(S))**4)
  P.draw()


def apply_control(s0,u, derivs={}):
  """s(u). apply controls u to state s0.

  With

    s=(x,y,alpha,speed,theta)
    u=(ddx,dtheta)

  s(u) is

    speed(u) = speed0 + ddx
    theta(u) = theta0 + dtheta
    alpha(u) = alpha0 + speed/car_length * tan(theta(u))
    x(u)     = x0 + v(alpha(u)) * speed
  """
  # parse arguments
  x0,y0,alpha0,speed0,theta0 = s0
  ddx,dtheta = u

  speed = speed0 + ddx
  theta = theta0 + dtheta
  alpha = alpha0 + speed/car_length * tan(theta)
  alpha = alpha % (2*pi)

  ca,sa = cos(alpha), sin(alpha)

  x = x0 + ca * speed
  y = y0 + sa * speed

  # aggregate into a state object

  res = { 'val': array((x,y,alpha,speed,theta)) }

  if 'du' in derivs:
    dspeeddu = (1,0)
    dthetadu = (0,1)
    dalphadu = (1/car_length*tan(theta), speed/car_length * (1+tan(theta)**2))
    dxdu = speed*outer((-sa,ca), dalphadu) + outer((ca,sa), (1,0))
    res['du'] = (dxdu[0], dxdu[1], dalphadu, dspeeddu, dthetadu)

  if 'ds' in derivs:
    assert 'du' in derivs, 'Need du to compute ds'
    dspeedds = (0,0,0,1,0)
    dthetads = (0,0,0,0,1)
    dalphads = (0,0,1) + dalphadu
    dxds = (1,0,-sa*speed,ca-sa*speed*dalphadu[0],-sa*speed*dalphadu[1])
    dyds = (0,1,ca*speed,sa+ca*speed*dalphadu[0],ca*speed*dalphadu[1])
    res['ds'] = (dxds,dyds,dalphads,dspeedds,dthetads)

  return res


def deriv_check(f, x, rtol, dh=1e-4):
  dx = dh * random.randn(len(x))

  numeric = f(x+dx)[0] - f(x)[0]
  analytic = dot(f(x)[1], dx)

  assert all(abs(analytic-numeric)/linalg.norm(dx) < rtol), \
    'num %s ana %s'%(numeric, analytic)


def test_apply_control():
  s0 = array((.5, -.7, pi/2, .04, -pi/4))
  u0 = 0.1, 0.01

  # check the derivative of the dynamics wrt u and s for each component of s
  for i in xrange(len(s0)):
    # check derivatives wrt du
    def f(u):
      res = apply_control(s0,u,{'du'})
      return res['val'][i], res['du'][i]
    deriv_check(f,u0,1e-3,dh=1e-6)

    # check derivatives wrt ds
    def f(s):
      res = apply_control(s,u0,{'du','ds'})
      return res['val'][i], res['ds'][i]
    deriv_check(f,s0,1e-3,dh=1e-6)

  print 'OK'


def show_results(path, S, costs, animated=0, target_speed=.1):
  P.ion()
  fig1 = P.figure(0); fig1.clear()
  ax = fig1.add_subplot(1,1,1)
  ax.axis('scaled')
  ax.axis([-1,1,-1,1])

  if path:
    draw_path(ax, path)

  if animated:
    animate_car(ax, S, remove_car=True, sleep=animated)
  else:
    animate_car(ax, S, remove_car=False, sleep=0,
                alphas=linspace(0.1,.5,len(S))**2)

  if costs is not None:
    # show summary statistics
    fig2 = P.figure(1); fig2.clear()
    ax2 = fig2.add_subplot(2,1,1)
    ax3 = fig2.add_subplot(2,1,2)

    ax2.plot(costs, label='state cost');
    ax2.set_ylabel('Controller score')
    ax3.plot([speed for _,_,_,speed,_ in S], label='actual')
    ax3.plot([0, len(S)], [target_speed, target_speed], 'k--',
                       label='target')
    ax3.legend(loc='best')
    ax3.set_ylabel('speed')
