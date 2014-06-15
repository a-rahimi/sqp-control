import pdb
from numpy import *
import sim


def planner_discrete(T, nstates, cost, reachable_set, s0):
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
  V = inf+zeros((nstates, T+2))
  I = zeros((nstates, T+1), int)
  # base case: Q_T(s,s_{T-1}) = 0
  V[:,-1] = 0
  for t in xrange(T,0,-1):
    for si in xrange(nstates):
      # the cost of transitioning to each reachable state
      q = [(cost(sj) + V[sj,t+1], sj) for sj in reachable_set(si) ]
      # the best state transision
      v, sj = min( q )
      # populate the table
      I[si,t] = sj
      V[si,t] = v

  # find the minimizers starting at V_1(s_0)
  Si = [s0]
  for t in xrange(1,T):
    Si.append( I[Si[-1], t] )
  return Si


def planner_stationary(gamma, nstates, cost, reachable_set, max_iterations=10):
  """
  Find states s_1...s_T that

  minimize  sum_{t=1}^T  gamma^(t-1) cost(s_t)
  subject to             s_t = reachable_states(s_{t-1})

  Returns a policy function s' = P(s) such that s' is reachable from s
  and so that the sequence s_t = P(s_{t-1}) starting with t=1 is
  optimal.
  """
  # the immediate cost of each state
  c0 = array([cost(si) for si in xrange(nstates)])
  # the state transition policy
  P = [None]*nstates

  # value function. initialize with the immediate cost
  v = c0 * 0
  vnew = v.copy()

  for it in xrange(max_iterations):
    for si in xrange(nstates):
      vnew[si], P[si] = min( (c0[sj] + gamma * v[sj], sj)
                             for sj in reachable_set(si) )

    if all(vnew == v):
      break
    v = vnew
  print 'ran for %d iterations'%it
  return P


def apply_policy(s0,P,T):
  """Apply the given policy function T times, starting with s0. Return
  the sequence of states generated.
  """
  Si = [s0]
  for t in xrange(1,T):
    Si.append( P[Si[-1]] )
  return Si

def quantize(v, minv, maxv, nv):
  v = clip(v, minv, maxv)
  iv = int(round((v-minv) / (maxv-minv) * (nv-1)))
  assert 0<=iv and iv<nv, 'bad quantization'
  return iv

def unquantize(iv, minv, maxv, nv):
  assert 0<=iv and iv<nv, 'invalid discrete coordinate'
  v = float32(iv) * (maxv-minv) / (nv-1) + minv
  #assert minv<=v and v<=maxv, 'bad unquantization'
  return v

def quantize_angle(v,nv):
  """a quantization that avoids the wraparound. iv=nv-1 corresponds
  the quantum that precedes 2 pi, instead of 2*pi."""
  return quantize(v%(2*pi),0,2*pi*(1-1./nv),nv)

def unquantize_angle(iv,nv):
  return unquantize(iv,0,2*pi*(1-1./nv),nv)


class DiscreteStates:
  """Functions that map between a continuous vector representation of
  the state to an integer."""

  def __init__(self, nalpha=8, nspeed=4, minspeed=0, maxspeed=.1,
               nx=29, minx=-1, maxx=1., miny=-1, maxy=1.,
               ntheta=3, max_theta=pi/4,
               ndtheta=3,  max_dtheta=5*pi/20,
               nddx=3, max_ddx=5*0.01):
    self.nalpha = nalpha
    self.minspeed = minspeed
    self.maxspeed = maxspeed
    self.nspeed = nspeed
    self.max_theta = max_theta
    self.ntheta = ntheta
    self.minx= minx
    self.maxx = maxx
    self.miny = miny
    self.maxy = maxy
    self.nx = nx
    self.ny = nx
    self.ndtheta = ndtheta
    self.max_dtheta = max_dtheta
    self.nddx = nddx
    self.max_ddx = max_ddx

    # a cache of reachable states
    self.reachable = [None]*self.nstates()

  def nstates(self):
    return self.nx * self.ny * self.nalpha * self.nspeed * self.ntheta


  def to_integer(self, s):
    # parse the state
    x, y, alpha, speed, theta = s

    # quantize each component
    ix = quantize(x, self.minx, self.maxx, self.nx)
    iy = quantize(y, self.miny, self.maxy, self.ny)
    ialpha = quantize_angle(alpha, self.nalpha)
    ispeed = quantize(speed, self.minspeed, self.maxspeed, self.nspeed)
    itheta = quantize(theta, -self.max_theta, self.max_theta, self.ntheta)

    return self.qcoords_to_integer(ix,iy,ialpha,ispeed,itheta)

  def qcoords_to_integer(self, ix, iy, ialpha, ispeed, itheta):
    "bijection from (ix,iy,ialpha,ispeed,itheta) to [0...nstates]"
    return (((iy*self.nx + ix)*self.nalpha + ialpha) * self.nspeed + ispeed) * self.ntheta + itheta

  def integer_to_qcoords(self, si):
    "bijection from [0...nstates] to (ix,iy,ialpha,ispeed,itheta)"
    r, itheta = divmod(si, self.ntheta)
    r, ispeed = divmod(r, self.nspeed)
    r, ialpha = divmod(r, self.nalpha)
    iy, ix = divmod(r, self.nx)
    return ix,iy,ialpha,ispeed,itheta

  def to_continuous(self, si):
    ix,iy,ialpha,ispeed,itheta = self.integer_to_qcoords(si)

    # map integer indices to real numbers
    x = unquantize(ix, self.minx, self.maxx, self.nx)
    y = unquantize(iy, self.miny, self.maxy, self.ny)
    alpha = unquantize_angle(ialpha, self.nalpha)
    speed = unquantize(ispeed, self.minspeed, self.maxspeed, self.nspeed)
    theta = unquantize(itheta, -self.max_theta, self.max_theta, self.ntheta)

    return array((x, y, alpha, speed, theta))

  def reachable_states(self, si):
    "the set of discrete states reachable from the given discrete state"
    # return a cached result if available
    if self.reachable[si] is not None:
      return self.reachable[si]

    s = self.to_continuous(si)
    res = set([
      self.to_integer( sim.apply_control(s, (ddx,dtheta))['val'] )
      for ddx in linspace(-self.max_ddx, self.max_ddx, self.nddx)
      for dtheta in linspace(-self.max_dtheta, self.max_dtheta, self.ndtheta)
      ])

    # cache before returning
    self.reachable[si] = res
    return res


if 'disc' not in globals():
  disc = DiscreteStates()


def test_discrete_states():
  nstates = disc.nstates()
  print 'There are %d states'%nstates

  # inspect every state
  for si in xrange(disc.nstates()):
    s = disc.to_continuous(si)

    # ensure it convert back to the same integer
    ti = disc.to_integer(s)
    assert ti == si, \
      'rediscritized state #%d is #%d. continuous is %s'%(si,ti,s)

    R = disc.reachable_states(si)

    # ensure its reachable states are valid
    for sj in R:
      assert 0<=sj and sj<nstates, 'state %d reaches invalid state %d'%(si,sj)

    # the quantized representation of each component
    ix,iy,ialpha,ispeed,itheta = disc.integer_to_qcoords(si)

    # It should be possible to turn the wheel at every state
    s_left = sim.apply_control(s, (0.,disc.max_dtheta))['val']
    si_left = disc.to_integer(s_left)
    if itheta<disc.ntheta-1:
      assert si_left!=si and si_left in R, 'left turn is not reachable'

    s_right = sim.apply_control(s, (0.,-disc.max_dtheta))['val']
    si_right = disc.to_integer(s_right)
    if itheta>0: # only check if the turn is possible
      assert si_right!=si and si_right in R, 'right turn is not reachable'

    # It should be possible to accelerate at every state
    s_faster = sim.apply_control(s, (disc.max_ddx,0.))['val']
    si_faster = disc.to_integer(s_faster)
    if ispeed<disc.nspeed-1:
      assert si_faster!=si and si_faster in R, "can't go faster"

    s_slower = sim.apply_control(s, (-disc.max_ddx,0.))['val']
    si_slower = disc.to_integer(s_slower)
    if ispeed>0:
      assert si_slower!=si and si_slower in R, "can't go slower"

    # The highest speed should cause the car to move
    if ispeed==disc.nspeed-1 and ix>0 and ix<disc.nx-1 and iy>0 and iy<disc.nx-1:
      s_next = sim.apply_control(s,(0.,0.))['val']
      si_next = disc.to_integer(s_next)
      assert si_next!=si and si_next in R, "no way to move"

  print 'OK'



def continuous_plan(T, cost, gamma, s0, disc=disc):
  P = planner_stationary(gamma, disc.nstates(),
                         lambda si: cost(disc.to_continuous(si)),
                         disc.reachable_states)
  Si = apply_policy(disc.to_integer(s0), P, 15)

  # convert states to continuous
  S = [disc.to_continuous(si) for si in Si]
  # the cost of each state
  Ls = [cost(s) for s in S]

  return S,Ls,Si


def test_discrete_planner():
  """a planner that plots a path with a fixed speed no adherence to
  the steering limits.
  """

  path = sim.genpath()
  target_speed = .1
  lambda_speed = 10.

  def cost(s):
    "the cost is just the deviation form the path"
    x,y,_,speed,_ = s
    return path(reshape((x,y),(1,2)))['val']**2 + lambda_speed * (speed-target_speed)**2

  s0 = (-.5, -.7, pi/2, .01, -pi/4)
  S,Ls,_ = continuous_plan(40, cost, 0.999, s0)
  sim.show_results(path, S, Ls, animated=0.1)
  sim.show_results(path, S, Ls, animated=0.)
  return S
