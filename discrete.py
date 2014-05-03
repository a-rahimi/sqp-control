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


def planner_discrete_stationary(gamma, nstates, cost, reachable_set,
                                max_iterations=1000):
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
  v = c0
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


class DiscreteStates:
  """Functions that map between a continuous vector representation of
  the state to an integer."""

  def __init__(self, theta=0, speed=.1, nalpha=8,
               minx=-1, maxx=1., miny=-1 ,maxy=1.):
    self.theta = theta
    self.speed = speed
    self.nalpha = 8
    self.minx= minx
    self.maxx = maxx
    self.miny = miny
    self.maxy = maxy
    self.nx = 1+int(ceil( (maxx-minx) / speed ))
    self.ny = 1+int(ceil( (maxy-miny) / speed ))

  def nstates(self):
    return self.nx * self.ny * self.nalpha

  def to_integer(self, s):
    # parse the state
    (x,y), alpha, _, _ = s

    # quantize the components independently
    x = clip(x, self.minx, self.maxx)
    ix = int(round((x-self.minx) / self.speed))
    y = clip(y, self.miny, self.maxy)
    iy = int(round((y-self.miny) / self.speed))
    alpha = alpha % (2*pi)
    ialpha = int(alpha/(2*pi) * self.nalpha)

    assert 0<=ix and ix<self.nx, 'bad ix %s from %s'%(ix,x)
    assert 0<=iy and iy<self.ny, 'bad iy %s from %s'%(iy,y)
    assert 0<=ialpha and ialpha<self.nalpha, \
      'bad ialpha %s from %s'%(ialpha,alpha)

    # bijection from (ix,iy,ialpha) to [0...nx*ny*nalpha]
    return (iy* self.nx + ix)* self.nalpha + ialpha

  def to_continuous(self, si):
    # bijection from [0...nx*ny*nalpha] to (ix,iy,ialpha)
    r, ialpha = divmod(si, self.nalpha)
    iy, ix = divmod(r, self.nx)

    # map to real
    x = ix * self.speed + self.minx
    y = iy * self.speed + self.miny
    alpha = ialpha * 2*pi/self.nalpha

    return ((x,y), alpha, self.speed, self.theta)

  def reachable_states(self, si):
    "the set of discrete states reachable from the given discrete state"
    (x,y),alpha,speed,theta = self.to_continuous(si)

    return [
      self.to_integer(((x + self.speed * cos(alpha+dalpha),
                        y + self.speed * sin(alpha+dalpha)),
                      alpha + dalpha,
                      speed,
                      theta))
      for dalpha in (-2*pi/self.nalpha, 0, 2*pi/self.nalpha) ]



def test_discrete_states():
  disc = DiscreteStates()

  nstates = disc.nstates()

  # inspect every state
  for si in xrange(disc.nstates()):
    s = disc.to_continuous(si)

    # ensure it's a valid state
    (x,y), alpha, _, _ = s
    assert x>=disc.minx, '%s is out of bounds'%x
    assert x<=disc.maxx, '%s is out of bounds'%x
    assert y>=disc.miny, '%s is out of bounds'%x
    assert y<=disc.maxy, '%s is out of bounds'%x
    assert alpha>=0, '%s is out of bounds'%x
    assert alpha<=2*pi, '%s is out of bounds'%x

    # ensure it convert back to the same integer
    ti = disc.to_integer(s)
    assert ti == si, \
      'rediscritized state #%d is #%d. continuous is %s'%(si,ti,s)

    # ensure its reachable states are valid
    for sj in disc.reachable_states(si):
      assert 0<=sj and sj<nstates, 'state %d reaches invalid state %d'%(si,sj)

  print 'OK'


def test_discrete_planner():
  disc = DiscreteStates()

  path = sim.genpath()
  s0 = ((.5, -.7), pi/2, .04, -pi/4)

  def cost(si):
    "the cost of a discrete state"
    xy,_,_,_ = disc.to_continuous(si)
    return path(reshape(xy,(1,2)))['val']**2

  #Si = planner_discrete(10, disc.nstates(), cost, disc.reachable_states,
  #                      disc.to_integer(s0))
  P = planner_discrete_stationary(0.8, disc.nstates(), cost,
                                  disc.reachable_states)
  Si = apply_policy(disc.to_integer(s0), P, 15)

  # convert to continuous
  S = [disc.to_continuous(si) for si in Si]
  Ls = [cost(si) for si in Si]

  sim.show_results(path, S, Ls, animated=0.5)

  return Si
