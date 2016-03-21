'''
FIN2 : a module created to
(1) help John Regan learn FIN2 concepts by coding them, and
(2) to allow him to do computations more easily because he can't use Excel fast
enough to have enough time to figure out the answers.
'''

import sys
from math import log, exp
try:
    from scipy.stats import norm
except ImportError:
    print 'FIN2 requires scipy to work properly'
    sys.exit()

# WARNING: All numbers should be floats -> x = 1.0

def impliedVolatility(className, args, callPrice=None, putPrice=None, high=500.0, low=0.0):
    '''Returns the estimated implied volatility'''
    if callPrice:
        target = callPrice
    if putPrice:
        target = putPrice

    decimals = len(str(target).split('.')[1])	# Count decimals
    for i in xrange(10000):	# To avoid infinite loops
        mid = (high + low) / 2.0
        if mid < 0.00001:
            mid = 0.00001
        if callPrice:
            estimate = eval(className)(args, volatility=mid, performance=True).callPrice
        if putPrice:
            estimate = eval(className)(args, volatility=mid, performance=True).putPrice
        if round(estimate, decimals) == target:
            break
        elif estimate > target:
            high = mid
        elif estimate < target:
            low = mid
    return mid

class GK:
    '''Garman-Kohlhagen
    Used for pricing European options on currencies

    GK([stockPrice, strikePrice, domesticRate, foreignRate, \
        yearsToExpiration], volatility=x, callPrice=y, putPrice=z)

    eg:
        c = FIN2.GK([1.4565, 1.45, .01, .02, .3], volatility=.20)
        c.callPrice				# Returns the call price
        c.putPrice				# Returns the put price
        c.callDelta				# Returns the call delta
        c.putDelta				# Returns the put delta
        c.callDelta2			# Returns the call dual delta
        c.putDelta2				# Returns the put dual delta
        c.callTheta				# Returns the call theta
        c.putTheta				# Returns the put theta
        c.callRhoD				# Returns the call domestic rho
        c.putRhoD				# Returns the put domestic rho
        c.callRhoF				# Returns the call foreign rho
        c.putRhoF				# Returns the call foreign rho
        c.vega					# Returns the option vega
        c.gamma					# Returns the option gamma

        c = FIN2.GK([1.4565, 1.45, .01, .02, .3], callPrice=0.0359)
        c.impliedVolatility		# Returns the implied volatility from the call price

        c = FIN2.GK([1.4565, 1.45, .01, .02, .3], putPrice=0.03)
        c.impliedVolatility		# Returns the implied volatility from the put price

        c = FIN2.GK([1.4565, 1.45, .1, .2, .3], callPrice=0.0359, putPrice=0.03)
        c.putCallParity			# Returns the put-call parity
    '''

    def __init__(self, args, volatility=None, callPrice=None, putPrice=None, performance=None):
        self.stockPrice = float(args[0])
        self.strikePrice = float(args[1])
        self.domesticRate = float(args[2])
        self.foreignRate = float(args[3])
        self.yearsToExpiration = float(args[4])

        # initialize properties to None
        for i in ['callPrice', 'putPrice', 'callDelta', 'putDelta', \
                  'callDelta2', 'putDelta2', 'callTheta', 'putTheta', \
                  'callRhoD', 'putRhoD', 'callRhoF', 'callRhoF', 'vega', \
                  'gamma', 'impliedVolatility', 'putCallParity']:
            self.__dict__[i] = None

        if volatility:
            self.volatility = float(volatility)

            self._a_ = self.volatility * self.yearsToExpiration**0.5
            self._d1_ = (log(self.stockPrice / self.strikePrice) + \
                        (self.domesticRate - self.foreignRate + \
                        (self.volatility**2)/2.0) * self.yearsToExpiration) / self._a_
            self._d2_ = self._d1_ - self._a_
            # Reduces performance overhead when computing implied volatility
            if performance:
                (self.callPrice, self.putPrice) = self._price()
            else:
                (self.callPrice, self.putPrice) = self._price()
                (self.callDelta, self.putDelta) = self._delta()
                (self.callDelta2, self.putDelta2) = self._delta2()
                (self.callTheta, self.putTheta) = self._theta()
                (self.callRhoD, self.putRhoD) = self._rhod()
                (self.callRhoF, self.putRhoF) = self._rhof()
                self.vega = self._vega()
                self.gamma = self._gamma()
                self.exerciceProbability = norm.cdf(self._d2_)

        if callPrice:
            self.callPrice = round(float(callPrice), 6)
            self.impliedVolatility = impliedVolatility(self.__class__.__name__, args, callPrice=self.callPrice)

        if putPrice and not callPrice:
            self.putPrice = round(float(putPrice), 6)
            self.impliedVolatility = impliedVolatility(self.__class__.__name__, args, putPrice=self.putPrice)

        if callPrice and putPrice:
            self.callPrice = float(callPrice)
            self.putPrice = float(putPrice)
            self.putCallParity = self._parity()

    def _price(self):
        '''Returns the option price: (Call price, Put price)'''
        if self.volatility == 0 or self.yearsToExpiration == 0:
            call = max(0.0, self.stockPrice - self.strikePrice)
            put  = max(0.0, self.strikePrice - self.stockPrice)
        if self.strikePrice == 0:
            raise ZeroDivisionError('The strike price cannot be zero')
        else:
            # call price
            call = exp(-self.foreignRate * self.yearsToExpiration) * \
                    self.stockPrice * norm.cdf(self._d1_) - \
                    exp(-self.domesticRate * self.yearsToExpiration) * \
                    self.strikePrice * norm.cdf(self._d2_)
            # put price
            put = exp(-self.domesticRate * self.yearsToExpiration) * \
                    self.strikePrice * norm.cdf(-self._d2_) - \
                    exp(-self.foreignRate * self.yearsToExpiration) * \
                    self.stockPrice * norm.cdf(-self._d1_)
        return (call, put)

    def _delta(self):
        '''Returns the option delta: (Call delta, Put delta)'''
        if self.volatility == 0 or self.yearsToExpiration == 0:
            call = 1.0 if self.stockPrice > self.strikePrice else 0.0
            put = -1.0 if self.stockPrice < self.strikePrice else 0.0
        if self.strikePrice == 0:
            raise ZeroDivisionError('The strike price cannot be zero')
        else:
            _b_ = exp(-self.foreignRate * self.yearsToExpiration)
            call = norm.cdf(self._d1_) * _b_
            put = -norm.cdf(-self._d1_) * _b_
        return (call, put)

    def _delta2(self):
        '''Returns the dual delta: (Call dual delta, Put dual delta)'''
        if self.volatility == 0 or self.yearsToExpiration == 0:
            call = -1.0 if self.stockPrice > self.strikePrice else 0.0
            put  =  1.0 if self.stockPrice < self.strikePrice else 0.0
        if self.strikePrice == 0:
            raise ZeroDivisionError('The strike price cannot be zero')
        else:
            _b_ = exp(-self.domesticRate * self.yearsToExpiration)
            call = -norm.cdf(self._d2_) * _b_
            put  = norm.cdf(-self._d2_) * _b_
        return (call, put)

    def _vega(self):
        '''Returns the option vega'''
        if self.volatility == 0 or self.yearsToExpiration == 0:
            return 0.0
        if self.strikePrice == 0:
            raise ZeroDivisionError('The strike price cannot be zero')
        else:
            return self.stockPrice * exp(-self.foreignRate * \
                    self.yearsToExpiration) * norm.pdf(self._d1_) * \
                    self.yearsToExpiration**0.5

    def _theta(self):
        '''Returns the option theta: (Call theta, Put theta)'''
        _b_ = exp(-self.foreignRate * self.yearsToExpiration)
        call = -self.stockPrice * _b_ * norm.pdf(self._d1_) * \
                self.volatility / (2 * self.yearsToExpiration**0.5) + \
                self.foreignRate * self.stockPrice * _b_ * \
                norm.cdf(self._d1_) - self.domesticRate * self.strikePrice * \
                _b_ * norm.cdf(self._d2_)
        put = -self.stockPrice * _b_ * norm.pdf(self._d1_) * \
                self.volatility / (2 * self.yearsToExpiration**0.5) - \
                self.foreignRate * self.stockPrice * _b_ * \
                norm.cdf(-self._d1_) + self.domesticRate * self.strikePrice * \
                _b_ * norm.cdf(-self._d2_)
        return (call / 365.0, put / 365.0)

    def _rhod(self):
        '''Returns the option domestic rho: (Call rho, Put rho)'''
        call = self.strikePrice * self.yearsToExpiration * \
                exp(-self.domesticRate * self.yearsToExpiration) * \
                norm.cdf(self._d2_) / 100.0
        put = -self.strikePrice * self.yearsToExpiration * \
                exp(-self.domesticRate * self.yearsToExpiration) * \
                norm.cdf(-self._d2_) / 100.0
        return (call, put)

    def _rhof(self):
        '''Returns the option foreign rho: (Call rho, Put rho)'''
        call = -self.stockPrice * self.yearsToExpiration * \
                exp(-self.foreignRate * self.yearsToExpiration) * \
                norm.cdf(self._d1_) / 100.0
        put = self.stockPrice * self.yearsToExpiration * \
                exp(-self.foreignRate * self.yearsToExpiration) * \
                norm.cdf(-self._d1_) / 100.0
        return (call, put)

    def _gamma(self):
        '''Returns the option gamma'''
        return (norm.pdf(self._d1_) * exp(-self.foreignRate * \
                self.yearsToExpiration)) / (self.stockPrice * self._a_)

    def _parity(self):
        '''Returns the put-call parity'''
        return self.callPrice - self.putPrice - (self.stockPrice / \
                ((1 + self.foreignRate)**self.yearsToExpiration)) + \
                (self.strikePrice / ((1 + self.domesticRate)**self.yearsToExpiration))

class BS:
    '''Black-Scholes
    Used for pricing European options on stocks without dividends

    BS([stockPrice, strikePrice, interestRate, yearsToExpiration], \
            volatility=x, callPrice=y, putPrice=z)

    eg:
        c = FIN2.BS([1.4565, 1.45, .01, .30], volatility=.20)
        c.callPrice				# Returns the call price
        c.putPrice				# Returns the put price
        c.callDelta				# Returns the call delta
        c.putDelta				# Returns the put delta
        c.callDelta2			# Returns the call dual delta
        c.putDelta2				# Returns the put dual delta
        c.callTheta				# Returns the call theta
        c.putTheta				# Returns the put theta
        c.callRho				# Returns the call rho
        c.putRho				# Returns the put rho
        c.vega					# Returns the option vega
        c.gamma					# Returns the option gamma

        c = FIN2.BS([1.4565, 1.45, .01, .30], callPrice=0.0359)
        c.impliedVolatility		# Returns the implied volatility from the call price

        c = FIN2.BS([1.4565, 1.45, .01, .30], putPrice=0.0306)
        c.impliedVolatility		# Returns the implied volatility from the put price

        c = FIN2.BS([1.4565, 1.45, .01, .30], callPrice=0.0359, putPrice=0.0306)
        c.putCallParity			# Returns the put-call parity
        '''

    def __init__(self, args, volatility=None, callPrice=None, putPrice=None, \
            performance=None):
        self.stockPrice = float(args[0])
        self.strikePrice = float(args[1])
        self.interestRate = float(args[2])
        self.yearsToExpiration = float(args[3])

        for i in ['callPrice', 'putPrice', 'callDelta', 'putDelta', \
                'callDelta2', 'putDelta2', 'callTheta', 'putTheta', \
                'callRho', 'putRho', 'vega', 'gamma', 'impliedVolatility', \
                'putCallParity']:
            self.__dict__[i] = None

        if volatility:
            self.volatility = float(volatility)

            self._a_ = self.volatility * self.yearsToExpiration**0.5
            self._d1_ = (log(self.stockPrice / self.strikePrice) + \
                    (self.interestRate + (self.volatility**2) / 2.0) * \
                    self.yearsToExpiration) / self._a_
            self._d2_ = self._d1_ - self._a_
            if performance:
                (self.callPrice, self.putPrice) = self._price()
            else:
                (self.callPrice, self.putPrice) = self._price()
                (self.callDelta, self.putDelta) = self._delta()
                (self.callDelta2, self.putDelta2) = self._delta2()
                (self.callTheta, self.putTheta) = self._theta()
                (self.callRho, self.putRho) = self._rho()
                self.vega = self._vega()
                self.gamma = self._gamma()
                self.exerciceProbability = norm.cdf(self._d2_)
        if callPrice:
            self.callPrice = round(float(callPrice), 6)
            self.impliedVolatility = impliedVolatility(\
                    self.__class__.__name__, args, callPrice=self.callPrice)
        if putPrice and not callPrice:
            self.putPrice = round(float(putPrice), 6)
            self.impliedVolatility = impliedVolatility(\
                    self.__class__.__name__, args, putPrice=self.putPrice)
        if callPrice and putPrice:
            self.callPrice = float(callPrice)
            self.putPrice = float(putPrice)
            self.putCallParity = self._parity()

    def _price(self):
        '''Returns the option price: (Call price, Put price)'''
        if self.volatility == 0 or self.yearsToExpiration == 0:
            call = max(0.0, self.stockPrice - self.strikePrice)
            put  = max(0.0, self.strikePrice - self.stockPrice)
        if self.strikePrice == 0:
            raise ZeroDivisionError('The strike price cannot be zero')
        else:
            call = self.stockPrice * norm.cdf(self._d1_) - \
                    self.strikePrice * exp(-self.interestRate * \
                    self.yearsToExpiration) * norm.cdf(self._d2_)
            put = self.strikePrice * exp(-self.interestRate * \
                    self.yearsToExpiration) * norm.cdf(-self._d2_) - \
                    self.stockPrice * norm.cdf(-self._d1_)
        return (call, put)

    def _delta(self):
        '''Returns the option delta: (Call delta, Put delta)'''
        if self.volatility == 0 or self.yearsToExpiration == 0:
            call = 1.0 if self.stockPrice > self.strikePrice else 0.0
            put = -1.0 if self.stockPrice < self.strikePrice else 0.0
        if self.strikePrice == 0:
            raise ZeroDivisionError('The strike price cannot be zero')
        else:
            call = norm.cdf(self._d1_)
            put = -norm.cdf(-self._d1_)
        return (call, put)

    def _delta2(self):
        '''Returns the dual delta: (Call dual delta, Put dual delta)'''
        if self.volatility == 0 or self.yearsToExpiration == 0:
            call = -1.0 if self.stockPrice > self.strikePrice else 0.0
            put  =  1.0 if self.stockPrice < self.strikePrice else 0.0
        if self.strikePrice == 0:
            raise ZeroDivisionError('The strike price cannot be zero')
        else:
            _b_ = exp(-self.interestRate * self.yearsToExpiration)
            call = -norm.cdf(self._d2_) * _b_
            put  = norm.cdf(-self._d2_) * _b_
        return [call, put]

    def _vega(self):
        '''Returns the option vega'''
        if self.volatility == 0 or self.yearsToExpiration == 0:
            return 0.0
        if self.strikePrice == 0:
            raise ZeroDivisionError('The strike price cannot be zero')
        else:
            return self.stockPrice * norm.pdf(self._d1_) * \
                    self.yearsToExpiration**0.5 / 100.0

    def _theta(self):
        '''Returns the option theta: (Call theta, Put theta)'''
        _b_ = exp(-self.interestRate * self.yearsToExpiration)
        call = -self.stockPrice * norm.pdf(self._d1_) * self.volatility / \
                (2 * self.yearsToExpiration**0.5) - self.interestRate * \
                self.strikePrice * _b_ * norm.cdf(self._d2_)
        put = -self.stockPrice * norm.pdf(self._d1_) * self.volatility / \
                (2 * self.yearsToExpiration**0.5) + self.interestRate * \
                self.strikePrice * _b_ * norm.cdf(-self._d2_)
        return (call / 365.0, put / 365.0)

    def _rho(self):
        '''Returns the option rho: (Call rho, Put rho)'''
        _b_ = exp(-self.interestRate * self.yearsToExpiration)
        call = self.strikePrice * self.yearsToExpiration * _b_ * \
                norm.cdf(self._d2_) / 100.0
        put = -self.strikePrice * self.yearsToExpiration * _b_ * \
                norm.cdf(-self._d2_) / 100.0
        return (call, put)

    def _gamma(self):
        '''Returns the option gamma'''
        return norm.pdf(self._d1_) / (self.stockPrice * self._a_)

    def _parity(self):
        '''Put-Call Parity'''
        return self.callPrice - self.putPrice - self.stockPrice + \
                (self.strikePrice / \
                ((1 + self.interestRate)**self.yearsToExpiration))

class Me:
    '''Merton
    Used for pricing European options on stocks with dividends

    Me([stockPrice, strikePrice, interestRate, annualDividends, \
            yearsToExpiration], volatility=x, callPrice=y, putPrice=z)

    eg:
        c = FIN2.Me([52, 50, 1, 1, 30], volatility=20)
        c.callPrice				# Returns the call price
        c.putPrice				# Returns the put price
        c.callDelta				# Returns the call delta
        c.putDelta				# Returns the put delta
        c.callDelta2			# Returns the call dual delta
        c.putDelta2				# Returns the put dual delta
        c.callTheta				# Returns the call theta
        c.putTheta				# Returns the put theta
        c.callRho				# Returns the call rho
        c.putRho				# Returns the put rho
        c.vega					# Returns the option vega
        c.gamma					# Returns the option gamma

        c = FIN2.Me([52, 50, 1, 1, 30], callPrice=0.0359)
        c.impliedVolatility		# Returns the implied volatility from the call price

        c = FIN2.Me([52, 50, 1, 1, 30], putPrice=0.0306)
        c.impliedVolatility		# Returns the implied volatility from the put price

        c = FIN2.Me([52, 50, 1, 1, 30], callPrice=0.0359, putPrice=0.0306)
        c.putCallParity			# Returns the put-call parity
    '''

    def __init__(self, args, volatility=None, callPrice=None, putPrice=None, performance=None):
        self.stockPrice = float(args[0])
        self.strikePrice = float(args[1])
        self.interestRate = float(args[2])
        self.dividend = float(args[3])
        self.dividendYield = self.dividend / self.stockPrice
        self.yearsToExpiration = float(args[4])

        for i in ['callPrice', 'putPrice', 'callDelta', 'putDelta', \
                'callDelta2', 'putDelta2', 'callTheta', 'putTheta', \
                'callRho', 'putRho', 'vega', 'gamma', 'impliedVolatility', \
                'putCallParity']:
            self.__dict__[i] = None

        if volatility:
            self.volatility = float(volatility)

            self._a_ = self.volatility * self.yearsToExpiration**0.5
            self._d1_ = (log(self.stockPrice / self.strikePrice) + \
                    (self.interestRate - self.dividendYield + \
                    (self.volatility**2) / 2.0) * self.yearsToExpiration) / \
                    self._a_
            self._d2_ = self._d1_ - self._a_
            if performance:
                (self.callPrice, self.putPrice) = self._price()
            else:
                (self.callPrice, self.putPrice) = self._price()
                (self.callDelta, self.putDelta) = self._delta()
                (self.callDelta2, self.putDelta2) = self._delta2()
                (self.callTheta, self.putTheta) = self._theta()
                (self.callRho, self.putRho) = self._rho()
                self.vega = self._vega()
                self.gamma = self._gamma()
                self.exerciceProbability = norm.cdf(self._d2_)
        if callPrice:
            self.callPrice = round(float(callPrice), 6)
            self.impliedVolatility = impliedVolatility(\
                    self.__class__.__name__, args, self.callPrice)
        if putPrice and not callPrice:
            self.putPrice = round(float(putPrice), 6)
            self.impliedVolatility = impliedVolatility(\
                    self.__class__.__name__, args, putPrice=self.putPrice)
        if callPrice and putPrice:
            self.callPrice = float(callPrice)
            self.putPrice = float(putPrice)
            self.putCallParity = self._parity()

    def _price(self):
        '''Returns the option price: (Call price, Put price)'''
        if self.volatility == 0 or self.yearsToExpiration == 0:
            call = max(0.0, self.stockPrice - self.strikePrice)
            put  = max(0.0, self.strikePrice - self.stockPrice)
        if self.strikePrice == 0:
            raise ZeroDivisionError('The strike price cannot be zero')
        else:
            call = self.stockPrice * exp(-self.dividendYield * \
                    self.yearsToExpiration) * norm.cdf(self._d1_) - \
                    self.strikePrice * exp(-self.interestRate * \
                    self.yearsToExpiration) * norm.cdf(self._d2_)
            put = self.strikePrice * exp(-self.interestRate * \
                    self.yearsToExpiration) * norm.cdf(-self._d2_) - \
                    self.stockPrice * exp(-self.dividendYield * \
                    self.yearsToExpiration) * norm.cdf(-self._d1_)
        return (call, put)

    def _delta(self):
        '''Returns the option delta: (Call delta, Put delta)'''
        if self.volatility == 0 or self.yearsToExpiration == 0:
            call = 1.0 if self.stockPrice > self.strikePrice else 0.0
            put = -1.0 if self.stockPrice < self.strikePrice else 0.0
        if self.strikePrice == 0:
            raise ZeroDivisionError('The strike price cannot be zero')
        else:
            _b_ = exp(-self.dividendYield * self.yearsToExpiration)
            call = _b_ * norm.cdf(self._d1_)
            put = _b_ *	(norm.cdf(self._d1_) - 1)
        return (call, put)

    # Verify
    def _delta2(self):
        '''Returns the dual delta: (Call dual delta, Put dual delta)'''
        if self.volatility == 0 or self.yearsToExpiration == 0:
            call = -1.0 if self.stockPrice > self.strikePrice else 0.0
            put  =  1.0 if self.stockPrice < self.strikePrice else 0.0
        if self.strikePrice == 0:
            raise ZeroDivisionError('The strike price cannot be zero')
        else:
            _b_ = exp(-self.interestRate * self.yearsToExpiration)
            call = -norm.cdf(self._d2_) * _b_
            put  = norm.cdf(-self._d2_) * _b_
        return (call, put)

    def _vega(self):
        '''Returns the option vega'''
        if self.volatility == 0 or self.yearsToExpiration == 0:
            return 0.0
        if self.strikePrice == 0:
            raise ZeroDivisionError('The strike price cannot be zero')
        else:
            return self.stockPrice * exp(-self.dividendYield * \
                    self.yearsToExpiration) * norm.pdf(self._d1_) * \
                    self.yearsToExpiration**0.5 / 100.0

    def _theta(self):
        '''Returns the option theta: (Call theta, Put theta)'''
        _b_ = exp(-self.interestRate * self.yearsToExpiration)
        _d_ = exp(-self.dividendYield * self.yearsToExpiration)
        call = -self.stockPrice * _d_ * norm.pdf(self._d1_) * \
                self.volatility / (2 * self.yearsToExpiration**0.5) + \
                self.dividendYield * self.stockPrice * _d_ * \
                norm.cdf(self._d1_) - self.interestRate * \
                self.strikePrice * _b_ * norm.cdf(self._d2_)
        put = -self.stockPrice * _d_ * norm.pdf(self._d1_) * \
                self.volatility / (2 * self.yearsToExpiration**0.5) - \
                self.dividendYield * self.stockPrice * _d_ * \
                norm.cdf(-self._d1_) + self.interestRate * \
                self.strikePrice * _b_ * norm.cdf(-self._d2_)
        return (call / 365.0, put / 365.0)

    def _rho(self):
        '''Returns the option rho: (Call rho, Put rho)'''
        _b_ = exp(-self.interestRate * self.yearsToExpiration)
        call = self.strikePrice * self.yearsToExpiration * _b_ * \
                norm.cdf(self._d2_) / 100.0
        put = -self.strikePrice * self.yearsToExpiration * _b_ * \
                norm.cdf(-self._d2_) / 100.0
        return (call, put)

    def _gamma(self):
        '''Returns the option gamma'''
        return exp(-self.dividendYield * self.yearsToExpiration) * \
                norm.pdf(self._d1_) / (self.stockPrice * self._a_)

    # Verify
    def _parity(self):
        '''Put-Call Parity'''
        return self.callPrice - self.putPrice - self.stockPrice + \
                (self.strikePrice / \
                ((1 + self.interestRate)**self.yearsToExpiration))


""" Get bond price from YTM """
def bond_price(par, T, ytm, coup, freq=2):
    freq = float(freq)
    periods = T * freq
    coupon = coup * par / freq
    dt = [(i+1)/freq for i in xrange(int(periods))]
    price = sum([coupon/(1+ytm/freq)**(freq*t) for t in dt]) + par/(1+ytm/freq)**(freq*T)
    return price

class Firm:
    def __init__(self, debt, equity, beta_debt, beta_equity, cost_of_debt, corpTax=0.35):
        self.D = float(debt)
        self.E = float(equity)
        self.beta_D = float(beta_debt)
        self.beta_E = float(beta_equity)
        self.R_d = float(cost_of_debt)
        self.corpTax = float(corpTax)

        self.V = self.D + self.E
        self.beta_A = (self.D/self.V)*self.beta_D + (self.E/self.V)*self.beta_E
        self.lev_beta_E = self.beta_A + (self.D/self.E)*(self.beta_A - self.beta_D)

    def Re(self, Rf, MRP):
        '''Cost of Equity (Re)'''
        return Rf + self.lev_beta_E * MRP

    def WACC(self, Rf, MRP):
        Re = Rf + self.lev_beta_E * MRP
        return (self.D/self.V)*self.R_d*(1-self.corpTax) + (self.E/self.V)*Re



