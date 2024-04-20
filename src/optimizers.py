import torch as t
import torch.optim
import torch.nn as nn



class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr):
        self.lr = lr
        params = list(params) #generator
        self.params = params
        defaults= dict(lr = lr, params = self.params)
        super(SGD, self).__init__(params, defaults)

    @t.inference_mode()
    def step(self):
        for i,p in enumerate(self.params):
            grad = p.grad
            self.params[i] -= self.lr*grad


class SGDMomentum(torch.optim.Optimizer):
    def __init__(self, params, lr):
        self.lr = lr
        params = list(params) #generator
        self.params = params
        defaults= dict(lr = lr, params = self.params)
        super(SGDMomentum, self).__init__(params, defaults)

    @t.inference_mode()
    def step(self):
        for i,p in enumerate(self.params):
            grad = p.grad
            self.params[i] -= self.lr*grad

class Adam(torch.optim.Optimizer):
    def __init__(self, params, lr, b1, b2, ep):
        params = list(params) #generator
        self.params = params
        self.t = 0
        self.m = t.zeros(size =(len(params),)) #first moment vector
        self.v = t.zeros(size =(len(params),)) #second moment vector
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.ep = ep
        defaults = dict(params = self.params, lr = lr, b1 = b1, b2 = b2, ep = ep)
        super(Adam, self).__init__(params, defaults)

    @t.inference_mode()
    def step(self):
        self.t += 1
        for i,p in enumerate(self.params):
            grad = p.grad
            self.m[i] = self.b1*self.m[i] + (1-self.b1)*grad
            self.v[i] = self.b2*self.v[i] + (1-self.b2)*(grad**2)
            m_hat = self.m[i]/(1-self.b1**self.t)
            v_hat = self.v[i]/(1-self.b2**self.t)
            self.params[i] -= self.lr*grad*m_hat/(t.sqrt(v_hat)+self.ep)

class RMSProp(torch.optim.Optimizer):
    def __init__(self, params, lr):
        self.lr = lr
        params = list(params) #generator
        self.params = params
        defaults= dict(lr = lr, params = self.params)
        super(RMSProp, self).__init__(params, defaults)

    @t.inference_mode()
    def step(self):
        for i,p in enumerate(self.params):
            grad = p.grad
            self.params[i] -= self.lr*grad

class AdaGrad(torch.optim.Optimizer):
    def __init__(self, params, lr):
        self.lr = lr
        params = list(params) #generator
        self.params = params
        defaults= dict(lr = lr, params = self.params)
        super(AdaGrad, self).__init__(params, defaults)

    @t.inference_mode()
    def step(self):
        for i,p in enumerate(self.params):
            grad = p.grad
            self.params[i] -= self.lr*grad