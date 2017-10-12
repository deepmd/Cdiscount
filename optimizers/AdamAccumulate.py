# https://www.kaggle.com/c/carvana-image-masking-challenge/discussion/38125

from keras.optimizers import Optimizer
import keras.backend as K

class AdamAccumulate(Optimizer):
    """Adam optimizer.

    Default parameters follow those provided in the original paper.

    # Arguments
        lr: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor.
        decay: float >= 0. Learning rate decay over each update.

    # References
        - [Adam - A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980v8)
    """

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-8, decay=0., accum_iters=5., **kwargs):
        super(AdamAccumulate, self).__init__(**kwargs)
        self.iterations = K.variable(0, name='iterations')
        self.lr = K.variable(lr, name='lr')
        self.beta_1 = K.variable(beta_1, name='beta_1')
        self.beta_2 = K.variable(beta_2, name='beta_2')
        self.epsilon = epsilon
        self.decay = K.variable(decay, name='decay')
        self.initial_decay = decay
        self.accum_iters = K.variable(accum_iters, name='accum_iters')

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * self.iterations))

        t = self.iterations + 1
        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                     (1. - K.pow(self.beta_1, t)))

        shapes = [K.get_variable_shape(p) for p in params]
        ms = [K.zeros(shape) for shape in shapes]
        vs = [K.zeros(shape) for shape in shapes]
        gs = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + ms + vs

        for p, g, m, v, ga in zip(params, grads, ms, vs, gs):
            flag = K.equal(self.iterations % self.accum_iters, 0)
            flag = K.cast(flag, dtype='float32')
            if K.eval(flag): 
                flag = 1

            ga_t = (1 - flag) * (ga + g)
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * (ga + flag * g) / self.accum_iters
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square((ga + flag * g) / self.accum_iters)
            p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

            self.updates.append(K.update(m, flag * m_t + (1 - flag) * m))
            self.updates.append(K.update(v, flag * v_t + (1 - flag) * v))
            self.updates.append(K.update(ga, ga_t))

            new_p = p_t
            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)
            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'decay': float(K.get_value(self.decay)),
                  'accum_iters': float(K.get_value(self.accum_iters)),
                  'epsilon': self.epsilon}
        base_config = super(AdamAccumulate, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

