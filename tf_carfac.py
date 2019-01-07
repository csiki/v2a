from tensorflow.contrib import autograph
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class CARFAC:

    def __init__(self, stim_len, fs, batch_size, dtype, dtype_np):
        self.npoints = stim_len
        self.batch_size = batch_size

        self.nsec = 100  # number of sections in the cochlea between
        xlow = 0.10  # lowest frequency position along the cochlea and
        xhigh = 0.90  # highest frequency position along the cochlea

        # BM parameters
        x = np.array([np.linspace(xhigh, xlow, self.nsec, dtype=dtype_np) for _ in range(batch_size)])  # position along the cochlea 1 = base, 0 = apex
        f = 165.4 * (10 ** (2.1 * x) - 1)  # Greenwood for humans
        self.a0 = tf.constant(np.cos(2 * np.pi * f / fs), dtype=dtype)  # a0 and c0 control the poles and zeros
        self.c0 = tf.constant(np.sin(2 * np.pi * f / fs), dtype=dtype)

        damping = 0.2  # damping factor
        self.r = tf.constant(1 - damping * 2 * np.pi * f / fs, dtype=dtype)
        self.r1 = tf.constant(1 - damping * 2 * np.pi * f / fs, dtype=dtype)  # pole & zero radius minimum (set point)
        self.h = self.c0 + 0  # p302 h=c0 puts the zeros 1/2 octave above poles
        self.g = (1 - 2 * self.a0 * self.r1 + self.r1 * self.r1) / (1 - (2 * self.a0 - self.h * self.c0) * self.r1 + self.r1 * self.r1)  # p303 this gives 0dB DC gain for BM

        f_hpf = 20  # p328 20Hz corner for the BM HPF
        self.q = tf.constant(1 / (1 + (2 * np.pi * f_hpf / fs)), dtype=dtype)  # corresponding IIR coefficient

        tau_in = 10e-3  # p329 transmitter creation time constant
        self.c_in = 1. / (fs * tau_in)  # p329 corresponding IIR coefficient
        tau_out = 0.5e-3  # p329 transmitter depletion time constant
        self.c_out = 1. / (fs * tau_out)  # p329 corresponding IIR coefficient
        tau_IHC = 80e-6  # p329 ~8kHz LPF for IHC output
        self.c_IHC = 1. / (fs * tau_IHC)  # corresponding IIR coefficient

        # OHC parameters
        self.scale = 0.1  # p313 NLF parameter
        self.offset = 0.04  # p313 NLF parameter
        self.b = 1.0  # automatic gain loop feedback (1=no undamping).
        self.d_rz = 0.7 * (1 - self.r1)  # p310 relative undamping

        # AGC loop parameters
        self.tau_AGC = .002 * 4 ** np.arange(4)  # p336

        # The AGC filters are decimated, i.e., running at a lower sample rate
        self.c_AGC = tf.constant(8 * 2 ** np.arange(4) / (fs * self.tau_AGC), dtype=dtype)

        # spatial filtering
        shift_AGC = self.c_AGC * 0.65 * np.sqrt(2) ** np.arange(4)
        spread_sq_AGC = self.c_AGC * (1.65 ** 2 + 1) * 2 ** np.arange(4)
        self.sa = (spread_sq_AGC + shift_AGC ** 2 - shift_AGC) / 2
        self.sb = (spread_sq_AGC + shift_AGC ** 2 + shift_AGC) / 2
        self.sc = 1 - self.sa - self.sb

        # create lists of tensors to ease tf assignment and slicing
        self.W0 = CARFAC.create_variable_tensor(x.shape, [1], dtype=dtype)
        self.W1 = CARFAC.create_variable_tensor(x.shape, [1], dtype=dtype)
        self.W1old = CARFAC.create_variable_tensor(x.shape, dtype=dtype)
        self.BM = CARFAC.create_variable_tensor([batch_size, self.nsec, self.npoints], [1, 2], dtype=dtype)
        self.BM_hpf = CARFAC.create_variable_tensor([batch_size, self.nsec, self.npoints], [2], dtype=dtype)
        self.trans = CARFAC.create_variable_tensor(x.shape, [], tf.ones, dtype=dtype)
        self.IHC = CARFAC.create_variable_tensor([batch_size, self.nsec, self.npoints], [2], dtype=dtype)
        self.IHCa = CARFAC.create_variable_tensor([batch_size, self.nsec, self.npoints], [2], dtype=dtype)
        self.In8 = CARFAC.create_variable_tensor(x.shape, dtype=dtype)
        self.In16 = CARFAC.create_variable_tensor(x.shape, dtype=dtype)
        self.In32 = CARFAC.create_variable_tensor(x.shape, dtype=dtype)
        self.In64 = CARFAC.create_variable_tensor(x.shape, dtype=dtype)
        self.AGC = CARFAC.create_variable_tensor([batch_size, self.nsec, self.npoints], [2], dtype=dtype)
        self.AGC0 = CARFAC.create_variable_tensor(x.shape, dtype=dtype)
        self.AGC1 = CARFAC.create_variable_tensor(x.shape, dtype=dtype)
        self.AGC2 = CARFAC.create_variable_tensor(x.shape, dtype=dtype)
        self.AGC3 = CARFAC.create_variable_tensor(x.shape, dtype=dtype)

    def run(self, stimulus):
        # prepare stimulus
        split_stim = tf.split(stimulus, self.npoints, axis=1)
        for i in range(self.npoints - 1):  # all but the last one
            self.BM[-1][i] = tf.reshape(split_stim[i], [self.batch_size])

        # run through cochlea
        for t in range(self.npoints):
            self._iterate(t)

    def _iterate(self, t):
        for s in range(self.nsec):  # multiplex through the sections to calculate BM filters
            self.W0new = self.BM[s - 1][t] + self.r[:, s] * (self.a0[:, s] * self.W0[s] - self.c0[:, s] * self.W1[s])
            self.W1[s] = self.r[:, s] * (self.a0[:, s] * self.W1[s] + self.c0[:, s] * self.W0[s])
            self.W0[s] = self.W0new
            self.BM[s][t] = self.g[:, s] * (self.BM[s - 1][t] + self.h[:, s] * self.W1[s])
        # to speed up simulation, operate on all sections simultaneously for what follows
        self.BM_hpf[t] = self.q * (self.BM_hpf[t - 1] + CARFAC.BM_at_t(self.BM, t) - CARFAC.BM_at_t(self.BM, t - 1))  # high-pass filter
        z = tf.nn.relu(self.BM_hpf[t] + 0.175)  # nonlinear function for IHC
        v_mem = z ** 3 / (z ** 3 + z ** 2 + 0.1)  # nonlinear function for IHC
        IHC_new = v_mem * self.trans  # IHC output
        self.trans += self.c_in * (1 - self.trans) - self.c_out * IHC_new  # update amount of neuro transmitter
        self.IHCa[t] = (1 - self.c_IHC) * self.IHCa[t - 1] + self.c_IHC * IHC_new  # Low-pass filter once
        self.IHC[t] = (1 - self.c_IHC) * self.IHC[t - 1] + self.c_IHC * self.IHCa[t]  # Low-pass filter twice
        v_OHC = tf.stack(self.W1, axis=1) - self.W1old  # OHC potential
        # sqr = (v_OHC * scale + offset)**2  # removed, not used
        NLF = 1 / (1 + (self.scale * v_OHC + self.offset) ** 2)  # nonlinear function for OHC
        self.In8 += self.IHC[t] / 8.0  # accumulate input
        if t % 64 == 0:  # subsample AGC1 by factor 64
            self.AGC3 = (1 - self.c_AGC[3]) * self.AGC3 + self.c_AGC[3] * self.In64  # LPF in time domain
            self.AGC3 = self.sa[3] * tf.manip.roll(self.AGC3, 1, -1) + self.sc[3] * self.AGC3 + self.sb[3] * tf.manip.roll(self.AGC3, -1, -1)  # LPF in spatial domain
            self.In64 *= 0  # reset input accumulator
        if t % 32 == 0:  # subsample AGC2 by factor 32
            self.AGC2 = (1 - self.c_AGC[2]) * self.AGC2 + self.c_AGC[2] * (self.In32 + 2 * self.AGC3)
            self.AGC2 = self.sa[2] * tf.manip.roll(self.AGC2, 1, -1) + self.sc[2] * self.AGC2 + self.sb[2] * tf.manip.roll(self.AGC2, -1, -1)
            self.In64 += self.In32
            self.In32 *= 0
        if t % 16 == 0:  # subsample ACG3 by factor 16
            self.AGC1 = (1 - self.c_AGC[1]) * self.AGC1 + self.c_AGC[1] * (self.In16 + 2 * self.AGC2)
            self.AGC1 = self.sa[1] * tf.manip.roll(self.AGC1, 1, -1) + self.sc[1] * self.AGC1 + self.sb[1] * tf.manip.roll(self.AGC1, -1, -1)
            self.In32 += self.In16
            self.In16 *= 0
        if t % 8 == 0:
            self.AGC0 = (1 - self.c_AGC[0]) * self.AGC0 + self.c_AGC[0] * (self.In8 + 2 * self.AGC1)
            self.AGC0 = self.sa[0] * tf.manip.roll(self.AGC0, 1, -1) + self.sc[0] * self.AGC0 + self.sb[0] * tf.manip.roll(self.AGC0, -1, -1)
            self.AGC[t] = self.AGC0  # store AGC output for plotting
            self.r = self.r1 + self.d_rz * (1 - self.AGC0) * NLF  # feedback to BM
            self.g = (1 - 2 * self.a0 * self.r + self.r * self.r) / (1 - (2 * self.a0 - self.h * self.c0) * self.r + self.r * self.r)  # gain for BM
            self.In16 += self.In8
            self.In8 *= 0
        else:
            self.AGC[t] = self.AGC[t - 1]

    def output(self):
        aaaa = tf.stack(self.IHC)
        return tf.transpose(aaaa, [1, 2, 0])

    @staticmethod
    def BM_at_t(BM, t):
        vals = []
        for s in range(len(BM)):
            vals.append(BM[s][t])
        return tf.transpose(tf.stack(vals), [1, 0])  # batch first

    @staticmethod
    # how variables are overwritten:
    # W0new, W1[s], W0[s], BM[s,t], BM_hpf[:,t], IHCa[:,t], IHC[:,t], AGC[:,t]
    def create_variable_tensor(shape, ow_dims=[], alloc_fun=tf.zeros, dtype=tf.float32):
        # e.g.: [64, 83, 1000], [1, 2], tf.zeros, tf.float32 --> 83x1000 sized list, w/ alloc_fun([64], dtype=dtype) elements
        if len(ow_dims) == 0:
            return alloc_fun(shape, dtype=dtype)

        list_dims = [shape[d] for d in ow_dims]
        tens_dims = [d for i, d in enumerate(shape) if i not in ow_dims]
        vartens = [0] * list_dims[0]

        if len(list_dims) == 1:
            for vi in range(len(vartens)):
                vartens[vi] = alloc_fun(tens_dims, dtype=dtype)
        elif len(list_dims) == 2:
            ld = list_dims[1]
            for vi in range(len(vartens)):
                vartens[vi] = [0] * ld
                for vi2 in range(ld):
                    vartens[vi][vi2] = alloc_fun(tens_dims, dtype=dtype)
        else:
            raise NotImplementedError

        return vartens


if __name__ == '__main__':

    # constants
    fs = 16000
    batch_size = 2
    sound_len = 8
    dtype = tf.float32
    dtype_np = np.float32

    # create input tone
    f0 = 700  # tone frequency
    t1 = np.arange(sound_len) / fs  # sample times
    gain = 0.1  # input gain
    stimulus = gain * np.sin(2 * np.pi * f0 * t1)
    stimulus = np.tile(stimulus, [batch_size, 1])
    stimulus = tf.constant(stimulus, dtype)

    # build and run model
    carfac = CARFAC(sound_len, fs, batch_size, dtype, dtype_np)
    carfac.run(stimulus)
    IHC = carfac.output()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        stimulus_, IHC_ = sess.run([stimulus, IHC])

    plt.figure(3, figsize=(12, 3)) # IHC output
    plt.plot(t1*1000, stimulus_[0], 'r')
    plt.plot(t1*1000, IHC_[0].T)
    plt.xlabel('t (ms)')
    plt.title('IHC response')
    plt.tight_layout()
    plt.show()
