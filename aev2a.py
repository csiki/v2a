import tensorflow as tf
import numpy as np
from utils import *
from glob import glob
import os, sys
import time
import tables
import audio_gen
import hearing
import tf_carfac
import tcn
import scipy.io.wavfile as wavfile
from pprint import pprint


class Draw:

    def __init__(self, nepoch, img_h, img_w, num_colors, grayscale, network_params,
                 logging=True, log_after=100, save_after=1000, training=True):
        self.nepoch = nepoch
        self.img_h = img_h
        self.img_w = img_w
        self.num_colors = num_colors
        self.grayscale = grayscale
        self.logging = logging
        self.log_after = log_after
        self.save_after = save_after
        self.training = training

        self.attention_n = network_params['attention_n']
        self.n_hidden = network_params['n_hidden']
        self.n_z = network_params['n_z']
        self.sequence_length = network_params['sequence_length']
        self.batch_size = network_params['batch_size']
        self.n_rnn_cells = network_params['n_rnn_cells']
        self.initial_lr = network_params['learning_rate']
        self.nonrecurrent_dec = network_params['nonrecurrent_dec']
        self.hearing_decoder = network_params['hearing_decoder']
        self.v1_gaussian = network_params['v1_gaussian']
        self.n_v1_write = network_params['n_v1_write'] if self.v1_gaussian else 1
        self.dtype = network_params['dtype']
        self.npdtype = network_params['npdtype']
        self.fs = network_params['fs']
        self.kl_weight = network_params['kl_weight']
        self.congruence_weight = network_params['congr_weight']
        self.share_parameters = False

        # misc
        available_gpus = get_available_gpus()
        audio_params = network_params['audio_gen']
        hearing_params = network_params['hearing']
        self.min_img_dim = min([self.img_h, self.img_w])
        self.v1_wr = tf.ones([self.batch_size, 1, 1, self.attention_n])  # makes the model draw only white pixels
        hearing_gpus = [available_gpus[0], available_gpus[0]]  # default, if only 1 gpu is available
        hearing_gpus[:] = [available_gpus[1]] * len(hearing_gpus) if len(available_gpus) > 1 else hearing_gpus
        hearing_gpus[1] = available_gpus[2] if len(available_gpus) > 2 else hearing_gpus[1]

        # model name is crazy long but easy to search in tensorboard
        self.model_name_format = 'img={}x{}x{},attention={},hidden={},z={},seq_len={},n_rnn={}-{},v1={},nv1write={},cw={},fs={},hearing={},sslen={}x{}*{}*{},constphase={},mfccs={}-{}-{},wg={}-{}-{}_showoff'
        self.model_name = self.model_name_format\
            .format(self.img_h, self.img_w, self.num_colors, self.attention_n, self.n_hidden, self.n_z, self.sequence_length,
                    self.n_rnn_cells[0], self.n_rnn_cells[1], self.v1_gaussian, self.n_v1_write, self.congruence_weight, self.fs, self.hearing_decoder,
                    audio_params['nsoundstream'], audio_params['nmodulation'], audio_params['section_len_msec'],
                    audio_params['soundscape_len_by_stream_len'], audio_params['const_phase'],
                    hearing_params['mfcss_nceps'], hearing_params['mfcss_frame_len'], hearing_params['mfcss_frame_step'],
                    hearing_params['wg_nfilters'], hearing_params['wg_kernel_len'], hearing_params['wg_strides'])

        if self.grayscale:
            self.images = tf.placeholder(self.dtype, [None, self.img_h, self.img_w])
        else:
            self.images = tf.placeholder(self.dtype, [None, self.img_h, self.img_w, self.num_colors])

        # audio gen init stuff
        self.const_phase_tens = None
        if audio_params['const_phase']:
            self.const_phase_tens = tf.constant(audio_gen.constant_phase(audio_params['nsoundstream'], self.batch_size),
                                                dtype=self.dtype,
                                                shape=[self.batch_size, audio_params['nsoundstream'], 1])
        if not network_params['z_indirection']:  # force the network to have as many hidden states as audio_gen params
            self.n_z = audio_gen.nparams_needed(audio_params['nsoundstream'], audio_params['nmodulation'],
                                                audio_params['varying_delta'], audio_params['const_phase'])

        # source of Gaussian randomness
        self.e = tf.random_normal([self.batch_size, self.n_z], mean=0., stddev=1.)  # Qsampler noise
        self.azim_e = tf.random_normal([self.batch_size, audio_params['nsoundstream'], audio_params['nmodulation']], mean=0., stddev=1.)

        # encoder/decoder RNN cells
        if network_params['residual_encoder']:  # encoder does not necessarily have to be residual
            self.enc_cells = [
                tf.contrib.rnn.ResidualWrapper(tf.contrib.rnn.LayerNormBasicLSTMCell(self.n_hidden, activation=tf.nn.tanh))
                for _ in range(self.n_rnn_cells[0] - 1)]
            # non-residual layer needed at the front as the input-output dimension is not the same for the first enc and dec layers
            self.enc_cells.insert(0, tf.contrib.rnn.LayerNormBasicLSTMCell(self.n_hidden, activation=tf.nn.tanh))
        else:
            self.enc_cells = [tf.contrib.rnn.LayerNormBasicLSTMCell(self.n_hidden, activation=tf.nn.tanh)
                              for _ in range(self.n_rnn_cells[0])]
        self.rnn_enc = tf.contrib.rnn.MultiRNNCell(self.enc_cells)  # encoder

        # decoder part
        if not self.nonrecurrent_dec:
            self.dec_cells = [
                tf.contrib.rnn.ResidualWrapper(tf.contrib.rnn.LayerNormBasicLSTMCell(self.n_hidden, activation=tf.nn.tanh))
                for _ in range(self.n_rnn_cells[1] - 1)]
            # put a non-residual layer at the front as the input-output dimension is not the same for the first enc and dec layers
            self.dec_cells.insert(0, tf.contrib.rnn.LayerNormBasicLSTMCell(self.n_hidden, activation=tf.nn.tanh))
            self.rnn_dec = tf.contrib.rnn.MultiRNNCell(self.dec_cells)  # decoder
            dec_state = self.rnn_dec.zero_state(self.batch_size, self.dtype)
        else:  # non-recurrent decoder
            self.rnn_dec = []
            for i in range(self.n_rnn_cells[1]):
                self.rnn_dec.append(tf.layers.Dense(self.n_hidden, activation=tf.nn.tanh))

        self.cs = [0] * self.sequence_length
        self.mu, self.logsigma, self.sigma = [0] * self.sequence_length, [0] * self.sequence_length, [0] * self.sequence_length

        h_dec_prev = tf.zeros((self.batch_size, self.n_hidden))
        enc_state = self.rnn_enc.zero_state(self.batch_size, self.dtype)

        # init hearing models  TODO add parameters for tcn net, wavegan, mfccs
        self.soundscape_len = audio_gen.soundscape_len(audio_params, self.fs)
        if self.hearing_decoder:  # tcn and carfac are off
            # self.carfac = tf_carfac.CARFAC(self.soundscape_len, self.fs, self.batch_size, self.dtype, self.npdtype)
            channels = [hearing_params['tcn_nhidden']] * hearing_params['tcn_nlevels']  # 1+2*(kernel_size-1)*(2^nlevels-1)
            # self.tcn_net = tcn.TemporalConvNet(channels, hearing_params['tcn_kernel_size'], hearing_params['tcn_dropout'])

        # set the initial value of the canvas image, so it doesn't start from gray, but from the background color
        # take into account that this initial value is passed through a sigmoid first, hence originally sigmoid(0)=0.5
        initial_canvas_val = -6  # sigmoid(-6) = 0.0024

        x = tf.reshape(self.images, [-1, self.img_h * self.img_w * self.num_colors])
        self.audio_gen_tensors = []
        self.soundscapes = []
        self.wr_attn_params = []
        self.soundscape_tensors = []
        wr_attn_tens = []
        for t in range(self.sequence_length):
            # error image + original image
            # -3 substracted, so the first image x_hat is black
            c_prev = tf.zeros((self.batch_size, self.img_h * self.img_w * self.num_colors)) + initial_canvas_val \
                if t == 0 else self.cs[t-1]
            x_hat = x - tf.sigmoid(c_prev)
            # read the image
            # r = self.read_basic(x,x_hat,h_dec_prev)
            r = self.read_attention(x, x_hat, h_dec_prev)
            # encode it to gauss distrib
            self.mu[t], self.logsigma[t], self.sigma[t], enc_state = self.encode(enc_state, tf.concat([r, h_dec_prev], 1))
            # sample from the distrib to get z
            z = self.sampleQ(self.mu[t], self.sigma[t])

            # indirection between z and audio synth params
            if network_params['z_indirection']:
                with tf.variable_scope('z_synth_indirection', reuse=self.share_parameters):
                    z = tf.layers.dense(z, audio_gen.nparams_needed(audio_params['nsoundstream'], audio_params['nmodulation'],
                                                                    audio_params['varying_delta'], audio_params['const_phase']))

            # audio processing
            with tf.variable_scope('synth', reuse=self.share_parameters):
                # soundscape = wavegan.WaveGANGenerator(z, use_batchnorm=False, train=True)
                synth_params = audio_gen.separate_params(z, audio_params['nsoundstream'], audio_params['nmodulation'],
                                                         audio_params['varying_delta'], self.logging, self.const_phase_tens)
                soundscape_tensors = audio_gen.build_audio_synthetizer(*synth_params, self.fs, self.dtype, self.batch_size, audio_params)
            self.soundscape_tensors.append(soundscape_tensors)

            self.soundscapes.append(soundscape_tensors['bin_sound_scape'])
            soundscape = soundscape_tensors['sound_scape']
            if self.logging and (t == 1 or t + 1 == self.sequence_length):  # don't want to spend too much time on summary
                tf.summary.audio('gen_audio_' + str(t), soundscape, self.fs, family='SYNTH_AUDIO')

            # below congruence costs of pitch and azimuth are computed on this
            f0 = tf.reshape(synth_params[0], [self.batch_size, audio_params['nsoundstream']])
            a0 = tf.reshape(synth_params[2], [self.batch_size, audio_params['nsoundstream']])
            azim0 = tf.reshape(synth_params[4], [self.batch_size, audio_params['nsoundstream']])
            self.audio_gen_tensors.append([azim0, f0, a0])

            if self.hearing_decoder:
                # if >1 GPUs available, place the hearing models on them
                with tf.variable_scope('hearing', reuse=self.share_parameters):
                    with tf.device(hearing_gpus[0]):
                        mfccs_hearing_repr = hearing.mfccs_hearing(soundscape, hearing_params, self.fs, self.share_parameters)
                    with tf.device(hearing_gpus[1]):
                        wg_hearing_repr = hearing.wavegan_hearing(soundscape, self.batch_size, hearing_params, self.fs, self.share_parameters)

                    # carfac and tcn are off, uncomment here and above at hearing init to activate them
                    # hearing_repr = hearing.carfac_hearing(self.carfac, soundscape, hearing_params['hearing_repr_len'])
                    # tcn_hearing_repr = hearing.tcn_hearing(self.tcn_net, soundscape, self.training, hearing_params, self.share_parameters)

                    # append dazim to hearing_repr
                    # raw_dazim is passed onto the network skipping the hearing model as the model is not binaural
                    raw_dazim = soundscape_tensors['raw_dazim']  # nbatch x nstream x nmodulation
                    raw_dazim = hearing.binaural_noise_hearing(raw_dazim, self.azim_e)
                    raw_dazim = tf.reshape(raw_dazim, [-1, audio_params['nsoundstream'] * audio_params['nmodulation']])

                    # hearing_repr = tf.concat([mfccs_hearing_repr, tcn_hearing_repr, raw_dazim], axis=1)
                    hearing_repr = tf.concat([mfccs_hearing_repr, wg_hearing_repr, raw_dazim], axis=1)
            else:  # pass audio_gen parameters raw to the decoder
                # gather raw network_params['n_z']audio_gen variables
                raw_phase = soundscape_tensors['raw_phase']
                raw_da = soundscape_tensors['raw_da']
                raw_df = soundscape_tensors['raw_df']
                raw_dazim = hearing.binaural_noise_hearing(soundscape_tensors['raw_dazim'], self.azim_e)  # simple binaural hearing

                # flatten and concat them
                raw_dazim = tf.reshape(raw_dazim, [-1, raw_dazim.shape[1] * raw_dazim.shape[2]])
                raw_phase = tf.reshape(raw_phase, [-1, raw_phase.shape[1] * raw_phase.shape[2]])
                raw_da = tf.reshape(raw_da, [-1, raw_da.shape[1] * raw_da.shape[2]])
                raw_df = tf.reshape(raw_df, [-1, raw_df.shape[1] * raw_df.shape[2]])

                hearing_repr = tf.concat([raw_dazim, raw_phase, raw_da, raw_df], axis=-1)

            # retrieve the hidden layer of RNN
            if not self.nonrecurrent_dec:
                h_dec, dec_state = self.decode_layer(dec_state, hearing_repr)
            else:
                h_dec = self.nonrecurrent_decode_layer(hearing_repr)

            write = self.write_attention(h_dec)
            wr_attn_tens.append(tf.reduce_mean(write, axis=1, keepdims=True))
            self.cs[t] = c_prev + write
            h_dec_prev = h_dec
            self.share_parameters = True  # from now on, share variables

        self.whole_soundscape = tf.concat(self.soundscapes, axis=1)
        if self.logging:
            tf.summary.audio('whole_bin_soundscape', self.whole_soundscape, self.fs, max_outputs=4)

        # the final timestep, generation loss
        self.generated_images = tf.nn.sigmoid(self.cs[-1])
        self.generation_loss = tf.nn.l2_loss(x - self.generated_images)
        if self.logging:
            tf.summary.scalar("gen_loss", self.generation_loss)
            tf.summary.image("gen_img", tf.reshape(self.generated_images, [-1, self.img_h, self.img_w, self.num_colors]))

        # latent loss
        kl_terms = [0]*self.sequence_length
        for t in range(self.sequence_length):
            mu2 = tf.square(self.mu[t])
            sigma2 = tf.square(self.sigma[t])
            logsigma = self.logsigma[t]
            kl_terms[t] = self.kl_weight * tf.reduce_sum(mu2 + sigma2 - 2*logsigma, 1) - self.kl_weight
        self.latent_loss = tf.reduce_mean(tf.add_n(kl_terms))
        if self.logging:
            tf.summary.scalar("lat_loss", self.latent_loss)

        # cost of pitch and azimuth congruence w/ gx and gy drawing positions
        if self.congruence_weight > 0:
            # let's constrain so gx: [0, width], gy: [0, height]; they are boundless otherwise, this cost will keep them down
            # audio_gen_params: [[seq1azim0, seq1f0, seq1a0], seq2, ...], azim0,f0,a0 size: batch x nsoundstream
            soundstream_pos = [[azim, f] for azim, f, _ in self.audio_gen_tensors]  # remove a0s
            # azim0: [-1,1], f0: [0,1], bring them to [0,width], [0,height], respectively
            # also flip f, because img coordinates increase downwards; and flip azim, because it's anti-clockwise
            soundstream_pos = [[self.img_w - (azim + 1.) * self.img_w / 2., self.img_h - f * self.img_h] for azim, f in soundstream_pos]
            wr_patch_pos = []
            if self.v1_gaussian:
                # attn_parameters: [[[seq1p1x, seq1p1y, s1p1d], [seq1p2x, seq1p2y, s1p2d]], seq2, ...], px,py,pd size: batch
                # first attn_parameters params inside sequences have to be stacked to get the form: batch x npatch (npatch == nsoundstream)
                for seq in self.wr_attn_params:
                    v1_patch_pos_x = [x for x, y, _, _ in seq]  # delta is not needed
                    v1_patch_pos_y = [y for x, y, _, _ in seq]
                    v1_patch_pos_x = tf.concat(v1_patch_pos_x, axis=-1)  # batch x npatch
                    v1_patch_pos_y = tf.concat(v1_patch_pos_y, axis=-1)
                    wr_patch_pos.append([v1_patch_pos_x, v1_patch_pos_y])

                # what if number of soundstreams != number of v1 patches
                if audio_params['nsoundstream'] != self.n_v1_write:
                    # avg both soundstreams and v1 patches out
                    wr_patch_pos = [[tf.reduce_mean(x, axis=-1), tf.reduce_mean(y, axis=-1)] for x, y in wr_patch_pos]
                    soundstream_pos = [[tf.reduce_mean(azim, axis=-1), tf.reduce_mean(f, axis=-1)] for azim, f in soundstream_pos]
            else:  # square shaped (original) writing patches, 1/seq
                # attn_parameters: [[seq1x, seq1y, seq1d], [s2x, s2y, s2d], ...]
                wr_patch_pos = [[x, y] for x, y, _, _ in self.wr_attn_params]  # remove ds
                # because we have only one patch/seq, but soundstreams can be multiple, soundstreams have to be averaged out
                # actually this is not necessary because tf.squared_difference below supports broadcasting
                # soundstream_pos = [[tf.reduce_mean(azim, axis=-1, keepdims=True), tf.reduce_mean(f, axis=-1, keepdims=True)]
                #                    for azim, f in soundstream_pos]
            # now soundstream_pos and wr_patch_pos are at the same size and scale, time to compute the cost as the MSE
            congr_cost_arr = []
            for ss_pos, v1_pos in zip(soundstream_pos, wr_patch_pos):
                congr_cost_arr.append(tf.squared_difference(ss_pos[0], v1_pos[0]) + tf.squared_difference(ss_pos[1], v1_pos[1]))
            pos_cost = tf.reduce_mean(tf.add_n(congr_cost_arr)) / self.sequence_length  # normalize by seq length

            # a0-drawsize: norm by the mean of draw sizes across iterations
            wr_attn_tens = tf.concat(wr_attn_tens, axis=1)
            overall_mean_write, overall_var_write = tf.nn.moments(wr_attn_tens, axes=[1])
            overall_std_write = tf.sqrt(overall_var_write)
            wr_attn_tens_norm = (wr_attn_tens - tf.expand_dims(overall_mean_write, axis=-1)) / tf.expand_dims(overall_std_write, axis=-1)  # to ~[-1,1] within an image
            # a0 from [0,1] to [-2,2] (z score); mean flatten soundstream dim; concat seq
            a0 = tf.concat([tf.reduce_mean(a, axis=1, keepdims=True) * 4 - 2 for _, _, a in self.audio_gen_tensors], axis=1)
            ampl_wrsize_cost = tf.reduce_mean(tf.squared_difference(a0, wr_attn_tens_norm)) / self.sequence_length * 200.

            self.congruence_cost = (pos_cost + ampl_wrsize_cost) * self.congruence_weight

            if self.logging:
                tf.summary.scalar("pos_loss", pos_cost)
                tf.summary.scalar("amplwrsize_loss", ampl_wrsize_cost)
                tf.summary.scalar("congr_loss", self.congruence_cost)

            # final cost
            self.cost = self.generation_loss + self.latent_loss + self.congruence_cost
        else:  # no congruence cost
            self.cost = self.generation_loss + self.latent_loss

        # gradient w/ clipping
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.lr = tf.train.exponential_decay(self.initial_lr, self.global_step, 250, 0.95)
        optimizer = tf.train.AdamOptimizer(self.lr, beta1=0.9)
        grads = optimizer.compute_gradients(self.cost)

        for i, (g, v) in enumerate(grads):
            if g is not None:
                grads[i] = (tf.clip_by_norm(g, 5), v)
        self.train_op = optimizer.apply_gradients(grads, global_step=self.global_step)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        if self.logging:
            self.merged_summary = tf.summary.merge_all()
            self.writer = tf.summary.FileWriter(os.path.join(os.getcwd(), 'summary', self.model_name), self.sess.graph)

    # given a hidden decoder layer:
    # locate where to put attention filters
    def attn_window(self, scope, h_dec):
        with tf.variable_scope(scope, reuse=self.share_parameters):
            parameters = dense(h_dec, self.n_hidden, 5)
        # gx_, gy_: center of 2d gaussian on a scale of -1 to 1
        gx_, gy_, log_sigma2, log_delta, log_gamma = tf.split(parameters, 5, 1)

        # move gx/gy to be a scale of -imgsize to +imgsize
        gx = (self.img_w + 1) / 2 * (gx_ + 1)
        gy = (self.img_h + 1) / 2 * (gy_ + 1)

        sigma2 = tf.exp(log_sigma2)
        # stride/delta: how far apart these patches will be
        delta = (self.min_img_dim - 1) / ((self.attention_n-1) * tf.exp(log_delta))

        if scope == 'write':
            self.wr_attn_params.append([gx, gy, delta, delta])  # the last one would be the angle but not applicable here
        return self.filterbank(gx, gy, sigma2, delta) + (tf.exp(log_gamma),)

    def multi_v1_attn_window(self, scope, h_dec):  # writing attention window
        attn_arr = []
        attn_params_prep = []
        for i in range(self.n_v1_write):
            scope_i = scope + str(i)
            with tf.variable_scope(scope_i, reuse=self.share_parameters):
                parameters = dense(h_dec, self.n_hidden, 6)
            gx_, gy_, log_sigma2, log_delta, log_gamma, angle = tf.split(parameters, 6, axis=1)
            gx = (self.img_w + 1) / 2 * (gx_ + 1)
            gy = (self.img_h + 1) / 2 * (gy_ + 1)
            sigma2 = tf.exp(log_sigma2)
            delta = (self.min_img_dim - 1) / ((self.attention_n - 1) * tf.exp(log_delta))

            if self.logging:
                tf.summary.histogram(scope_i + '_angle', angle, family=scope.upper() + '_V1_ANGLE')
                tf.summary.histogram(scope_i + '_gx', gx, family=scope.upper() + '_V1_GX')
                tf.summary.histogram(scope_i + '_gy', gy, family=scope.upper() + '_V1_GY')
                tf.summary.histogram(scope_i + '_sigma2', sigma2, family=scope.upper() + '_V1_SIGMA2')
                tf.summary.histogram(scope_i + '_delta', delta, family=scope.upper() + '_V1_DELTA')

            attn_params_prep.append([gx, gy, delta, angle])
            filterbank = self.v1_filterbank(gx, gy, angle, sigma2, delta) + (tf.exp(log_gamma),)
            attn_arr.append(filterbank)

        # [[[seq1p1x, seq1p1y, s1p1d], [seq1p2x, seq1p2y, s1p2d]], [[seq2p1x, seq2p1y, s2p1d], [seq2p2x, seq2p2y, s2p2d]] ...]
        self.wr_attn_params.append(attn_params_prep)
        return attn_arr

    # Given a center, distance, and spread
    # Construct [attention_n x attention_n] patches of gaussian filters
    # represented by Fx = horizontal gaussian, Fy = vertical guassian
    def filterbank(self, gx, gy, sigma2, delta):
        # 1 x N, look like [[0,1,2,3,4]]
        grid_i = tf.reshape(tf.cast(tf.range(self.attention_n), self.dtype), [1, -1])
        # centers for the individual patches
        mu_x = gx + (grid_i - self.attention_n/2 - 0.5) * delta
        mu_y = gy + (grid_i - self.attention_n/2 - 0.5) * delta
        mu_x = tf.reshape(mu_x, [-1, self.attention_n, 1])
        mu_y = tf.reshape(mu_y, [-1, self.attention_n, 1])
        # 1 x 1 x imgsize, looks like [[[0,1,2,3,4,...,27]]]
        im_a = tf.reshape(tf.range(self.img_w, dtype=self.dtype), [1, 1, -1])
        im_b = tf.reshape(tf.range(self.img_h, dtype=self.dtype), [1, 1, -1])
        # list of gaussian curves for x and y
        sigma2 = tf.reshape(sigma2, [-1, 1, 1])
        Fx = tf.exp(-tf.square(im_a - mu_x) / (2*sigma2))
        Fy = tf.exp(-tf.square(im_b - mu_y) / (2*sigma2))
        # normalize so area-under-curve = 1
        Fx = Fx / tf.maximum(tf.reduce_sum(Fx, 2, keepdims=True), 1e-8)
        Fy = Fy / tf.maximum(tf.reduce_sum(Fy, 2, keepdims=True), 1e-8)
        return Fx, Fy

    # given a center, variance, spread and angle,
    # constructs a [attention_n x 1] patches of gaussian filters
    # andgle constraint: [-pi/2, pi/2]
    # only used when writing
    def v1_filterbank(self, gx, gy, angle, sigma2, delta):
        # gx and gy are the center coordinates
        # construct mu_x, mu_y then rotate by given angle
        # mu_x is initially same for all patches
        grid_i = tf.reshape(tf.range(self.attention_n, dtype=self.dtype), [1, -1])
        mu_x = tf.zeros([self.batch_size, self.attention_n], dtype=self.dtype)  # [[0,0,...]], batch x attn
        mu_y = (grid_i - self.attention_n / 2 - 0.5) * delta    # [[y1,y2,...]]

        mu = tf.stack([mu_x, mu_y], axis=1)
        cos = tf.cos(angle)
        sin = tf.sin(angle)
        rot_mx = tf.concat([[cos, -sin], [sin, cos]], axis=2)
        rot_mx = tf.transpose(rot_mx, [1, 0, 2])  # batch x 2 x 2
        mu = tf.matmul(rot_mx, mu) + tf.stack([gx, gy], axis=1)
        mu_x, mu_y = tf.split(mu, 2, axis=1)
        mu_x = tf.transpose(mu_x, [0, 2, 1])
        mu_y = tf.transpose(mu_y, [0, 2, 1])
        sigma2 = tf.reshape(sigma2, [-1, 1, 1])

        im_a = tf.reshape(tf.range(self.img_w, dtype=self.dtype), [1, 1, -1])
        im_b = tf.reshape(tf.range(self.img_h, dtype=self.dtype), [1, 1, -1])
        # Fxy = tf.exp(-(tf.square(im_a - mu_x) / (2*sigma2) + tf.square(im_b - mu_y) / (2*sigma2)))
        Fx = tf.exp(-tf.square((im_a - mu_x)) / (2 * sigma2))
        Fy = tf.exp(-tf.square((im_b - mu_y)) / (2 * sigma2))
        Fx = Fx / tf.maximum(tf.reduce_sum(Fx, 2, keepdims=True), 1e-8)
        Fy = Fy / tf.maximum(tf.reduce_sum(Fy, 2, keepdims=True), 1e-8)

        return Fx, Fy

    # the read() operation without attention
    def read_basic(self, x, x_hat, h_dec_prev):
        return tf.concat([x, x_hat], 1)

    def read_attention(self, x, x_hat, h_dec_prev):

        def filter_img_layer(img_layer, Fx, Fy, gamma):
            Fxt = tf.transpose(Fx, perm=[0, 2, 1])
            img_layer = tf.reshape(img_layer, [-1, self.img_h, self.img_w])

            # apply the gaussian patches
            glimpse = tf.matmul(Fy, tf.matmul(img_layer, Fxt))
            glimpse = tf.reshape(glimpse, [-1, self.attention_n**2])

            # finally scale this glimpse w/ the gamma parameter
            return glimpse * gamma

        # we have the parameters for a patch of gaussian filters. apply them.
        def filter_img(img, Fx, Fy, gamma):

            img = tf.reshape(img, [-1, self.img_h, self.img_w, self.num_colors])
            img_t = tf.transpose(img, perm=[3, 0, 1, 2])

            # color1, color2, color3, color1, color2, color3, etc.
            batch_colors_array = tf.reshape(img_t, [self.num_colors * self.batch_size, self.img_h, self.img_w])
            Fx_array = tf.tile(Fx, [self.num_colors, 1, 1])
            Fy_array = tf.tile(Fy, [self.num_colors, 1, 1])
            # else:
            #     Fx_array = tf.concat(Fx, 0)
            #     Fy_array = tf.concat(Fy, 0)

            Fxt = tf.transpose(Fx_array, perm=[0, 2, 1])

            # Apply the gaussian patches:
            glimpse = tf.matmul(Fy_array, tf.matmul(batch_colors_array, Fxt))
            glimpse = tf.reshape(glimpse, [self.num_colors, self.batch_size, self.attention_n, self.attention_n])
            glimpse = tf.transpose(glimpse, [1,2,3,0])
            glimpse = tf.reshape(glimpse, [self.batch_size, self.attention_n*self.attention_n*self.num_colors])
            # finally scale this glimpse w/ the gamma parameter
            return glimpse * tf.reshape(gamma, [-1, 1])

        # regular grid like attention window used to read (v1 gaussian patches are used only at writing)
        if self.grayscale:
            Fx, Fy, gamma = self.attn_window("read_layer", h_dec_prev)
            x = filter_img_layer(x, Fx, Fy, gamma)
            x_hat = filter_img_layer(x_hat, Fx, Fy, gamma)
            return tf.concat([x, x_hat], 1)
        else:  # multi color
            Fx, Fy, gamma = self.attn_window("read", h_dec_prev)
            x = filter_img(x, Fx, Fy, gamma)
            x_hat = filter_img(x_hat, Fx, Fy, gamma)
            return tf.concat([x, x_hat], 1)

    # encode an attention patch
    def encode(self, prev_state, image):
        # update the RNN with image
        with tf.variable_scope("encoder", reuse=self.share_parameters):
            hidden_layer, next_state = self.rnn_enc(image, prev_state)
            hidden_layer = tf.layers.batch_normalization(hidden_layer)

        # map the RNN hidden state to latent variables
        with tf.variable_scope("mu", reuse=self.share_parameters):
            mu = dense(hidden_layer, self.n_hidden, self.n_z)
        with tf.variable_scope("sigma", reuse=self.share_parameters):
            logsigma = dense(hidden_layer, self.n_hidden, self.n_z)
            sigma = tf.exp(logsigma)  # + 1e-5  # add eps if nans appear: https://www.reddit.com/r/MachineLearning/comments/4eqifs/gaussian_observation_vae/
        return mu, logsigma, sigma, next_state

    def sampleQ(self, mu, sigma):
        if self.training:
            return mu + sigma * self.e
        return mu

    def decode_layer(self, prev_state, latent):
        # update decoder RNN with latent var
        with tf.variable_scope("decoder", reuse=self.share_parameters):
            hidden_layer, next_state = self.rnn_dec(latent, prev_state)
            hidden_layer = tf.layers.batch_normalization(hidden_layer)

        return hidden_layer, next_state

    def nonrecurrent_decode_layer(self, latent):
        with tf.variable_scope("nonrecurrent_decoder", reuse=self.share_parameters):
            for i in range(self.n_rnn_cells[1]):
                latent = self.rnn_dec[i](latent)
        return latent

    def write_basic(self, hidden_layer):
        # map RNN hidden state to image
        with tf.variable_scope("write", reuse=self.share_parameters):
            decoded_image_portion = dense(hidden_layer, self.n_hidden, self.img_h * self.img_w * self.num_colors)
        return decoded_image_portion

    def write_attention(self, hidden_layer):

        if self.v1_gaussian:  # v1 attention
            # tested, doesn't add much if w is variable, in fact, it draws black as well
            # with tf.variable_scope("writeW", reuse=self.share_parameters):
            #     w = dense(hidden_layer, self.n_hidden, self.attention_n)
            attn_arr = self.multi_v1_attn_window("write", hidden_layer)

            wrs = []
            for Fx, Fy, gamma in attn_arr:
                Fx = tf.reshape(Fx, [self.batch_size, self.attention_n, 1, -1])
                Fy = tf.reshape(Fy, [self.batch_size, self.attention_n, -1, 1])
                Fxy = tf.matmul(Fy, Fx)
                Fxy = tf.transpose(Fxy, [0, 2, 3, 1])

                # w = tf.reshape(w, [-1, 1, 1, self.attention_n])
                # wr = tf.reduce_sum(Fxy * w, axis=3)
                wr = tf.reduce_sum(Fxy * self.v1_wr, axis=3)
                wr = tf.reshape(wr, [-1, self.img_h * self.img_w])

                wrs.append(wr * (1. / gamma))

            return tf.add_n(wrs)

        # original write attention
        with tf.variable_scope("writeW", reuse=self.share_parameters):
            w = dense(hidden_layer, self.n_hidden, self.attention_n * self.attention_n * self.num_colors)

        # w contains the values to write at each Gaussian patch
        w = tf.reshape(w, [self.batch_size, self.attention_n, self.attention_n, self.num_colors])
        w_t = tf.transpose(w, perm=[3, 0, 1, 2])

        Fx, Fy, gamma = self.attn_window("write", hidden_layer)
        # color1, color2, color3, color1, color2, color3, etc.
        w_array = tf.reshape(w_t, [self.num_colors * self.batch_size, self.attention_n, self.attention_n])
        Fx_array = tf.tile(Fx, [self.num_colors, 1, 1])
        Fy_array = tf.tile(Fy, [self.num_colors, 1, 1])

        Fyt = tf.transpose(Fy_array, perm=[0, 2, 1])
        # [vert, attn_n] * [attn_n, attn_n] * [attn_n, horiz]
        wr = tf.matmul(Fyt, tf.matmul(w_array, Fx_array))
        # sep_colors = tf.reshape(wr, [self.batch_size, self.num_colors, self.img_size**2])
        wr = tf.reshape(wr, [self.num_colors, self.batch_size, self.img_h, self.img_w])
        wr = tf.transpose(wr, [1, 2, 3, 0])
        wr = tf.reshape(wr, [self.batch_size, self.img_h * self.img_w * self.num_colors])

        return wr * tf.reshape(1.0 / gamma, [-1, 1])

    def get_data(self, dataset):
        return tables.open_file(dataset, mode='r')

    def get_batch(self, data, indices=None, batch_size=None, start_stop_index=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        if indices is None and start_stop_index is None:
            indices = np.random.randint(0, data.shape[0], (batch_size,))

        elif start_stop_index is not None:
            return np.array([get_image(d, self.grayscale)
                             for d in data[start_stop_index[0]:start_stop_index[1]]]).astype(self.npdtype)

        return np.array(
            [get_image(data[i], self.grayscale) for i in indices]).astype(self.npdtype)

    def train(self, dataset, restore=True, model_name=None):
        self.model_name = model_name or self.model_name

        data = self.get_data(dataset)
        base = self.get_batch(data.root.train_img, np.arange(0, self.batch_size))
        data_len = data.root.train_img.shape[0]

        ims(os.path.join(os.getcwd(), 'results', self.model_name, 'base.png'), merge_color(base, [8, self.batch_size // 8]))

        saver = tf.train.Saver(max_to_keep=2)
        if restore and os.path.exists(os.path.join(os.getcwd(), 'training', self.model_name)):
            saver.restore(self.sess, tf.train.latest_checkpoint(os.path.join(os.getcwd(), 'training', self.model_name)))
            print('MODEL "{}" RESTORED'.format(self.model_name), file=sys.stderr)
        else:
            print('NEW MODEL "{}" IS BEING TRAINED'.format(self.model_name), file=sys.stderr)

        start_time = time.time()
        for e in range(self.nepoch):
            nbatch = (data_len // self.batch_size) - 2
            for i in range(nbatch):

                batch_images = self.get_batch(data.root.train_img)  # LOAD RANDOM BATCHES
                cs, attn_params, gen_loss, lat_loss, _, glob_step = self.sess.run([self.cs, self.wr_attn_params, self.generation_loss,
                                                                                   self.latent_loss, self.train_op, self.global_step],
                                                                                  feed_dict={self.images: batch_images})
                if (e * nbatch + i + 1) % self.log_after == 0 and self.logging:
                    time_spent = time.time() - start_time
                    start_time = time.time()
                    s = self.sess.run(self.merged_summary, feed_dict={self.images: batch_images})
                    self.writer.add_summary(s, glob_step)
                    s = tf.Summary(value=[tf.Summary.Value(tag='training_time', simple_value=time_spent)])
                    self.writer.add_summary(s, glob_step)

                    # run on test set
                    batch_images = self.get_batch(data.root.test_img, start_stop_index=(0, self.batch_size))
                    test_gen_loss, test_lat_loss = self.sess.run([self.generation_loss, self.latent_loss], feed_dict={self.images: batch_images})
                    sg = tf.Summary(value=[tf.Summary.Value(tag='test_genloss', simple_value=test_gen_loss)])
                    sl = tf.Summary(value=[tf.Summary.Value(tag='test_latloss', simple_value=test_lat_loss)])
                    self.writer.add_summary(sg, glob_step)
                    self.writer.add_summary(sl, glob_step)

                    print("glob {} epoch {} iter {} genloss {} latloss {} testgenloss {} time {}".format(
                        glob_step, e, i, gen_loss, lat_loss, test_gen_loss, time_spent), file=sys.stderr)

                    if gen_loss != gen_loss:  # nan
                        print('NaN value found, exiting', file=sys.stderr)
                        return

                if (e * nbatch + i + 1) % self.save_after == 0:
                    saver.save(self.sess, os.path.join(os.getcwd(), 'training', self.model_name, 'train'), global_step=glob_step)
                    print('MODEL "{}" SAVED at iteration {}'.format(self.model_name, e * self.nepoch + i), file=sys.stderr)

                    cs = 1.0/(1.0+np.exp(-np.array(cs)))  # x_recons=sigmoid(canvas)

                    for cs_iter in range(self.sequence_length):
                        results = cs[cs_iter]
                        results_square = np.reshape(results, [-1, self.img_h, self.img_w, self.num_colors])
                        # print results_square.shape
                        ims(os.path.join(os.getcwd(), 'results', self.model_name, str(e)+'-'+str(i)+'-step-'+str(cs_iter)+'.png'),
                            merge_color(results_square, [8, self.batch_size // 8]))

    def gen_vids(self, dataset, training_path=None, model_name=None, output_prefix=''):
        # pass random batch, save output images and sounds, concat images and add sound w/ ffmpeg
        self.model_name = model_name or self.model_name
        training_path = training_path or os.path.join(os.getcwd(), 'training')

        data = self.get_data(dataset)
        batch = self.get_batch(data.root.test_img, np.arange(0, self.batch_size))
        saver = tf.train.Saver(max_to_keep=2)
        saver.restore(self.sess, tf.train.latest_checkpoint(os.path.join(training_path, self.model_name)))

        cs, soundscapes, attn_params, gen_loss, lat_loss = self.sess.run(
            [self.cs, self.whole_soundscape, self.wr_attn_params, self.generation_loss, self.latent_loss],
            feed_dict={self.images: batch})

        print("genloss {} latloss {}".format(gen_loss, lat_loss), file=sys.stderr)

        cs = 1.0 / (1.0 + np.exp(-np.array(cs)))  # x_recons=sigmoid(canvas)

        # save images
        for cs_iter in range(self.sequence_length):
            canvas_images = np.reshape(cs[cs_iter], [-1, self.img_h, self.img_w])

            # iterate through all images at this cs_iter, and save them temporaly
            for idx, img in enumerate(canvas_images):
                ims('tmp/' + output_prefix + 'img_{0}_iter_{1:0=2d}.png'.format(idx, cs_iter), img)

        # save sounds
        for i in range(self.batch_size):
            wavfile.write('tmp/' + output_prefix + 'sound_{}.wav'.format(i), self.fs, soundscapes[i])

        # concat imgs + add sound, use ffmpeg from command line
        nimg_per_sec = int(1. / (self.soundscape_len / self.fs))
        for i in range(self.batch_size):
            os.system('ffmpeg -r {} -i tmp/{}img_{}_iter_%02d.png -i tmp/{}sound_{}.wav -shortest -strict -2 -vcodec libx264 -y tmp/{}movie_{}.mp4'
                      .format(nimg_per_sec, output_prefix, i, output_prefix, i, output_prefix, i))  # mpeg4 if libx264 does not work

        # concat videos together to a single video; first fill mylist.txt with the list of videos
        # ffmpeg -f concat -safe 0 -i mylist.txt -c copy concat/table3-nov1-8seq.mp4

        # remove temporal images and sounds
        os.system('rm tmp/*.png tmp/*.wav')

    def prepare_run_single(self, training_path):
        saver = tf.train.Saver(max_to_keep=2)
        saver.restore(self.sess, tf.train.latest_checkpoint(training_path))

    def run_single(self, img, gen_img_needed=False, canvas_imgs_needed=False):
        batch = np.expand_dims(img, 0)

        if canvas_imgs_needed:
            cs, gen_imgs, soundscapes = self.sess.run([self.cs, self.generated_images, self.whole_soundscape],
                                                      feed_dict={self.images: batch})
            cs = 1.0 / (1.0 + np.exp(-np.array(cs)))  # x_recons=sigmoid(canvas)
            cs = np.reshape(cs, [self.sequence_length, self.img_h, self.img_w])
            return soundscapes[0], np.reshape(gen_imgs[0], [self.img_h, self.img_w]), cs

        if gen_img_needed:
            soundscapes, gen_imgs = self.sess.run([self.whole_soundscape, self.generated_images],
                                                  feed_dict={self.images: batch})
            return soundscapes[0], np.reshape(gen_imgs[0], [self.img_h, self.img_w])

        soundscapes = self.whole_soundscape.eval(feed_dict={self.images: batch}, session=self.sess)
        return soundscapes[0]

    def view(self, dataset, model_name=None):
        self.model_name = model_name or self.model_name

        data = self.get_data(dataset)
        base = self.get_batch(data.root.test_img, np.arange(0, self.batch_size))

        # base += 1
        # base /= 2

        ims(os.path.join(os.getcwd(), 'results', self.model_name, 'base.png'), merge_color(base, [8, self.batch_size // 8]))

        saver = tf.train.Saver(max_to_keep=2)
        saver.restore(self.sess, tf.train.latest_checkpoint(os.path.join(os.getcwd(), 'training', self.model_name)))

        cs, attn_params, gen_loss, lat_loss = self.sess.run([self.cs, self.wr_attn_params, self.generation_loss, self.latent_loss],
                                                            feed_dict={self.images: base})
        print("genloss {} latloss {}".format(gen_loss, lat_loss), file=sys.stderr)

        cs = 1.0/(1.0+np.exp(-np.array(cs)))  # x_recons=sigmoid(canvas)

        for cs_iter in range(self.sequence_length):
            results = cs[cs_iter]
            results_square = np.reshape(results, [-1, self.img_h, self.img_w, self.num_colors])

            ims(os.path.join(os.getcwd(), 'results', self.model_name, 'view-clean-step-' + str(cs_iter) + '.png'),
                merge_color(results_square, [8, self.batch_size // 8]))


from config import *


if __name__ == '__main__':

    # args: dataset_path, log_after, save_after
    dataset = sys.argv[1]  # path to dataset
    log_after = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    save_after = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    config_name = sys.argv[4] if len(sys.argv) > 4 else None
    lr = float(sys.argv[5]) if len(sys.argv) > 5 else None
    train_or_test = sys.argv[6] == 'train' if len(sys.argv) > 6 else True
    test_out_prefix = sys.argv[7] if len(sys.argv) > 7 else ''
    model_to_test_name = sys.argv[8] if len(sys.argv) > 8 else None

    print('TENSORFLOW VERSION:', tf.__version__, file=sys.stderr)
    print('DATA SET:', dataset, file=sys.stderr)

    # create necessary folders
    if not os.path.exists('training'):
        os.mkdir('training')
    if not os.path.exists('summary'):
        os.mkdir('summary')
    if not os.path.exists('results'):
        os.mkdir('results')

    # model parameters TODO most of it to config
    nepoch = 10000
    img_h = 120
    img_w = 160
    grayscale = True
    num_colors = 1 if grayscale else 3
    logging = True if log_after > 0 else False  # whether to create tensorboard summaries while training

    # load config, save if name given
    # FIXME configs should be saved in a single file
    network_params = load_config(config_name)
    if lr:
        network_params['learning_rate'] = lr
    if config_name:
        save_config(network_params, config_name)
    pprint(network_params, stream=sys.stderr)

    model = Draw(nepoch, img_h, img_w, num_colors, grayscale, network_params, logging=logging,
                 log_after=log_after, save_after=save_after, training=train_or_test)
    print('MODEL IS BUILT', file=sys.stderr)

    # print number of params
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        name = variable.name
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        print(name, shape, variable_parameters)
        total_parameters += variable_parameters
    print('total_parameters', total_parameters, file=sys.stderr)

    if train_or_test:
        model.train(dataset, restore=True)
    else:
        model.gen_vids(dataset, output_prefix=test_out_prefix, model_name=model_to_test_name, training_path='/media/viktor/0C22201D22200DF0/triton/triton_training/training/')
        model.view(dataset)
