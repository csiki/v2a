import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os


# human hearing constants
# JND_AZIM = 0.017453  # FIXME 1 degree in rad now, but should be the smallest in literature at angle 0 - not used yet here
# JND_AMPL = 3.  # FIXME 3db for now
# JND_FREQ = 2.  # FIXME 2 Hz for now

MIN_FREQ = 100.
MAX_AMPL = 1.
# we perceive 10 octaves total (if fs is 44100), and have 64 oct/sec sensitivity
# with a 10 ms section length, that's a 0.64 oct change max
# f of 0->1 here is 0->10 octaves (if fs is 44100), so 10/0.64=15.6th of the spectrum can change within 10ms
# 1/15.6 = 0.064 is the max dfreq for one modulation, if 5 modulations, max dfreq is around 0.32
# if fs < 44100 or frequency max < 20kHz, the spectrum is smaller, so max dfreq > 0.32
# if section length < 10ms, max dfreq should get lowered
MAX_DFREQ = 0.5  # max change summed up across all modulations
MAX_DA = 0.5  # max change across all modulations
MAX_DAZIM = np.pi / 2.  # across all modulations, in radian

ALPHA_A = 0.3  # alpha parameter of amplitude distribution
# A_F = 92.  # 'A' parameter of frequency distribution, defining the spread of freq in use, 163 for humans
# 92. --> 11603 Hz max
# 87. --> 10978 Hz max
# 60. -->  7600 Hz max  # defined below

HEAD_RADIUS = 8.75 / 100.  # m
SPEED_OF_SOUND = 343.  # m/s


# Woordowrth model: from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3985894/
def ITD2(angle, freq):
    return HEAD_RADIUS / SPEED_OF_SOUND * (angle + tf.sin(angle))


# fitted to Figure 1 of https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0089033
# in dB: ILD = (f/10000)^(0.42)*sin(ang)*20
# B = 20*log10(p1/p0) from https://en.wikipedia.org/wiki/Sound_localization
# p1/p0 = 10^(B/20)
# in pressure level ratio, as left/right = 10^((f/10000)^(0.42)*sin(ang)*20/20) = 10^((f/10000)^(0.42)*sin(ang))
def ILD2(angle, freq):  # my own constructed ILD that works for + and - angles
    return tf.pow(10., tf.pow(freq / 10000., 0.42) * tf.sin(angle))


# upsample to stream length: df, da, dazim
def upsample_1d(x, nsoundstream, to_size, name, dtype):
    # x = tf.expand_dims(x, axis=1)
    x = tf.expand_dims(x, axis=-1)
    x = tf.image.resize_bilinear(x, [nsoundstream, to_size])  # sound stream dimension stays intact
    return tf.cast(tf.reshape(x, [-1, nsoundstream, to_size]), dtype=dtype, name=name + '_upsampled')


# soundscape = k x soundstream = k x nmodulation x soundsection
# if delta is constant, then just repeat it for nmodulation times, and then provide as input
def build_audio_synthetizer(f0_in, df_in, a0_in, da_in, azim0_in, dazim_in, phase_in, fs, dtype, batch_size, stream_params):

    nsoundstream = stream_params['nsoundstream']
    nmodulation = stream_params['nmodulation']  # number of modulations = number of sound sections

    attack_len = int(stream_params['attack_len_msec'] / 1000. * fs)
    decay_len = int(stream_params['decay_len_msec'] / 1000. * fs)
    section_len = int(stream_params['section_len_msec'] / 1000. * fs)
    soundstream_len = int(nmodulation * section_len)
    soundscape_len = int(soundstream_len * stream_params['soundscape_len_by_stream_len'])

    # time and ramper
    t = tf.cast(tf.range(start=0, limit=soundstream_len, delta=1) / fs, dtype=dtype, name='t')
    ramper = tf.concat([tf.constant(np.logspace(-1., 0., attack_len), dtype=dtype),
                        tf.ones(soundstream_len - attack_len - decay_len, dtype=dtype),
                        tf.constant(np.logspace(0., -1., decay_len), dtype=dtype)], axis=0)

    # freq
    # from https://www.ncbi.nlm.nih.gov/pubmed/2373794
    # CF = A(10^(αx) − k), CF distribution in Hz, x is [0,1], alpha=2.1, k=0.85, A=165.4 for humans
    # A set to 100 to have a max freq of 12kHz
    if fs > 44000:  # choose with of freq distribution according to the sampling freq
        A_F = 92.
    elif fs > 22000:
        A_F = 87.
    else:  # 16k
        A_F = 60.
    df = tf.cumsum(df_in * MAX_DFREQ / nmodulation, axis=-1) + f0_in
    df = tf.clip_by_value(df, 0., 1.)
    raw_df = A_F * (tf.pow(10., 2.1 * df) - 0.85) + MIN_FREQ

    # amplitude
    # TODO apply function on da to account for gain by freq nonlinearities
    # inverse of y = x^α, α=0.3 --> x = y^(1/α)
    da = tf.cumsum(da_in * MAX_DA / nmodulation, axis=-1) + a0_in
    da = tf.nn.relu(da)  # only positive values are sensible
    da = tf.pow(da, 1./ALPHA_A)  # inverse of y = x^alpha loudness function
    raw_da = tf.clip_by_value(da, 0., MAX_AMPL)

    # azimuth
    # TODO apply freq dependent accuracy function to dazim
    # azim0_in = tf.tanh(azim0_in)  # encourage center values close to 0, but no need, tanh already computed in separate_params
    # dazim_in = dazim_in
    azim0 = azim0_in * (np.pi / 2.)  # [-1,1] -> [-pi/2,pi/2]
    dazim = dazim_in * MAX_DAZIM / nmodulation  # [-1,1] -> [-max_dazim,max_dazim]
    raw_dazim = tf.cumsum(dazim, axis=-1) + azim0  # FIXME dazim gets out of [-pi/2,pi/2] interval

    # phase
    raw_phase = phase_in  # no scaling
    stream_phase_offset = tf.cast((soundscape_len - soundstream_len) * raw_phase, dtype=tf.int32)  # stops backprop

    df = upsample_1d(raw_df, nsoundstream, soundstream_len, 'df', dtype)
    da = upsample_1d(raw_da, nsoundstream, soundstream_len, 'da', dtype)
    dazim = upsample_1d(raw_dazim, nsoundstream, soundstream_len, 'dazim', dtype)

    if stream_params['binaural']:
        # compute ITD and ILD
        itd0_roll = tf.cast(ITD2(azim0, f0_in) * fs / 2., dtype=tf.int32)  # how much to roll the soundscape to emulate ITD
        itd = ITD2(dazim, df)
        ild = ILD2(dazim, df)  # >1 means the left is louder, <1 right is louder
        ild_sqrt = tf.sqrt(ild)  # to influence both channels equally by ILD, and not favour 1 channel

        # construct sound
        sound_stream_l = da * tf.sin(2. * np.pi * df * t + itd/2.)  # push one, pull the other equally
        sound_stream_r = da * tf.sin(2. * np.pi * df * t - itd/2.)

        # construct binaural sound: batch x stream x time x chan(l,r)
        sound_stream_l = sound_stream_l * ild_sqrt
        sound_stream_r = sound_stream_r / ild_sqrt
        two_channel_ramp = tf.stack([ramper, ramper], axis=-1)
        sound_stream = tf.stack([sound_stream_l, sound_stream_r], axis=3) * two_channel_ramp  # l;r binaural stream

        # place soundstream in soundscape
        bin_sound_scape = tf.concat([sound_stream, tf.zeros([batch_size, nsoundstream, soundscape_len - soundstream_len, 2], dtype=dtype)], axis=2)
        # sound_scape = tf.map_fn(lambda x: tf.manip.roll(x[0], x[1], axis=[0]), [sound_scape, stream_phase_offset], dtype=dtype)
        bin_sound_scape = tf.map_fn(lambda stream_x: tf.map_fn(lambda x: tf.manip.roll(x[0], x[1], axis=[-2]), [stream_x[0], stream_x[1]],
                                                               dtype=dtype), [bin_sound_scape, stream_phase_offset], dtype=dtype)

        sound_scape_l = tf.map_fn(lambda stream_x: tf.map_fn(lambda x: tf.manip.roll(x[0], x[1], axis=[-1]), [stream_x[0], stream_x[1]],
                                       dtype=dtype), [bin_sound_scape[:, :, :, 0], -itd0_roll], dtype=dtype)
        sound_scape_r = tf.map_fn(lambda stream_x: tf.map_fn(lambda x: tf.manip.roll(x[0], x[1], axis=[-1]), [stream_x[0], stream_x[1]],
                                       dtype=dtype), [bin_sound_scape[:, :, :, 1], itd0_roll], dtype=dtype)

        # sound_scape_l = tf.map_fn(lambda x: tf.manip.roll(x[0], x[1], axis=[0]), [sound_scape[:,:,0], -itd0_roll], dtype=dtype)
        # sound_scape_r = tf.map_fn(lambda x: tf.manip.roll(x[0], x[1], axis=[0]), [sound_scape[:,:,1], itd0_roll], dtype=dtype)
        bin_sound_scape = tf.stack([sound_scape_l, sound_scape_r], axis=3)
        bin_sound_scape = tf.reduce_sum(bin_sound_scape, axis=1)
        bin_sound_scape = bin_sound_scape / tf.reduce_max(bin_sound_scape)  # to range [0,1]
        #bin_sound_scape = bin_sound_scape / tf.reshape(tf.reduce_max(bin_sound_scape, axis=1) + 1e-6, [-1, 1, 2])

    else:  # single channel, no need for binaural sound
        bin_sound_scape = None

    sound_stream = da * tf.sin(2. * np.pi * df * t) * ramper
    sound_scape = tf.concat([sound_stream, tf.zeros([batch_size, nsoundstream, soundscape_len - soundstream_len], dtype=dtype)], axis=-1)
    sound_scape = tf.map_fn(lambda stream_x: tf.map_fn(lambda x: tf.manip.roll(x[0], x[1], axis=[-1]), [stream_x[0], stream_x[1]], dtype=dtype),
                            [sound_scape, stream_phase_offset], dtype=dtype)
    sound_scape = tf.reduce_sum(sound_scape, axis=1)
    sound_scape = sound_scape / tf.reshape(tf.reduce_max(sound_scape, axis=1) + 1e-6, [batch_size, 1])

    return {'sound_scape': sound_scape, 'sound_stream': sound_stream, 'df': df, 'da': da, 'dazim': dazim, 't': t,
            'raw_dazim': raw_dazim, 'raw_da': raw_da, 'raw_df': raw_df, 'raw_phase': raw_phase, 'bin_sound_scape': bin_sound_scape}


def write_wav_files(data, fs, path, prefix):  # 20*40*20*4/batch_size
    data_norm = data / np.max(np.abs(data))  # [-1,1]
    for i, w in enumerate(data_norm):
        wavfile.write(os.path.join(path, 'v1_10_' + str(prefix) + '_' + str(i) + '.wav'), fs, w)


def nparams_needed(nsoundstream, nmodulation, varying_delta, const_phase):  # batch size is not incorporated
    return (3 if const_phase else 4) * nsoundstream + 3 * nsoundstream * (nmodulation if varying_delta else 1)


def constant_phase(nsoundstream, batch_size):
    # just place soundstreams evenly
    return np.array([np.reshape(np.linspace(0, 1, nsoundstream), [nsoundstream, 1])] * batch_size)


def separate_params(inp_params, nsoundstream, nmodulation, varying_delta, logging, const_phase_tens):
    # inp_params: batch_size x nparams_needed
    # split to: f0_in, df_in, a0_in, da_in, azim0_in, dazim_in, phase_in
    d_inp_len = nmodulation if varying_delta else 1
    if const_phase_tens is None:
        split_sizes = [nsoundstream, nsoundstream * d_inp_len, nsoundstream, nsoundstream * d_inp_len, nsoundstream,
                       nsoundstream * d_inp_len, nsoundstream]
        f0_in, df_in, a0_in, da_in, azim0_in, dazim_in, phase_in = tf.split(inp_params, split_sizes, axis=1)
        phase_in = tf.layers.dense(phase_in, phase_in.shape[1], activation=tf.nn.sigmoid)  # FIXME try to remove sigmoid and rather clip by val
    else:  # const phase
        split_sizes = [nsoundstream, nsoundstream * d_inp_len, nsoundstream, nsoundstream * d_inp_len, nsoundstream,
                       nsoundstream * d_inp_len]
        f0_in, df_in, a0_in, da_in, azim0_in, dazim_in = tf.split(inp_params, split_sizes, axis=1)
        phase_in = const_phase_tens  # no need for constraints it's already between [0,1]

    # constrain tensors, values might be too large, not fitting the [0,1] and [-1,1] interval needed for audio_gen
    # f0_in, a0_in, phase_in (already solved): [0,1]
    f0_in = tf.layers.dense(f0_in, f0_in.shape[1], activation=tf.nn.sigmoid)  # FIXME try to remove sigmoid and rather clip by val, it's safe
    a0_in = tf.layers.dense(a0_in, a0_in.shape[1], activation=tf.nn.sigmoid)
    # azim0_in, df_in, da_in, dazim_in: [-1,1]
    azim0_in = tf.layers.dense(azim0_in, azim0_in.shape[1], activation=tf.nn.tanh)
    df_in = tf.layers.dense(df_in, df_in.shape[1], activation=tf.nn.tanh)
    da_in = tf.layers.dense(da_in, da_in.shape[1], activation=tf.nn.tanh)
    dazim_in = tf.layers.dense(dazim_in, dazim_in.shape[1], activation=tf.nn.tanh)

    if logging:
        tf.summary.histogram('f0', f0_in, family='SYNTH_PARAMS')
        tf.summary.histogram('a0', a0_in, family='SYNTH_PARAMS')
        tf.summary.histogram('phase', phase_in, family='SYNTH_PARAMS')
        tf.summary.histogram('azim0', azim0_in, family='SYNTH_PARAMS')
        tf.summary.histogram('df', df_in, family='SYNTH_PARAMS')
        tf.summary.histogram('da', da_in, family='SYNTH_PARAMS')
        tf.summary.histogram('dazim', dazim_in, family='SYNTH_PARAMS')

    f0_in = tf.reshape(f0_in, [-1, nsoundstream, 1])
    a0_in = tf.reshape(a0_in, [-1, nsoundstream, 1])
    azim0_in = tf.reshape(azim0_in, [-1, nsoundstream, 1])
    phase_in = tf.reshape(phase_in, [-1, nsoundstream, 1])

    df_in = tf.reshape(df_in, [-1, nsoundstream, d_inp_len])
    da_in = tf.reshape(da_in, [-1, nsoundstream, d_inp_len])
    dazim_in = tf.reshape(dazim_in, [-1, nsoundstream, d_inp_len])

    if not varying_delta:  # repeat delta values
        df_in = tf.tile(df_in, [1, 1, nmodulation])
        da_in = tf.tile(da_in, [1, 1, nmodulation])
        dazim_in = tf.tile(dazim_in, [1, 1, nmodulation])

    return f0_in, df_in, a0_in, da_in, azim0_in, dazim_in, phase_in


# needed in color draw model init
def soundscape_len(audio_params, fs):
    section_len = int(audio_params['section_len_msec'] / 1000. * fs)
    soundstream_len = int(audio_params['nmodulation'] * section_len)
    return int(soundstream_len * audio_params['soundscape_len_by_stream_len'])


def gen_single_input(f0_in, a0_in, azim0_in, phase_in, df_in, da_in, dazim_in,
                     f0_, a0_, azim0_, phase_, df_, da_, dazim_, batch_size=1):
    return {
        f0_in: [[[f0_]]*nsoundstream]*batch_size,
        a0_in: [[[a0_]]*nsoundstream]*batch_size,
        azim0_in: [[[azim0_]]*nsoundstream]*batch_size,
        phase_in: constant_phase(nsoundstream, batch_size),  # equal spacing from phase_
        df_in: [[df_]*nsoundstream]*batch_size,
        da_in: [[da_]*nsoundstream]*batch_size,
        dazim_in: [[dazim_]*nsoundstream]*batch_size
    }


def rand_gen_input(f0_in, a0_in, azim0_in, phase_in, df_in, da_in, dazim_in, batch_size, varying_delta, const_phase):
    if not const_phase:
        phase = np.random.rand(batch_size, nsoundstream, 1)
    else:
        phase = constant_phase(nsoundstream, batch_size)

    return {
        f0_in: np.random.rand(batch_size, nsoundstream, 1),
        a0_in: np.random.rand(batch_size, nsoundstream, 1),
        azim0_in: np.random.rand(batch_size, nsoundstream, 1) * 2. - 1.,
        phase_in: phase,
        df_in: (np.random.rand(batch_size, nsoundstream, nmodulation)
                if varying_delta else np.repeat(np.random.rand(batch_size, nsoundstream, 1), nmodulation, 2)) * 2. - 1.,
        da_in: (np.random.rand(batch_size, nsoundstream, nmodulation)
                if varying_delta else np.repeat(np.random.rand(batch_size, nsoundstream, 1), nmodulation, 2)) * 2. - 1.,
        dazim_in: (np.random.rand(batch_size, nsoundstream, nmodulation)
                   if varying_delta else np.repeat(np.random.rand(batch_size, nsoundstream, 1), nmodulation, 2)) * 2. - 1.
    }


# ! NOTE: ONLY USED WHEN RUNNING THIS SCRIPT
DEFAULT_STREAM_PARAMS = {
    'binaural': True,
    'varying_delta': True,
    'nsoundstream': 3,
    'nmodulation': 4,  # number of modulations: number of sound sections
    'section_len_msec': 10,
    'attack_len_msec': 1,
    'decay_len_msec': 1,
    'soundscape_len_by_stream_len': 2.5,  # should be >=1
    'const_phase': True
}


if __name__ == '__main__':
    import sounddevice
    from scipy.io import wavfile

    # constants
    fs = 44100
    dtype = tf.float32
    batch_size = 1
    nsoundstream = DEFAULT_STREAM_PARAMS['nsoundstream']
    nmodulation = DEFAULT_STREAM_PARAMS['nmodulation']
    const_phase = DEFAULT_STREAM_PARAMS['const_phase']
    soundscape_len = soundscape_len(DEFAULT_STREAM_PARAMS, fs), soundscape_len(DEFAULT_STREAM_PARAMS, fs)/fs*1000
    print('soundscape len:', soundscape_len, 'ms')
    print('params needed:', nparams_needed(nsoundstream, nmodulation, True, DEFAULT_STREAM_PARAMS['const_phase']))

    # placeholders of parameters
    f0_in = tf.placeholder(dtype, (batch_size, nsoundstream, 1), name='f0')  # [0,1]
    a0_in = tf.placeholder(dtype, (batch_size, nsoundstream, 1), name='a0')  # [0,1]
    azim0_in = tf.placeholder(dtype, (batch_size, nsoundstream, 1), name='azim0')  # [-1,1], counterclock-wise
    phase_in = tf.placeholder(dtype, (batch_size, nsoundstream, 1), name='phase')  # [0,1]
    df_in = tf.placeholder(dtype, (batch_size, nsoundstream, nmodulation), name='df')  # [-1,1]
    da_in = tf.placeholder(dtype, (batch_size, nsoundstream, nmodulation), name='da')  # [-1,1]
    dazim_in = tf.placeholder(dtype, (batch_size, nsoundstream, nmodulation), name='dazim')  # [-1,1]
    placeholders = [f0_in, a0_in, azim0_in, phase_in, df_in, da_in, dazim_in]

    # build synthetizer
    synth_tensors = build_audio_synthetizer(f0_in, df_in, a0_in, da_in, azim0_in, dazim_in, phase_in, fs, dtype, batch_size,
                                            DEFAULT_STREAM_PARAMS)

    # generate input
    varying_delta = False
    inp1 = gen_single_input(*placeholders, f0_=0.3, a0_=0.7, azim0_=0, phase_=0.,
                            df_=[0] * nmodulation, da_=[0] * nmodulation, dazim_=[-0.4] * nmodulation,
                            batch_size=batch_size)
    rand_inp = rand_gen_input(*placeholders, batch_size, varying_delta, const_phase)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sound_scape_, t_, df_, da_, dazim_ = sess.run([synth_tensors['bin_sound_scape'], synth_tensors['t'],
                                                       synth_tensors['df'], synth_tensors['da'], synth_tensors['dazim']], rand_inp)
        print('DF', df_)
        print('DA', da_)
        print('DAZIM', dazim_)

        # binary plot:
        # plt.plot(np.arange(len(sound_scape_[0])), sound_scape_[0,:,0], np.arange(len(sound_scape_[0])), sound_scape_[0,:,1]); plt.legend(['l', 'r'])
        # plt.figure(); plt.plot(np.arange(len(df_[0])), df_[0])

        # print('min', np.min(sound_scape_[0]), 'max', np.max(sound_scape_[0]))
        # print('amin', np.min(ramper_[0]), 'amax', np.max(ramper_[0]))
        # print(ramper_.tolist())
        # print('df', df_[0])

        # generate wav files
        # for i in range(20*40*20*4//batch_size):
        #     write_wav_files(sound_scape_, fs, '/media/viktor/0C22201D22200DF0/audio_data/stream', i)

        plt.plot(np.arange(len(sound_scape_[0, :])) / fs * 1000, sound_scape_[0, :] / np.max(sound_scape_[0, :], None))
        plt.legend(['left', 'right'])
        plt.xlabel('t (ms)')
        plt.ylabel('A (sound pressure)')
        repeated_soundscape = np.tile(sound_scape_[0, :], [20, 1])
        sounddevice.play(repeated_soundscape, fs)
        plt.show()
