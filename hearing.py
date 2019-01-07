import numpy as np
import tensorflow as tf
import ops
import wavegan
import tcn


def lstm_hearing(soundscape, window_len, n_hidden, batch_size, share_parameters):
    ncells = 10
    win_size = 2352
    overlap = 0.75  # 0.5

    lstm_cells = tf.contrib.rnn.LayerNormBasicLSTMCell(n_hidden)
    h_prev = tf.zeros((batch_size, n_hidden))
    # TODO


def noise_hearing(f_mod, a_mod, phase):
    # FIXME if you perform noising before the audio_gen scaling, then it's "just" a second layer of Gaussian noise
    # TODO noise f0, and df separately, former dependent on pitch discrimination skills, latter on freq mod discr ability (given the length of the soundstreams)
    # TODO noise a0 and df sep., by abs ampl discr skills and ampl mod discr ability (given the length of the soundstreams)
    # min freq discr (in cochlea addressing logarithmic scale)
    pass # TODO


def binaural_noise_hearing(azim_mod, normal_noise):
    # TODO still not incorporate the effects of too much soundstreams played in one soundscape on loc error
    # dazim is around [-pi/2, pi/2], abs val is max pi/2 + nmodulation*MAX_DAZIM
    # noise the azimuth modulations proportinately to abs value of azimuth - less accurate towards the lateral
    # accuracy: 1° (0.017453 rad) at 0, 10° at pi/4, 12° (0.261799 rad) at pi/2
    # exp function is fitted to these datapoints:
    def localization_accuracy(azim_mod):
        return 0.0647 * tf.exp(tf.abs(azim_mod)) - 0.0506
    # working with ~98 percentile: azim + 2std = max_noised_azim, azim - 2std = min_noised_azim
    # --> 4std = accuracy --> std = accuracy/4
    # degree1_rad = 0.017453
    # degree5_rad = 0.087276
    # degree15_rad = 0.261800
    # noise_amount = tf.abs(azim_mod) / (np.pi/2) * (degree15_rad / 4.) + degree1_rad / 4.
    noise_amount = localization_accuracy(azim_mod) / 4.
    return azim_mod + normal_noise * noise_amount


# from https://github.com/JEddy92/TimeSeries_Seq2Seq/blob/master/notebooks/TS_Seq2Seq_Conv_Intro.ipynb
#  and https://colab.research.google.com/drive/1la33lW7FQV1RicpfzyLq9H0SH1VSD4LE#scrollTo=cRTtl0mey-go&forceEdit=true&offline=true&sandboxMode=true
def tcn_hearing(tcn_net, soundscape, training, hearing_params, share_params):
    with tf.variable_scope("tcn_hearing", reuse=share_params):
        soundscape = tf.expand_dims(soundscape, axis=-1)
        output = tcn_net(soundscape, training=training)
        output = tf.reshape(output, [-1, output.shape[1] * output.shape[2]])
        output = tf.layers.dense(output, hearing_params['hearing_repr_len'], activation=tf.nn.tanh, name='tcn_out_dense1')
        output = tf.layers.batch_normalization(output, name='tcn_out_batch1')
        output = tf.layers.dense(output, hearing_params['hearing_repr_len'], activation=tf.nn.tanh, name='tcn_out_dense2')
        output = tf.layers.batch_normalization(output, name='tcn_out_batch2')
        return output


def wavenet_hearing(soundscape):
    pass # TODO


def wavegan_hearing(soundscape, batch_size, hearing_params, fs, share_params):
    with tf.variable_scope("wavegan_hearing", reuse=share_params):
        kernel_len_sample = int(hearing_params['wg_kernel_len'] * fs)
        strides_sample = int(hearing_params['wg_strides'] * fs)
        soundscape_rs = tf.reshape(soundscape, [batch_size, -1, 1])
        hearing_repr = wavegan.WaveGANDiscriminator(soundscape_rs,
                                                    dim=hearing_params['wg_nfilters'], kernel_len=kernel_len_sample,
                                                    strides=strides_sample, use_batchnorm=True)
        hearing_repr = tf.layers.dense(hearing_repr, hearing_params['hearing_repr_len'], activation=tf.nn.tanh)
        hearing_repr = tf.layers.batch_normalization(hearing_repr)
        return hearing_repr


# from https://www.tensorflow.org/api_guides/python/contrib.signal#Computing_Mel_Frequency_Cepstral_Coefficients_MFCCs_
def mfccs_hearing(soundscape, hearing_params, fs, share_params):
    with tf.variable_scope("mfccs_hearing", reuse=share_params):
        frame_length_sample = int(hearing_params['mfcss_frame_len'] * fs)
        frame_step_sample = int(hearing_params['mfcss_frame_step'] * fs)
        stft = tf.contrib.signal.stft(soundscape, frame_length_sample, frame_step_sample)
        magnitude_spectrograms = tf.abs(stft)

        num_spectrogram_bins = magnitude_spectrograms.shape[-1].value
        lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, fs/2, 100
        linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
            num_mel_bins, num_spectrogram_bins, fs, lower_edge_hertz, upper_edge_hertz)
        mel_spectrograms = tf.tensordot(magnitude_spectrograms, linear_to_mel_weight_matrix, 1)
        mel_spectrograms.set_shape(magnitude_spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))

        log_offset = 1e-6
        log_mel_spectrograms = tf.log(mel_spectrograms + log_offset)

        num_mfccs = hearing_params['mfcss_nceps']
        mfccs = tf.contrib.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)[..., :num_mfccs]
        mfccs = tf.reshape(mfccs, [-1, num_mfccs * mfccs.shape[1].value])

        mfccs = tf.layers.dense(mfccs, hearing_params['hearing_repr_len'], activation=tf.nn.tanh)
        mfccs = tf.layers.batch_normalization(mfccs)

        return mfccs


def carfac_hearing(carfac, soundscape, hearing_repr_len):
    carfac.run(soundscape)
    hearing_repr = carfac.output()
    hearing_repr = tf.expand_dims(hearing_repr, -1)  # batch x cochlea section x soundlen x 1
    hearing_repr = tf.image.resize_bilinear(hearing_repr, [carfac.nsec, hearing_repr_len])
    return tf.reshape(hearing_repr, [-1, carfac.nsec * hearing_repr_len])
