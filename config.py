import numpy as np
import tensorflow as tf
import json


# TODO write simple description here on how configs work
# TODO FIXME fix all the places where configs/Draw model are used (gen_disentangle, test_model, ...)
# TODO finalize confgis.json then: git update-index --assume-unchanged path/to/file.txt


# BEST CONFIGS
  # ! means not enough contestants, * means confident
# nohearing v1
#   !apartment: img=120x160x1,attention=12,hidden=1024,z=39,seq_len=30,n_rnn=3-3,v1=True,nv1write=3,cw=1.0,fs=44100,hearing=False,sslen=3x3*8*1.2,mfccs=1.0-100-0.025,wg=0.002-64-0.01_properreader_residual_apartment
#   *simple hand: img=120x160x1,attention=20,hidden=1024,z=39,seq_len=30,n_rnn=(3, 3),v1=True,nv1write=3,cw=0.1,fs=44100,hearing=False,sslen=3x3*8*1.2,mfccs=0.1-100-0.025,wg=0.002-64-0.01_properreader_residual
#       sslen to 10ms
#       seq to 36

# nohearing nov1
#   !apartment: img=120x160x1,attention=20,hidden=1024,z=39,seq_len=20,n_rnn=(3, 3),v1=False,nv1write=3,cw=0.1,fs=44100,hearing=False,sslen=3x3*8*1.2,mfccs=0.1-100-0.025,wg=0.002-64-0.01_properreader_residual_apartment3ss
#   *simple hand: img=120x160x1,attention=20,hidden=1024,z=26,seq_len=20,n_rnn=(3, 3),v1=False,nv1write=3,cw=0.1,fs=44100,hearing=False,sslen=2x3*8*1.2,mfccs=0.1-100-0.025,wg=0.002-64-0.01_properreader_residual
#       sslen to 10ms
#   *simple hand realistic: img=120x160x1,attention=12,hidden=1024,z=66,seq_len=12,n_rnn=3-3,v1=False,nv1write=3,cw=0.1,fs=22050,hearing=False,sslen=2x10*10*1.5,constphase=True,mfccs=100-0.025-0.002,wg=64-0.01-0.002
#       fs to 44100

# hearing nov1
#   apartment:
#   simple hand:



AUDIO_GEN_PARAMS = {
    'binaural': True,  # FIXME false is not implemented properly at the decoder, hearing part
    'varying_delta': True,  # whether modulations are constant or varying
    'nsoundstream': 3,
    'nmodulation': 4,  # number of modulations: number of sound sections
    'section_len_msec': 8,
    'attack_len_msec': 1,  # ramp-up length for each soundstream
    'decay_len_msec': 1,  # ramp-down length
    'soundscape_len_by_stream_len': 1.5,  # should be >=1, defines the amount of overlap between streams
    'const_phase': True
}

HEARING_PARAMS = {
    'models_used': {'mfccs': True,
                    'wavegan': True,
                    'tcn': False,
                    'carfac': False},  # carfac is highly experimental = doesn't scale/work really
    'mfcss_frame_len': 0.010,  # s
    'mfcss_frame_step': 0.002,  # s
    'mfcss_nceps': 100,  # number of coeffs, 13 for ASR, max 100
    'wg_nfilters': 64,
    'wg_kernel_len': 0.010,  # s
    'wg_strides': 0.002,  # s
    'hearing_repr_len': 512,
    'tcn_nlevels': 11,
    'tcn_nhidden': 32,
    'tcn_kernel_size': 2,
    'tcn_dropout': 0.1
}

DEFAULT_NETWORK_PARAMS = {
    'dtype': 'float32',
    'input_dim': (120, 160, 1),  # height, width, depth (1=grayscale, 3=RGB)
    'attention_n': 20,
    'n_hidden': 32,
    'n_z': 32,  # if z_indirection is false, this value will be overwritten to be the number of audio_gen params
    'z_indirection': False,  # if true, the number of Gaussian noised variables can be different from the number of audio_gen params
    'sequence_length': 4,
    'batch_size': 64,
    'n_rnn_cells': (2, 2),  # should have a weak decoder [and strong encoder]
    'learning_rate': 5e-5,  # initial learning rate
    'nonrecurrent_dec': False,
    'residual_encoder': True,  # only for the old models leave it true - e.g. v1 cheat 26seq 4mod
    'hearing_decoder': False,  # if false, audio_gen params are passed to the decoder raw, not the soundscape
    'v1_gaussian': True,  # whether lines should be drawn on the canvas instead of grids
    'n_v1_write': 3,  # ignored if v1_gaussian is false
    'kl_weight': 0.1,  # beta value on KL divergence, keep it around 0.1, 0.5
    'congr_weight': 0.1,  # congruence weight
    'fs': 16000,  # audio sampling freq; 22050 for hearing, 44100 for non-hearing
    'audio_gen': AUDIO_GEN_PARAMS,
    'hearing': HEARING_PARAMS
}


def load_config(cfg_id):

    # load from file
    # structure: {cfg_id: {cfg: <model_config>, models: [<list_of_long_model_names>]}}
    with open('configs.json', 'rt') as f:
        configs = json.load(f)

    if cfg_id not in configs:
        return DEFAULT_NETWORK_PARAMS

    # add missing parameters if any
    params = configs[cfg_id]['cfg']
    for missing in (DEFAULT_NETWORK_PARAMS.keys() - params.keys()):
        params[missing] = DEFAULT_NETWORK_PARAMS[missing]

    for missing in (DEFAULT_NETWORK_PARAMS['audio_gen'].keys() - params['audio_gen'].keys()):
        params['audio_gen'][missing] = DEFAULT_NETWORK_PARAMS['audio_gen'][missing]

    for missing in (DEFAULT_NETWORK_PARAMS['hearing'].keys() - params['hearing'].keys()):
        params['hearing'][missing] = DEFAULT_NETWORK_PARAMS['hearing'][missing]

    return params  # model names are not returned


def save_config(cfg_id, params, model_name):
    with open('configs.json', 'rt') as f:
        configs = json.load(f)

    if cfg_id not in configs:
        configs[cfg_id] = {'cfg': params, 'models': [model_name]}  # first model with this cfg
    elif model_name not in configs[cfg_id]['models']:  # add new model with existing cfg
        configs[cfg_id]['models'].append(model_name)

    with open('configs.json', 'wt') as f:
        json.dump(configs, f, indent=4)
