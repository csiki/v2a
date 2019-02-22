from aev2a import *
import sys
import cv2
import requests
import numpy as np
import pymatlab
from skimage.transform import resize
from skimage import feature
from collections import deque
from scipy.io import wavfile
import matplotlib.pyplot as plt

import simpleaudio as saudio
# from wave import open as waveOpen
from ossaudiodev import open as ossOpen
from ossaudiodev import AFMT_S16_NE
import threading


def play_sound(soundscape):  # FIXME depricated, too low level
    dsp = ossOpen('/dev/dsp', 'w')
    dsp.setparameters(AFMT_S16_NE, nchannel, fs)
    dsp.write(soundscape.tobytes())
    dsp.close()


# good ones:
#    final-nov1-noh-24seq-2ss-4mod
#    v1-extra-26seq-4mod-cheat - USED IN THESIS
# apartment:
#    ap-nov1-extra-30seq2
#    ap-nov1-long-final-extra
# table:
#    table3-nov1-8seq - USED IN THESIS
#    table3-nov1-8seq-zind


# scp -r tothv1@triton.aalto.fi:/scratch/work/tothv1/v2a/autoencoder/draw-color/configs .
CFG_TO_MODEL = {  # TODO rewrite to be loaded from a file, to which a record is added every time a training starts
    'final-nov1-noh-24seq-2ss-4mod':    'img=120x160x1,attention=20,hidden=1024,z=30,seq_len=24,n_rnn=3-3,v1=False,nv1write=3,cw=0.142,fs=44100,hearing=False,sslen=2x4*8*1.2,constphase=True,mfccs=100-0.025-0.002,wg=64-0.01-0.002_final',
    # 'ap-final-nov1-noh-24seq-2ss-4mod': 'img=120x160x1,attention=20,hidden=1024,z=30,seq_len=24,n_rnn=3-3,v1=False,nv1write=3,cw=0.142,fs=44100,hearing=False,sslen=2x4*8*1.2,constphase=True,mfccs=100-0.025-0.002,wg=64-0.01-0.002_final_apartment',
    # 'final-v1-noh-24seq-2ss-4mod':      'img=120x160x1,attention=20,hidden=1024,z=24,seq_len=24,n_rnn=3-3,v1=True,nv1write=2,cw=0.142,fs=44100,hearing=False,sslen=2x4*8*1.2,constphase=True,mfccs=100-0.025-0.002,wg=64-0.01-0.002_final',
    'v1-extra-26seq-4mod-cheat': 'img=120x160x1,attention=20,hidden=1024,z=45,seq_len=26,n_rnn=3-3,v1=True,nv1write=3,cw=0.1,fs=44100,hearing=False,sslen=3x4*8*1.2,constphase=True,mfccs=100-0.025-0.002,wg=64-0.01-0.002_final_extra_v1_cheat',
    # 'ap-nov1-extra-30seq2': 'img=120x160x1,attention=20,hidden=1024,z=36,seq_len=30,n_rnn=3-3,v1=False,nv1write=3,cw=0.1,fs=44100,hearing=False,sslen=3x3*8*1.5,constphase=True,mfccs=100-0.025-0.002,wg=64-0.01-0.002_final_extra_apartment2',
    'nov1-extra-4mod': 'img=120x160x1,attention=20,hidden=1024,z=45,seq_len=20,n_rnn=3-3,v1=False,nv1write=3,cw=0.1,fs=44100,hearing=False,sslen=3x4*8*1.5,constphase=True,mfccs=100-0.025-0.002,wg=64-0.01-0.002_final_extra',
    # 'v1-extra-4mod': 'img=120x160x1,attention=20,hidden=1024,z=45,seq_len=20,n_rnn=3-3,v1=True,nv1write=3,cw=0.1,fs=44100,hearing=False,sslen=3x4*8*1.5,constphase=True,mfccs=100-0.025-0.002,wg=64-0.01-0.002_final_extra',
    # 'maxifast-v1-2': 'img=120x160x1,attention=20,hidden=1024,z=1024,seq_len=14,n_rnn=4-2,v1=True,nv1write=4,cw=0.1,fs=44100,hearing=False,sslen=4x4*5*1.5,constphase=True,mfccs=64-0.032-0.008,wg=64-0.01-0.002_maxifast_2',
    # 'maxifast-nov1-2': 'img=120x160x1,attention=20,hidden=1024,z=1024,seq_len=14,n_rnn=4-2,v1=False,nv1write=4,cw=0.1,fs=44100,hearing=False,sslen=4x4*5*1.5,constphase=True,mfccs=64-0.032-0.008,wg=64-0.01-0.002_maxifast_2nov1',
    # 'ideal-v1-hand': 'img=120x160x1,attention=16,hidden=1024,z=54,seq_len=20,n_rnn=4-1,v1=True,nv1write=3,cw=0.1,fs=44100,hearing=False,sslen=3x5*8*1.2,constphase=True,mfccs=100-0.025-0.002,wg=64-0.01-0.002_ideal',
    # 'ap-nov1-long-final-extra': 'img=120x160x1,attention=12,hidden=1024,z=45,seq_len=42,n_rnn=3-3,v1=False,nv1write=3,cw=0.1,fs=44100,hearing=False,sslen=3x4*10*1.5,constphase=True,mfccs=100-0.025-0.002,wg=64-0.01-0.002_final_extra_long_apartment',
    'table3-v1-8seq': 'img=120x160x1,attention=20,hidden=1024,z=128,seq_len=8,n_rnn=3-3,v1=True,nv1write=3,cw=1.0,fs=44100,hearing=False,sslen=3x4*10*1.5,constphase=True,mfccs=100-0.01-0.002,wg=64-0.01-0.002_table3',
    'table3-nov1-8seq': 'img=120x160x1,attention=20,hidden=1024,z=128,seq_len=8,n_rnn=3-3,v1=False,nv1write=3,cw=1.0,fs=44100,hearing=False,sslen=3x4*10*1.5,constphase=True,mfccs=100-0.01-0.002,wg=64-0.01-0.002_table3',
    'table3-nov1-8seq-zind': 'img=120x160x1,attention=20,hidden=1024,z=128,seq_len=8,n_rnn=3-3,v1=False,nv1write=3,cw=1.0,fs=44100,hearing=False,sslen=3x4*10*1.5,constphase=True,mfccs=100-0.01-0.002,wg=64-0.01-0.002_table3_zind'
}

CFG_TO_SOUND_LEN = {  # TODO save this in file too, no manual tracking
    'final-v1-noh-30seq-3ss':           30*3*8*1.2 / 1000,
    'final-v1-noh-30seq-2ss':           30*3*8*1.2 / 1000,
    'final-nov1-noh-30seq-2ss':         30*3*8*1.2 / 1000,
    'final-nov1-noh-24seq-2ss-4mod':    24*4*8*1.2 / 1000,
    'ap-final-nov1-noh-24seq-3ss':      24*3*8*1.2 / 1000,
    'ap-final-nov1-noh-30seq-2ss':      30*3*8*1.2 / 1000,
    'ap-final-nov1-noh-24seq-2ss-4mod': 24*4*8*1.2 / 1000,
    'final-v1-noh-24seq-2ss-4mod':      24*4*8*1.2 / 1000,
    'v1-extra-26seq-4mod-cheat':        26*4*8*1.2 / 1000,
    'ap-nov1-extra-30seq2':             30*3*8*1.5 / 1000,
    'maxifast-v1-2':                    14*4*5*1.5 / 1000,
    'maxifast-nov1-2':                  14*4*5*1.5 / 1000,
    'ideal-v1-hand':                    20*5*8*1.2 / 1000,
    'ap-nov1-long-final-extra':         42*4*10*1.5 / 1000,
    'table3-v1-8seq':                   8*4*10*1.5 / 1000,
    'table3-nov1-8seq':                 8*4*10*1.5 / 1000,
    'table3-nov1-8seq-zind':            8*4*10*1.5 / 1000
}

# TODO more hints on how to run
# if model is not downloaded yet, run dl_and_test.sh
# matlab has to run, start it from console like: matlab &
# run this script after starting IP Webcam app on mobile
#     USB tethering has to be turned on + mobile connected to PC
#     both mobile data and wifi should be turned off
# usage: python3.6 run_proto.py <cfg_name> <mobile_ip> [test:optional]
if __name__ == '__main__':
    argv = sys.argv
    if len(sys.argv) < 3:
        print('WRONG ARGUMENTS, CHECK SCRIPT FOR DETAILS!', file=sys.stderr)
        # argv = ['', 'final-v1-noh-30seq-3ss', '192.168.42.129', 'test']
        exit(1)

    # params
    config_name = argv[1]
    mobile_ip = argv[2]
    test = True if len(argv) > 3 and argv[3] == 'test' else False
    shot_url = "http://" + mobile_ip + ":8080/shot.jpg"
    model_name = CFG_TO_MODEL[config_name]
    model_root = '/media/viktor/0C22201D22200DF0/triton/triton_training/training/'  # FIXME load from global config file

    # build matlab session
    matlab_session = pymatlab.session_factory()  # TODO load needed CORF matlab functions (or cd to dir)
    print('MATLAB SESSION ESTABLISHED')

    # build V2A model
    nepoch = 10000
    img_h = 120
    img_w = 160
    num_colors = 1
    v1_activation = False  # whether to load the matlab txt files or jpegs
    crop_img = False
    grayscale = True  # if true, 2D image is fed, 3D otherwise; set true when feeding 1 layer of CORF3D
    only_layer = None
    complement = False

    os.system('play ~/bin/example.wav')  # TODO rm

    network_params = load_config(config_name)
    network_params['batch_size'] = 1

    model = Draw(nepoch, img_h, img_w, num_colors, grayscale, network_params,
                 logging=False, log_after=1000, save_after=2000, training=False)
    model.prepare_run_single(model_root + model_name)
    print('MODEL IS BUILT')

    # prepare audio
    fs = network_params['fs']
    nchannel = 2 if network_params['audio_gen']['binaural'] else 1
    sound_len = CFG_TO_SOUND_LEN[config_name]
    sound_thread = None

    # start streaming and convertimg images
    run_times = deque([0, 0, 0], maxlen=3)
    play_obj = None
    sound_start = time.time() - 100
    try:
        play_obj = None
        while True:

            if (time.time() - sound_start) < (sound_len - np.mean(run_times)):  # play_obj and play_obj.is_playing():
                # time.sleep(0.0001)
                continue

            comp_start = time.time()

            # img mobile -> pc
            img_resp = requests.get(shot_url)
            img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
            img = cv2.imdecode(img_arr, cv2.IMREAD_GRAYSCALE)
            img = resize(img, (img_h, img_w), anti_aliasing=True, preserve_range=False)

            # img -> corf
            matlab_session.putvalue('ain', img)
            matlab_session.run('a = convertImage(ain);')  # 0.15 sec
            corf = matlab_session.getvalue('a')
            # corf = feature.canny(img, sigma=1.).astype(float)  # supafast

            # corf -> sound
            if test:
                soundscape, gen_img = model.run_single(corf, test)  # 0.4 rt
            else:
                soundscape = model.run_single(corf, test)
            soundscape = np.int16(soundscape / np.max(np.abs(soundscape)) * 32767)
            # _, soundscape = wavfile.read('../app/example.wav')  # for testing
            sound_start = time.time()  # time when it would start
            #play_obj = saudio.play_buffer(soundscape, 2, 2, 44100)  # slower
            #if sound_thread and sound_thread.isAlive():
            #    sound_thread.join()

            while play_obj and play_obj.is_playing():
                time.sleep(0.000001)
            #sound_thread = threading.Thread(target=play_sound, args=(soundscape,))  #, daemon=True)  # 0.42 time
            play_obj = saudio.play_buffer(soundscape, 2, 2, 44100)
            #sound_thread.start()

            # measure time
            run_times.append(sound_start - comp_start)

            # logging/plotting
            if test:
                print(np.mean(run_times))  # should be around 0.4 sec latency
                cv2.imshow("AndroidCam", img)
                cv2.imshow("CORFCam", corf)
                cv2.imshow("GenImgCam", gen_img)
                if cv2.waitKey(1) == 27:
                    break
    finally:
        pass
