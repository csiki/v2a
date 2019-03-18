from aev2a import *
import sys
import cv2
import requests
import numpy as np
import pymatlab
from skimage.transform import resize
from collections import deque
import simpleaudio as saudio
from skimage.filters import sobel


# TODO test
# Matlab has to run in the background if the CORF edge detection algo is used
# Run this script after starting IP Webcam app on your Android phone
#     USB tethering has to be turned on and the phone connected to PC via USB
#     Both mobile data and wifi should be turned off on the phone
# Usage: python3.6 run_proto.py test|fast corf|sobel mobile_ip cfg_id model_name_postfix


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('WRONG ARGUMENTS, CHECK SCRIPT FOR DETAILS!', file=sys.stderr)
        exit(1)

    # params
    test_run = sys.argv[1] == 'test'
    edge_detection = sys.argv[2].lower()  # corf | sobel for now; corf is the more sophisticated algo, but requires matlab to run
    mobile_ip = sys.argv[3]
    config_id = sys.argv[4] if len(sys.argv) > 4 else 'default'  # have to be defined in configs.json
    model_name_postfix = sys.argv[5] if len(sys.argv) > 5 else ''

    shot_url = "http://" + mobile_ip + ":8080/shot.jpg"

    network_params = load_config(config_id)
    network_params['batch_size'] = 1
    model_name = find_model(config_id, model_name_postfix)
    sound_len = audio_gen.soundscape_len(network_params['audio_gen'], network_params['fs'])
    model_root = 'training/'

    # build matlab session (if the matlab edge detection is used)
    if edge_detection == 'corf':
        matlab_session = pymatlab.session_factory()
        matlab_session.run('addpath matlab/faster_corf')  # TODO test
        print('MATLAB SESSION ESTABLISHED')

    # build V2A model
    model = Draw(network_params, model_name_postfix, logging=False, training=False)
    model.prepare_run_single(model_root + model_name)
    print('MODEL IS BUILT')

    # prepare audio
    fs = network_params['fs']
    nchannel = 2 if network_params['audio_gen']['binaural'] else 1

    # start streaming and convertimg images
    run_times = deque([0, 0, 0], maxlen=3)
    play_obj = None
    sound_start = time.time() - 100
    try:
        play_obj = None
        while True:

            if (time.time() - sound_start) < (sound_len - np.mean(run_times)):  # play_obj and play_obj.is_playing():
                # time.sleep(0.00001)
                continue

            comp_start = time.time()

            # img mobile -> pc
            img_resp = requests.get(shot_url)
            img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
            img = cv2.imdecode(img_arr, cv2.IMREAD_GRAYSCALE)
            img = resize(img, (network_params['input_dim'][0], network_params['input_dim'][1]), anti_aliasing=True, preserve_range=False)

            # img -> contour
            if edge_detection == 'corf':
                matlab_session.putvalue('ain', img)
                matlab_session.run('a = convertImage(ain);')  # 0.15 sec
                contour = matlab_session.getvalue('a')
            elif edge_detection == 'sobel':
                contour = sobel(img)
            else:
                raise ValueError('Edge detection algorithm can either be "corf" or "sobel", not {}!'.format(edge_detection))

            # contour -> sound
            if test_run:
                soundscape, gen_img = model.run_single(contour, test_run)  # 0.4 rt
            else:
                soundscape = model.run_single(contour, test_run)
            soundscape = np.int16(soundscape / np.max(np.abs(soundscape)) * 32767)
            sound_start = time.time()  # time when it would start

            while play_obj and play_obj.is_playing():
                time.sleep(0.000001)
            play_obj = saudio.play_buffer(soundscape, 2, 2, 44100)

            # measure time
            run_times.append(sound_start - comp_start)

            # logging/plotting
            if test_run:
                print(np.mean(run_times))  # should be around 0.4 sec latency
                cv2.imshow("AndroidCam", img)
                cv2.imshow("CORFCam", contour)
                cv2.imshow("GenImgCam", gen_img)
                if cv2.waitKey(1) == 27:
                    break
    finally:
        pass
