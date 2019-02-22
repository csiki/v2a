from aev2a import *
import sys
import cv2
import numpy as np
import simpleaudio as saudio
from run_proto import CFG_TO_MODEL, CFG_TO_SOUND_LEN


# shows image and corresponding sound
# select to test images from the training or test set
# select whether to choose random images from the set, or sequence from the beginning
# right arrow moves to the next image (random or next in sequence), left arrow moves to previous, esc exits
# should draw iteration by iteration and play sound at the same time
# usage: python test_model.py <cfg_name> [<test|train>] [<rand|seq>]

# TODO record key strokes with the input (and output) image included so you can perform accuracy tests; could also record reaction time

if __name__ == '__main__':
    argv = sys.argv
    config_name = argv[1]
    test_set = argv[2] == 'test' if len(argv) > 2 else True
    rand_select = argv[3] == 'rand' if len(argv) > 3 else True

    dataset = '/media/viktor/0C22201D22200DF0/hand_gestures/simple_hand.hdf5'
    if 'ap-' in config_name:
        dataset = '/media/viktor/0C22201D22200DF0/hand_gestures/apartment.hdf5'
    model_name = CFG_TO_MODEL[config_name]
    model_root = '/media/viktor/0C22201D22200DF0/triton/triton_training/training/'
    sound_len = CFG_TO_SOUND_LEN[config_name]

    RIGHT_BTN = ord('d')
    LEFT_BTN = ord('a')

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

    network_params = load_config(config_name)
    network_params['batch_size'] = 1
    pprint(network_params)

    model = Draw(nepoch, img_h, img_w, num_colors, grayscale, network_params,
                 logging=False, log_after=1000, save_after=2000, training=False)
    model.prepare_run_single(model_root + model_name)
    print('MODEL IS BUILT')

    # load dataset
    data = model.get_data(dataset)
    dataptr = data.root.test_img if test_set else data.root.train_img
    loadnsamples = 64

    # test model image by image
    batch_round = 0
    while True:
        # select image
        if rand_select:
            batch = model.get_batch(dataptr, batch_size=loadnsamples)
        else:
            if (batch_round+1) * loadnsamples > dataptr.shape[0]:
                batch_round = 1
            indices = np.arange(batch_round * loadnsamples, (batch_round+1) * loadnsamples)
            batch = model.get_batch(dataptr, indices=indices)

        # iterate through the batch
        img_i = 0
        while img_i < loadnsamples:
            # run model
            soundscape, gen_img, cs = model.run_single(batch[img_i], canvas_imgs_needed=True)
            soundscape = np.int16(soundscape / np.max(np.abs(soundscape)) * 32767)
            cv2.imshow("Original", batch[img_i])
            if cv2.waitKey(1) == 27:
                print('EXITED'); exit(0)
            cv2.imshow("Generated", gen_img)
            if cv2.waitKey(1) == 27:
                print('EXITED'); exit(0)

            # repeat sound and drawing
            play_obj = None
            img_same = True
            while img_same:
                while play_obj and play_obj.is_playing():
                    time.sleep(0.000001)

                play_obj = saudio.play_buffer(soundscape, 2, 2, 44100)

                for i in range(model.sequence_length):
                    cv2.imshow("Decoded", cs[i])
                    c = cv2.waitKey(1)
                    if c == 27:
                        print('EXITED'); exit(0)
                    elif c == RIGHT_BTN:
                        img_i += 1; img_same = False
                        print('next image', img_i); break
                    elif c == LEFT_BTN and img_i > 0:
                        img_i -= 1; img_same = False
                        print('prev image', img_i); break
                    time.sleep(sound_len / model.sequence_length)
