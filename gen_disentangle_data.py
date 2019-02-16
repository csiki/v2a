from aev2a import *
import sys
import tables
import numpy as np
import pickle
from run_proto import CFG_TO_MODEL, CFG_TO_SOUND_LEN

# run as python3.6 test_model.py <cfg_name> [<test|train>] [<rand|seq>]


def img_to_uint(img):
    if np.min(img) < 0:
        img -= np.min(img)  # if there is guaranteed to be black on the image, this works
    img /= np.max(img)
    img *= 255
    return img.astype(np.uint8)


if __name__ == '__main__':
    argv = sys.argv
    config_name = argv[1] if len(argv) > 1 else 'table3-nov1-8seq-zind'  # table3-nov1-8seq, v1-extra-26seq-4mod-cheat
    test_set = argv[2] == 'test' if len(argv) > 2 else True
    rand_select = argv[3] == 'rand' if len(argv) > 3 else False

    dataset = '/media/viktor/0C22201D22200DF0/hand_gestures/simple_hand.hdf5'
    if 'ap-' in config_name:
        dataset = '/media/viktor/0C22201D22200DF0/hand_gestures/apartment.hdf5'
    elif 'table' in config_name:
        dataset = '/media/viktor/0C22201D22200DF0/hand_gestures/table3.hdf5'
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
    pprint(network_params)

    model = Draw(nepoch, img_h, img_w, num_colors, v1_activation, crop_img, grayscale, network_params, logging=False,
                 only_layer=only_layer, img_complement=complement, log_after=1000, save_after=2000, training=False)
    model.prepare_run_single(model_root + model_name)
    print('MODEL IS BUILT')

    # load dataset
    data, is_hdf5 = model.get_data(dataset)
    dataptr = data.root.test_img if test_set else data.root.train_img
    batch_size = 64

    # test model image by image
    batch_round = 0
    # data_to_save = []
    nseq = network_params['sequence_length']
    nsoundstream = network_params['audio_gen']['nsoundstream']
    n_v1_write = model.n_v1_write
    v1_gaussian = network_params['v1_gaussian']
    section_len = int(network_params['audio_gen']['section_len_msec'] / 1000. * network_params['fs'])
    nmodulation = network_params['audio_gen']['nmodulation']
    soundstream_len = int(nmodulation * section_len)
    soundscape_len = int(soundstream_len * network_params['audio_gen']['soundscape_len_by_stream_len']) * nseq
    float_dtype = tables.Float32Atom()
    ss_dtype = tables.Int16Atom()
    img_dtype = tables.UInt8Atom()

    set_text = '_test' if test_set else '_train'
    hdf5_file = tables.open_file('/media/viktor/0C22201D22200DF0/hand_gestures/' + 'gendata_' + config_name + set_text + '.hdf5', mode='w')
    cs_storage = hdf5_file.create_earray(hdf5_file.root, 'cs', img_dtype, shape=[0, nseq, model.img_h, model.img_w])
    if test_set:
        ss_storage = hdf5_file.create_earray(hdf5_file.root, 'soundscapes', ss_dtype, shape=[0, soundscape_len, 2])
    img_storage = hdf5_file.create_earray(hdf5_file.root, 'gen_img', img_dtype, shape=[0, model.img_h, model.img_w])
    inp_img_storage = hdf5_file.create_earray(hdf5_file.root, 'inp_img', img_dtype, shape=[0, model.img_h, model.img_w])
    df_storage = hdf5_file.create_earray(hdf5_file.root, 'df', float_dtype, shape=[0, nseq, nsoundstream, soundstream_len])
    da_storage = hdf5_file.create_earray(hdf5_file.root, 'da', float_dtype, shape=[0, nseq, nsoundstream, soundstream_len])
    dazim_storage = hdf5_file.create_earray(hdf5_file.root, 'dazim', float_dtype, shape=[0, nseq, nsoundstream, soundstream_len])
    gx_storage = hdf5_file.create_earray(hdf5_file.root, 'gx', float_dtype, shape=[0, nseq, n_v1_write])
    gy_storage = hdf5_file.create_earray(hdf5_file.root, 'gy', float_dtype, shape=[0, nseq, n_v1_write])
    delta_storage = hdf5_file.create_earray(hdf5_file.root, 'delta', float_dtype, shape=[0, nseq, n_v1_write])

    raw_df_storage = hdf5_file.create_earray(hdf5_file.root, 'raw_df', float_dtype, shape=[0, nseq, nsoundstream, nmodulation])
    raw_da_storage = hdf5_file.create_earray(hdf5_file.root, 'raw_da', float_dtype, shape=[0, nseq, nsoundstream, nmodulation])
    raw_dazim_storage = hdf5_file.create_earray(hdf5_file.root, 'raw_dazim', float_dtype, shape=[0, nseq, nsoundstream, nmodulation])
    if v1_gaussian:
        angle_storage = hdf5_file.create_earray(hdf5_file.root, 'angle', float_dtype, shape=[0, nseq, n_v1_write])
    # with open('gendata_' + config_name + '.pickle', 'wb') as f:
    # pickle_f = open('gendata_' + config_name + '.pickle', 'wb')
    while True:
        # select image
        if rand_select:
            batch = model.get_batch(dataptr, is_hdf5, batch_size=batch_size)
        else:
            if (batch_round+1) * batch_size > dataptr.shape[0]:
                batch_round = 1
                break  # out
            indices = np.arange(batch_round * batch_size, (batch_round + 1) * batch_size)
            batch = model.get_batch(dataptr, is_hdf5, indices=indices)
        # batch = np.expand_dims(batch, 0)

        # run model
        cs, inp_imgs, gen_imgs, soundscapes, ss_tensors, wr_tensors = model.sess.run([model.cs, model.images, model.generated_images,
                                                                                        model.whole_soundscape, model.soundscape_tensors,
                                                                                        model.wr_attn_params],
                                                                                       feed_dict={model.images: batch})
        cs = 1.0 / (1.0 + np.exp(-np.array(cs)))  # nseq x batch x height x width
        cs = np.reshape(cs, [model.sequence_length, batch_size, model.img_h, model.img_w])
        cs = np.transpose(cs, [1, 0, 2, 3])
        # soundscape, gen_img, cs = soundscapes[0], np.reshape(gen_imgs[0], [model.img_h, model.img_w]), cs

        # ss_tensors: nseq x [dict] x batch x ...
        # we need df, da, dazim in the form of [dict] x batch x nseq x ...
        df_shape = ss_tensors[0]['df'].shape
        da_shape = ss_tensors[0]['da'].shape
        dazim_shape = ss_tensors[0]['dazim'].shape
        df = np.zeros([batch_size, nseq, nsoundstream, soundstream_len], dtype=model.npdtype)
        da = np.zeros([batch_size, nseq, nsoundstream, soundstream_len], dtype=model.npdtype)
        dazim = np.zeros([batch_size, nseq, nsoundstream, soundstream_len], dtype=model.npdtype)

        raw_df = np.zeros([batch_size, nseq, nsoundstream, nmodulation], dtype=model.npdtype)
        raw_da = np.zeros([batch_size, nseq, nsoundstream, nmodulation], dtype=model.npdtype)
        raw_dazim = np.zeros([batch_size, nseq, nsoundstream, nmodulation], dtype=model.npdtype)
        # wr_tensors are similar: nseq x ndrawing x [gx, gy, delta] x batch x 1
        # reshape into the form of: [gx, gy, delta] x batch x nseq x ndrawing

        gx = np.zeros([batch_size, len(wr_tensors), n_v1_write], dtype=model.npdtype)
        gy = np.zeros([batch_size, len(wr_tensors), n_v1_write], dtype=model.npdtype)
        delta = np.zeros([batch_size, len(wr_tensors), n_v1_write], dtype=model.npdtype)
        if v1_gaussian:
            angle = np.zeros([batch_size, len(wr_tensors), n_v1_write], dtype=model.npdtype)

        for i_seq in range(len(ss_tensors)):
            df[:, i_seq] = ss_tensors[i_seq]['df']
            da[:, i_seq] = ss_tensors[i_seq]['da']
            dazim[:, i_seq] = ss_tensors[i_seq]['dazim']

            raw_df[:, i_seq] = ss_tensors[i_seq]['raw_df']
            raw_da[:, i_seq] = ss_tensors[i_seq]['raw_da']
            raw_dazim[:, i_seq] = ss_tensors[i_seq]['raw_dazim']
            for d in range(n_v1_write):
                if v1_gaussian:
                    gx[:, i_seq, d] = np.reshape(wr_tensors[i_seq][d][0], [batch_size])
                    gy[:, i_seq, d] = np.reshape(wr_tensors[i_seq][d][1], [batch_size])
                    delta[:, i_seq, d] = np.reshape(wr_tensors[i_seq][d][2], [batch_size])
                    angle[:, i_seq, d] = np.reshape(wr_tensors[i_seq][d][3], [batch_size])
                else:  # only one drawing per round
                    gx[:, i_seq, d] = np.reshape(wr_tensors[i_seq][0], [batch_size])
                    gy[:, i_seq, d] = np.reshape(wr_tensors[i_seq][1], [batch_size])
                    delta[:, i_seq, d] = np.reshape(wr_tensors[i_seq][2], [batch_size])
                    # angle[:, i_seq, d] = np.reshape(wr_tensors[i_seq][3], [batch_size])

        # ss_tensors_realigned = {'df': df, 'da': da, 'dazim': dazim}
        # wr_tensors_realigned = {'gx': gx, 'gy': gy, 'delta': delta}

        # print(ss_tensors_realigned, wr_tensors_realigned)
        print('.', end='', flush=True)

        # for i in range(batch_size):
        if True:
            # record = {}
            # record['cs'] = cs[i]
            # record['gen_img'] = np.reshape(gen_imgs[i], [model.img_h, model.img_w])
            # record['soundscape'] = np.int16(soundscapes[i] / np.max(np.abs(soundscapes[i])) * 32767)
            # record['ss_tensors'] = {k: v[i] for k, v in ss_tensors_realigned.items()}
            # record['wr_tensors'] = {k: v[i] for k, v in wr_tensors_realigned.items()}

            cs_storage.append(img_to_uint(cs))
            img_storage.append(img_to_uint(np.reshape(gen_imgs, [batch_size, model.img_h, model.img_w])))
            inp_img_storage.append(img_to_uint(np.reshape(inp_imgs, [batch_size, model.img_h, model.img_w])))
            if test_set:
                ss_storage.append(np.int16(soundscapes / np.max(np.abs(soundscapes)) * 32767))
            df_storage.append(df)
            da_storage.append(da)
            dazim_storage.append(dazim)

            raw_df_storage.append(raw_df)
            raw_da_storage.append(raw_da)
            raw_dazim_storage.append(raw_dazim)

            gx_storage.append(gx)
            gy_storage.append(gy)
            delta_storage.append(delta)
            if v1_gaussian:
                angle_storage.append(angle)

            # pickle.dump(record, pickle_f)
            # data_to_save.append(record)

        batch_round += 1

    # with open('gendata_' + config_name + '.pickle', 'wb') as f:
    #     pickle.dump(data_to_save, f)
    hdf5_file.close()
