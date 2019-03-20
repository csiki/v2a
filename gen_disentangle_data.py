from aev2a import *
import sys
import tables
import numpy as np

# run as python3.6 gen_disentangle_data.py cfg_name test|train rand|seq model_name_postfix


def img_to_uint(img):
    if np.min(img) < 0:
        img -= np.min(img)  # if there is guaranteed to be black on the image, this works
    img /= np.max(img)
    img *= 255
    return img.astype(np.uint8)


if __name__ == '__main__':

    config_id = sys.argv[1] if len(sys.argv) > 1 else 'default'  # have to be defined in configs.json
    dataset = sys.argv[2] if len(sys.argv) > 2 else 'data/simple_hand.hdf5'  # path to dataset, default can be downloaded
    test_set = sys.argv[3] == 'test' if len(sys.argv) > 3 else True  # training by default
    rand_select = sys.argv[4] == 'rand' if len(sys.argv) > 4 else True
    model_name_postfix = sys.argv[5] if len(sys.argv) > 5 else ''  # if having more models with the same config

    network_params = load_config(config_id)
    network_params['batch_size'] = 1
    model_name = find_model(config_id, model_name_postfix)
    sound_len = audio_gen.soundscape_len(network_params['audio_gen'], network_params['fs'])
    model_root = 'training/'

    RIGHT_BTN = ord('d')
    LEFT_BTN = ord('a')

    pprint(network_params)

    model = Draw(network_params, model_name_postfix, logging=False, training=False)
    model.prepare_run_single(model_root + model_name)
    print('MODEL IS BUILT')

    # load dataset
    data = model.get_data(dataset)
    dataptr = data.root.test_img if test_set else data.root.train_img
    batch_size = 64

    # test model image by image
    batch_round = 0
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
    hdf5_file = tables.open_file('data/gendata_' + config_id + set_text + '.hdf5', mode='w')
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

    while True:
        # select image
        if rand_select:
            batch = model.get_batch(dataptr, batch_size=batch_size)
        else:
            if (batch_round+1) * batch_size > dataptr.shape[0]:
                batch_round = 1
                break  # out
            indices = np.arange(batch_round * batch_size, (batch_round + 1) * batch_size)
            batch = model.get_batch(dataptr, indices=indices)

        # run model
        cs, inp_imgs, gen_imgs, soundscapes, ss_tensors, wr_tensors = model.sess.run([model.cs, model.images, model.generated_images,
                                                                                      model.whole_soundscape, model.soundscape_tensors,
                                                                                      model.wr_attn_params],
                                                                                     feed_dict={model.images: batch})
        cs = 1.0 / (1.0 + np.exp(-np.array(cs)))  # nseq x batch x height x width
        cs = np.reshape(cs, [model.sequence_length, batch_size, model.img_h, model.img_w])
        cs = np.transpose(cs, [1, 0, 2, 3])

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

        print('.', end='', flush=True)

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

        batch_round += 1
    hdf5_file.close()
