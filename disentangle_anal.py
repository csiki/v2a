import tables, csv
import matplotlib.pyplot as plt
import numpy as np
import sys, time, math
from PIL import Image, ImageOps
import cv2
from scipy.stats.stats import pearsonr
from sklearn.metrics import mutual_info_score
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from numpy import pi


def img_labeling(imgs, test_set, out_name):
    labels = []
    for i, img in enumerate(imgs):
        img_same = True
        d = 0
        print('DATA', i)
        while img_same:
            cv2.imshow("GenImgCam", img)
            c = cv2.waitKey(1)
            if c == 27:  # esc
                labels.append(d)
                print(d)
                img_same = False
            elif c != -1:
                print(d, '+', c - 48)
                d = d * 10 + c - 48

    if not test_set:
        labels = np.arange(len(imgs))
    with open(out_name + set_text + '.csv', 'wt') as f:
        writer = csv.writer(f)
        writer.writerows([[l] for l in labels])


def create_sprite(imgs, grid, image_height, image_width, sprite_img_path):
    big_image = Image.new(
        mode='RGB',
        size=(image_width * grid, image_height * grid),
        color=(0, 0, 0, 0))  # fully transparent

    for i in range(len(imgs)):
        row = int(i / grid)
        col = int(i % grid)
        img = ImageOps.invert(Image.fromarray(imgs[i]))
        img = img.resize((image_height, image_width), Image.ANTIALIAS)
        row_loc = int(row * image_height)
        col_loc = int(col * image_width)
        big_image.paste(img, (col_loc, row_loc))  # NOTE: the order is reverse due to PIL saving
        # print(row_loc, col_loc)

    big_image.save(sprite_img_path)


def embed_features(sound_features, embedding_name, meta_name, sprite_img_path, image_height, image_width):
    embedding_var = tf.Variable(sound_features, name='sound_features')

    with tf.Session() as sess:
        # Create summary writer.
        writer = tf.summary.FileWriter('./graphs/' + embedding_name + set_text, sess.graph)
        # Initialize embedding_var
        sess.run(embedding_var.initializer)
        # Create Projector config
        config = projector.ProjectorConfig()
        # Add embedding visualizer
        embedding = config.embeddings.add()
        # Attache the name 'embedding'
        embedding.tensor_name = embedding_var.name
        # Metafile which is described later
        embedding.metadata_path = './' + meta_name + set_text + '.csv'
        embedding.sprite.image_path = './' + sprite_img_path
        embedding.sprite.single_image_dim.extend([image_height, image_width])
        # Add writer and config to Projector
        projector.visualize_embeddings(writer, config)
        # Save the model
        saver_embed = tf.train.Saver([embedding_var])
        saver_embed.save(sess, './graphs/' + embedding_name + set_text + '/' + embedding_name + set_text + '.ckpt', 1)

    writer.close()


# analysis:
# 1) can/gear position compared to azim and freq values
#    compare the average nonblack pixel position to avg azim and freq values
# 2) compare hand Gaussian patches (gx, gy, delta) to average sound qualities (da, df, dazim)
# 3) compute mutual information between da, df, dazim and angle of drawing
# 4) t-sne sound features, label them with images in tensorboard
#    embed concat of raw soundscape qualities, assign them to reconstructed hand images
# 5) [cluster raw sound features and check which clusters, with what kind of sounds belong to different binned angles]
#    OR plot a 3D function of sound features, connecting the lines on the angle axis,
#      so you can see how it changes with different angles visually
# 6) cluster sound properties of cans vs gears from the table dataset
# 7) plot distribution of table dataset sound properties, can vs gear, play sounds from each group

ANAL = {'1': False, '2': False, '3': False, '4': False, '5': False, '4/labeling': False, '4/create_sprite': False,
        '5/draw_hand': False, '5/hist': False, '5/lame_plot': False, '6': True, '6/labeling': False,
        '6/create_sprite': False, '6/embed': False, '7': False, '8': True}

config_id = sys.argv[1] if len(sys.argv) > 1 else 'default'  # have to be defined in configs.json
test_set = sys.argv[2] == 'test' if len(sys.argv) > 2 else True
table_cfg = sys.argv[3] if len(sys.argv) > 3 else 'default'
hand_cfg = sys.argv[4] if len(sys.argv) > 4 else 'default'

set_text = '_test' if test_set else '_train'
table_data_path = 'data/gendata_' + table_cfg + set_text + '.hdf5'
hand_data_path = 'data/gendata_' + hand_cfg + set_text + '.hdf5'
table_data_file = tables.open_file(table_data_path, mode='r')
hand_data_file = tables.open_file(hand_data_path, mode='r')

print(table_data_file)
print(hand_data_file)

if ANAL['1']:
    # 1)
    # avg nonblack pixel pos, delta
    table_imgs = np.array(table_data_file.root.gen_img)
    avg_obj_pos = []
    for img in table_imgs:
        nonblack = np.array(np.where(img > 10))
        avg_obj_pos.append(np.mean(nonblack, axis=1))
    avg_obj_pos = np.array(avg_obj_pos)
    delta = np.array(table_data_file.root.delta)
    delta_avg = delta.mean(axis=(1,2))

    # gather avg da, df, dazim
    da = np.array(table_data_file.root.da)
    df = np.array(table_data_file.root.df)
    dazim = np.array(table_data_file.root.dazim)
    # dazim[dazim > 0.8] = 0
    print('dazim minmaxmean', dazim.min(), dazim.max(), dazim.mean())

    avg_da = da.mean(axis=(1,2,3))
    avg_df = df.mean(axis=(1,2,3))
    avg_dazim = dazim.mean(axis=(1,2,3))

    print('lin y-df', pearsonr(avg_obj_pos[:,0], avg_df))
    print('lin y-log(df)', pearsonr(avg_obj_pos[:,0], np.log(avg_df)))
    print('lin x-df', pearsonr(avg_obj_pos[:,1], avg_df))
    print('lin y-dazim', pearsonr(avg_obj_pos[:,0], avg_dazim))
    print('lin x-dazim', pearsonr(avg_obj_pos[:,1], avg_dazim))
    print('lin delta-da', pearsonr(delta_avg, avg_da))

    fig1 = plt.figure()
    ax1 = plt.subplot(111)
    nstream_per_image = df.shape[1] * df.shape[2]
    df_streamss = df.mean(-1)#.reshape([df.shape[0], nstream_per_image]).reshape([-1])
    y_pos_streams = np.tile(np.expand_dims(avg_obj_pos[:, 0], -1), [1, df.shape[2]]).reshape([-1])
    for i in range(df.shape[1]):
        df_streams = df_streamss[:, i, :].reshape([-1])
        ax1.spines['right'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax1.set_yscale("log", nonposy='clip')
        ax1.scatter(120-y_pos_streams, df_streams, s=1, c='C0' if i == 0 else 'C1')
        # ax1.set_yticks([600, 1000, 20000])
        ax1.set_ylim(100, 10000)
        plt.xlabel('Mean Vertical Object Position (px)')
        plt.ylabel('Mean Soundstream Frequency (Hz)')
        plt.xlim([0, 120])
        #plt.ylim(ymin=600)
        # plt.title('table y-df')

    fig2 = plt.figure()
    ax2 = plt.subplot(111)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.scatter(160 - avg_obj_pos[:,1], avg_dazim, s=1)
    # plt.title('table x-dazim')
    plt.xlabel('Mean Horizontal Object Position (px)')
    plt.ylabel('Mean Soundscape Azimuth (rad)')
    plt.xlim([0, 160])

    fig3 = plt.figure()
    ax3 = plt.subplot(111)
    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax3.set_yscale("log", nonposy='clip')
    ax3.scatter(delta_avg, avg_da, s=1)
    # plt.title('table delta-da')

    # more in detail: avg drawing position vs sound prop of each stream
    nstream_per_image = dazim.shape[1]*dazim.shape[2]
    dazim_streams = dazim.mean(-1).reshape([dazim.shape[0], nstream_per_image]).reshape([-1])
    x_pos_streams = np.tile(np.expand_dims(avg_obj_pos[:, 1], -1), [1, nstream_per_image]).reshape([-1])
    fig4 = plt.figure()
    ax4 = plt.subplot(111)
    ax4.spines['right'].set_visible(False)
    ax4.spines['top'].set_visible(False)
    ax4.scatter(160 - x_pos_streams, dazim_streams, s=1)
    ax4.set_yticks([-pi/2, -pi/4, 0, pi/4, pi/2])
    ax4.set_yticklabels(['$-\\frac{\pi}{2}$', '$-\\frac{\pi}{4}$', '0', '$\\frac{\pi}{4}$', '$\\frac{\pi}{2}$'])
    # plt.title('table x-dazim')
    plt.xlabel('Mean Horizontal Object Position (px)')
    plt.ylabel('Mean Soundstream Azimuth (rad)')
    plt.xlim([0, 160])

    # even more detail, separated the different soundscapes
    nstream_per_image = dazim.shape[1] * dazim.shape[2]
    dazim_streamss = dazim.mean(-1)#.reshape([dazim.shape[0], dazim.shape[1], dazim.shape[2]])
    x_pos_streams = np.tile(np.expand_dims(avg_obj_pos[:, 1], -1), [1, dazim.shape[2]]).reshape([-1])
    fig4 = plt.figure()
    ax4 = plt.subplot(111)
    for i in range(dazim.shape[1]):
        dazim_streams = dazim_streamss[:, i, :].reshape([-1])
        ax4.spines['right'].set_visible(False)
        ax4.spines['top'].set_visible(False)
        ax4.scatter(160 - x_pos_streams, dazim_streams, s=1, c='C0' if i == 0 else 'C1')
        ax4.set_yticks([-pi / 2, -pi / 4, 0, pi / 4, pi / 2])
        ax4.set_yticklabels(['$-\\frac{\pi}{2}$', '$-\\frac{\pi}{4}$', '0', '$\\frac{\pi}{4}$', '$\\frac{\pi}{2}$'])
        # plt.title('table x-dazim')
        plt.xlabel('Mean Horizontal Object Position (px)')
        plt.ylabel('Mean Soundstream Azimuth (rad)')
        plt.xlim([0, 160])


    plt.show()




# 2)
# gather avg da, df, dazim; gx, gy, delta
da = np.array(hand_data_file.root.da)
df = np.array(hand_data_file.root.df)
dazim = np.array(hand_data_file.root.dazim)
avg_da = da.mean(axis=3).reshape([-1])
avg_df = df.mean(axis=3).reshape([-1])
avg_dazim = dazim.mean(axis=3).reshape([-1])

gx = np.array(hand_data_file.root.gx).reshape([-1])
gy = np.array(hand_data_file.root.gy).reshape([-1])
delta = np.array(hand_data_file.root.delta).reshape([-1])
angle = np.array(hand_data_file.root.angle).reshape([-1])

if ANAL['2']:
    print('')
    print('lin gy-df', pearsonr(gy, avg_df))
    print('lin gy-log(df)', pearsonr(gy, np.log(avg_df)))
    print('lin gx-dazim', pearsonr(gx, avg_dazim))
    print('lin delta-da', pearsonr(delta, avg_da))
    print('mut gy-df', mutual_info_score(gy, avg_df))
    print('mut gx-dazim', mutual_info_score(gx, avg_dazim))
    print('mut delta-da', mutual_info_score(delta, avg_da))
    print('low mut gy-dazim', mutual_info_score(gy, avg_dazim))

    fig1 = plt.figure()
    ax1 = plt.subplot(111)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.scatter(160-gx, avg_dazim, s=1)
    ax1.set_yticks([-pi / 2, -pi / 4, 0, pi / 4, pi / 2])
    ax1.set_yticklabels(['$-\\frac{\pi}{2}$', '$-\\frac{\pi}{4}$', '0', '$\\frac{\pi}{4}$', '$\\frac{\pi}{2}$'])
    # plt.title('hand gx-dazim')
    plt.xlabel('Mean Horizontal Patch Position (px)')
    plt.ylabel('Mean Soundstream Azimuth (rad)')
    plt.xlim([0, 160])

    fig2 = plt.figure()
    ax2 = plt.subplot(111)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.scatter(120-gy, avg_df, s=1)
    ax2.set_yscale("log", nonposy='clip')
    ax2.set_ylim(100, 10000)
    # plt.title('hand gy-df')
    plt.xlabel('Mean Vertical Patch Position (px)')
    plt.ylabel('Mean Soundstream Frequency (Hz)')
    plt.xlim([0, 120])
    plt.ylim(ymin=100)

    fig3 = plt.figure()
    ax3 = plt.subplot(111)
    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax3.scatter(delta, avg_da, s=1)
    # plt.title('hand delta-da')

    plt.show()




if ANAL['3']:
    # 3) what's up with angle
    print('')
    print('lin angle-df', pearsonr(angle, avg_df))
    print('lin angle-dazim', pearsonr(angle, avg_dazim))
    print('lin angle-da', pearsonr(angle, avg_da))

    # import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D
    # subsampl = np.random.permutation(np.arange(len(avg_da)))[:2000]
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.scatter(zs=np.log(avg_df[subsampl]), xs=avg_da[subsampl], ys=avg_dazim[subsampl])
    # # ax.set_yscale('log')
    # ax.set_xlabel('da')
    # ax.set_ylabel('dazim')
    # ax.set_zlabel('df')
    #
    # from sklearn import linear_model
    # from sklearn.metrics import r2_score
    # clf = linear_model.LinearRegression()
    # X = np.transpose(np.stack([avg_df, avg_da, avg_dazim]))
    # clf.fit(X, angle)
    # print(clf.coef_)
    # aaa = np.array(clf.coef_)
    # pred = np.sum(X * aaa, axis=1)
    # plt.scatter(angle, pred, s=1)
    # print(r2_score(angle, pred))
    #
    # plt.show()

    fig1 = plt.figure()
    ax1 = plt.subplot(111)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.scatter(angle, avg_dazim, s=1)
    plt.title('hand angle-dazim')

    fig2 = plt.figure()
    ax2 = plt.subplot(111)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.scatter(angle, avg_df, s=1)
    ax2.set_yscale("log", nonposy='clip')
    plt.title('hand angle-df')

    fig3 = plt.figure()
    ax3 = plt.subplot(111)
    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax3.scatter(angle, avg_da, s=1)
    plt.title('hand angle-da')

    plt.show()




# 4)
nsample = hand_data_file.root.raw_da.shape[0]
raw_da = np.array(hand_data_file.root.raw_da)
raw_df = np.array(hand_data_file.root.raw_df)
raw_dazim = np.array(hand_data_file.root.raw_dazim)

raw_da_ss = np.reshape(raw_da, [nsample, -1])
raw_df_ss = np.reshape(raw_df, [nsample, -1])
raw_dazim_ss = np.reshape(raw_dazim, [nsample, -1])

if ANAL['4']:

    # write meta data csv, gather labels manually for each hand posture in test set
    # 1,2,3,4,5 labels are for the postures used in the experiments
    # 42 is the european 4, 666 is the metal, 420 is the surf posture, 0 is the hole, 22 is the american 2
    imgs = np.array(hand_data_file.root.gen_img)

    if ANAL['4/labeling']:
        img_labeling(imgs, test_set, 'hand_posture_labels')

    # create sprite image
    grid = int(math.sqrt(nsample)) + 1
    image_height = int(8192 / grid)         # tensorboard supports sprite images up to 8192 x 8192
    image_width = int(8192 / grid)
    sprite_img_path = 'hand_sprite_image' + set_text + '.jpg'

    if ANAL['4/create_sprite']:
        create_sprite(imgs, grid, image_height, image_width, sprite_img_path)

    # embed features
    sound_features = np.concatenate([raw_da_ss, raw_df_ss, raw_dazim_ss], axis=-1)
    embed_features(sound_features, 'hand_embedding', 'hand_posture_labels', sprite_img_path, image_height, image_width)




if ANAL['5']:
    # 5)
    # plot sound feature modulation lines along the angle dimension
    Y_DIM_NAME = 'angle'
    Z_DIM_NAME = 'dazim'
    nbin = 30
    perc = 0.95

    nstream = raw_da.shape[-2]
    nmodulations = raw_da.shape[-1]
    angle_mods = np.array(hand_data_file.root.angle).reshape([-1])  # * 180 / np.pi  # each corresponds to a stream
    gx_mods = np.array(hand_data_file.root.gx).reshape([-1])
    gy_mods = np.array(hand_data_file.root.gy).reshape([-1])

    raw_df_mods = np.reshape(raw_df, [-1, nmodulations])
    raw_da_mods = np.reshape(raw_da, [-1, nmodulations])
    raw_dazim_mods = np.reshape(raw_dazim, [-1, nmodulations])  # * 180 / np.pi

    # for each stream, there is an angle = for each set of modulations, there is an angle
    # angle: -1.6 -- 2.2, bin it, avg the sound qualities inside bins
    angle_bin_boundaries = np.linspace(np.min(angle_mods), 1.6, num=nbin)  # * 180/np.pi
    gx_bin_boundaries = np.linspace(np.min(gx_mods), np.max(gx_mods), num=nbin)
    gy_bin_boundaries = np.linspace(np.min(gy_mods), np.max(gy_mods), num=nbin)
    angle_bins = np.digitize(angle_mods, angle_bin_boundaries)
    gx_bins = np.digitize(gx_mods, gx_bin_boundaries)
    gy_bins = np.digitize(gy_mods, gy_bin_boundaries)
    _bins = {'angle': angle_bins, 'gx': gx_bins, 'gy': gy_bins}
    _bin_boundaries = {'angle': angle_bin_boundaries, 'gx': gx_bin_boundaries, 'gy': gy_bin_boundaries}
    _mods = {'angle': angle_mods, 'gx': gx_mods, 'gy': gy_mods}

    maxbin = np.max(_bins[Y_DIM_NAME])

    import scipy.stats as st
    binned_avg_vectors = {'da': [], 'df': [], 'dazim': [], 'angle': [], 'gx': [], 'gy': []}
    binned_perc_vectors = {'da': [[], []], 'df': [[], []], 'dazim': [[], []]}  # 95th percentile, lower and upper bound
    for b in np.arange(1, maxbin + 1):
        indices = np.where(_bins[Y_DIM_NAME] == b)

        binned_angle = np.reshape(angle_mods[np.where(angle_bins == b)], [-1, 1])
        binned_angle = np.tile(binned_angle, [1, nmodulations])
        binned_avg_vectors['angle'].append(np.mean(binned_angle, axis=0))

        binned_gx = np.tile(np.reshape(gx_mods[np.where(gx_bins == b)], [-1, 1]), [1, nmodulations])
        binned_avg_vectors['gx'].append(np.mean(binned_gx, axis=0))
        binned_gy = np.tile(np.reshape(gy_mods[np.where(gy_bins == b)], [-1, 1]), [1, nmodulations])
        binned_avg_vectors['gy'].append(np.mean(binned_gy, axis=0))

        binned_da = raw_da_mods[indices, :].reshape([-1, nmodulations])
        binned_df = raw_df_mods[indices, :].reshape([-1, nmodulations])
        binned_dazim = raw_dazim_mods[indices, :].reshape([-1, nmodulations])
        binned_avg_vectors['da'].append(binned_da.mean(axis=0))
        binned_avg_vectors['df'].append(binned_df.mean(axis=0))
        binned_avg_vectors['dazim'].append(binned_dazim.mean(axis=0))

        # 95 percentile for each modulation: 2 x nmod
        # from https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data
        binned_percs_da = [[], []]
        binned_percs_df = [[], []]
        binned_percs_dazim = [[], []]
        for i in range(nmodulations):
            binned_perc_da = st.t.interval(perc, binned_da[:,i].size - 1, loc=binned_da[:,i].mean(), scale=st.sem(binned_da[:,i].reshape([-1])))
            binned_perc_df = st.t.interval(perc, binned_df[:,i].size - 1, loc=binned_df[:,i].mean(), scale=st.sem(binned_df[:,i].reshape([-1])))
            binned_perc_dazim = st.t.interval(perc, binned_dazim[:, i].size - 1, loc=binned_dazim[:, i].mean(), scale=st.sem(binned_dazim[:, i].reshape([-1])))
            binned_percs_da[0].append(binned_perc_da[0])
            binned_percs_df[0].append(binned_perc_df[0])
            binned_percs_dazim[0].append(binned_perc_dazim[0])
            binned_percs_da[1].append(binned_perc_da[1])
            binned_percs_df[1].append(binned_perc_df[1])
            binned_percs_dazim[1].append(binned_perc_dazim[1])

        binned_perc_vectors['da'][0].append(np.array(binned_percs_da[0]))
        binned_perc_vectors['df'][0].append(np.array(binned_percs_df[0]))
        binned_perc_vectors['dazim'][0].append(np.array(binned_percs_dazim[0]))
        binned_perc_vectors['da'][1].append(np.array(binned_percs_da[1]))
        binned_perc_vectors['df'][1].append(np.array(binned_percs_df[1]))
        binned_perc_vectors['dazim'][1].append(np.array(binned_percs_dazim[1]))

    # # not averaged/binned plot
    # x_dim = np.arange(nmodulations)
    # color = [0, 0, 0]
    # c = 0
    # for z_dim, angl in zip(raw_dazim_mods, angle_mods):
    #     ax.plot(x_dim, angl, z_dim, color=color)
    #     color[2] += 1./angle_mods.shape[0]
    #     color[2] = color[2] if color[2] <= 1. else 1.
    #     c += 1
    #     if c > 800:
    #         break
    # plt.ylabel('Angle (rad)')
    # plt.xlabel('x')

    # plt.figure()
    # plt.hist(raw_dazim_mods.reshape([-1]))
    # plt.show()

    # create mesh
    import matplotlib.colors as colors

    Z_var = np.stack(binned_avg_vectors[Z_DIM_NAME])
    x_mod = np.arange(nmodulations)
    concat_binned = {k: np.concatenate(v, axis=0) for k,v in binned_avg_vectors.items()}
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    X_mod, Y_dim = np.meshgrid(x_mod, np.stack(binned_avg_vectors[Y_DIM_NAME], axis=0).mean(axis=1))
    # ax.contour3D(X_mod, Y_dim, Z_var, 100, cmap='binary')
    _colors = {'da': None, 'dazim': None, 'df': colors.LogNorm(np.min(Z_var), np.max(Z_var))}
    ax.plot_surface(X_mod, Y_dim, Z_var, cmap='cool', edgecolor='none', norm=_colors[Z_DIM_NAME], alpha=1.)  # 600, 8000 for df
    _labels_by_Y = {'angle': 'V1 Patch Angle (rad)', 'gx': 'Horizontal Position (px)', 'gy': 'Vertical Position (px)'}
    _labels_by_Z = {'da': 'Amplitude', 'df': 'Frequency (Hz)', 'dazim': 'Sound Source Location (rad)'}
    if Y_DIM_NAME == 'angle':
        ax.set_yticks([-pi / 2, -pi / 4, 0, pi / 4, pi / 2])
        ax.set_yticklabels(['$-\\frac{\pi}{2}$', '$-\\frac{\pi}{4}$', '0', '$\\frac{\pi}{4}$', '$\\frac{\pi}{2}$'])
    if Z_DIM_NAME == 'dazim':
        ax.set_zticks([-pi / 4, 0, pi / 4])
        ax.set_zticklabels(['$-\\frac{\pi}{4}$', '0', '$\\frac{\pi}{4}$'])
    ax.set_xlabel('Modulation Index')
    ax.set_ylabel(_labels_by_Y[Y_DIM_NAME])
    ax.set_zlabel(_labels_by_Z[Z_DIM_NAME])
    ax.view_init(35, 35)
    # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.set_xticks(np.arange(nmodulations))

    # draw 95 percentile as transparent meshgrid, x is discrete
    Z_var_low = np.stack(binned_perc_vectors[Z_DIM_NAME][0], axis=0)
    Z_var_high = np.stack(binned_perc_vectors[Z_DIM_NAME][1], axis=0)
    ax.plot_surface(X_mod, Y_dim, Z_var_low, cmap='binary', edgecolor='gray', norm=_colors[Z_DIM_NAME], alpha=0.2)
    ax.plot_surface(X_mod, Y_dim, Z_var_high, cmap='binary', edgecolor='none', norm=_colors[Z_DIM_NAME], alpha=0.2)

    if ANAL['5/draw_hand']:
        # get a 5 finger hand posture, get those drawings that have angles close to the bin where
        five_fing_i = 38  # works with test set only 46, 38, 2
        ff_cs = hand_data_file.root.cs[five_fing_i]
        ff_angle = hand_data_file.root.angle[five_fing_i]
        # ff_bin_width = 2  # amount of bin boundaries * 2, between which the drawing is colored
        # criteria_bin = int(np.argmax(np.mean(Z_var, axis=-1)))  # around which the drawings are colored
        # if criteria = 8, then bin_boundaries[8]--bin_boundaries[9] is the interval we're looking for
        # ff_lower_bound, ff_upper_bound = _bin_boundaries[Y_DIM_NAME][criteria_bin - (ff_bin_width - 1)], \
        #                                  _bin_boundaries[Y_DIM_NAME][min(criteria_bin + ff_bin_width, len(_bin_boundaries[Y_DIM_NAME])-1)]
        # ff_lower_bound, ff_upper_bound = 0.6, 1. # in rad -0.9, -0.1
        ff_color_purple = 1. - np.array([219./255, 36./255, 255./255])
        ff_color_blue = np.array([30./255, 225./255, 255./255])
        angle_colors = [(0.6, 1., ff_color_blue), (-0.88, -0.24, ff_color_purple)]  # (lower bound, upper bound, color) 34-57, -50--13
        print('finger angle minmaxmeanyo', ff_angle.min(), ff_angle.max(), ff_angle.mean(),
              np.logical_and(angle_colors[1][0] < ff_angle, ff_angle < angle_colors[1][1]).sum())

        canvas = []
        ff_cs = ff_cs.astype(np.float32) / 255.
        ff_cs = np.concatenate([ff_cs[np.newaxis, 0],  ff_cs[1:] - ff_cs[:-1]])
        ff_cs = np.tile(np.expand_dims(ff_cs, -1), [1, 1, 3])  # to float RGB
        for c, angl in zip(ff_cs, ff_angle):
            # compute the coloring
            # if majority of angles are within the bound, color it
            coloring = np.zeros(c.shape)  # default
            for lower_bound, upper_bound, col in angle_colors:
                if np.sum(np.logical_and(lower_bound < angl[1], angl[1] < upper_bound)) > nstream/4.1:  # majority
                    coloring = c * col*2
                    print(lower_bound, upper_bound, angl)
                elif lower_bound < 0 and np.sum(np.logical_and(lower_bound < angl, angl < upper_bound)) > 0:  # gany
                    coloring = c * col * 2
                    print('huhu')
            c = c + coloring
            canvas.append(c)
            cv2.imshow("canvas", c)
            if cv2.waitKey(1) == 27:  # esc
                break
            time.sleep(0.2)

        # invert, sum, invert
        canvas = np.stack(canvas, axis=0)
        canvas = canvas / np.max(canvas)
        canvas = np.sum(canvas, axis=0)
        canvas = 1. - canvas
        cv2.imshow("canvas", canvas)
        # cv2.imwrite('finger_angle_colored.jpg', canvas)
        if cv2.waitKey(1) == 27:  # esc
            pass

    if ANAL['5/hist']:
        # histogram of angles
        fig, ax = plt.subplots()
        plt.hist(_mods[Y_DIM_NAME], 30)
        plt.xlabel(_labels_by_Y[Y_DIM_NAME])
        ax.set_xticks([-pi / 2, -pi / 4, 0, pi / 4, pi / 2])
        ax.set_xticklabels(['$-\\frac{\pi}{2}$', '$-\\frac{\pi}{4}$', '0', '$\\frac{\pi}{4}$', '$\\frac{\pi}{2}$'])


    # averaged/binned lame line plot
    if ANAL['5/lame_plot']:
        fig2, ax2 = plt.subplots(subplot_kw={'projection': '3d'})
        color = [0, 0, 0]
        for z_dim, angl in zip(binned_avg_vectors['df'], binned_avg_vectors['angle']):
            ax2.plot(x_mod, angl, z_dim, color=color)  # FIXME sets the color of all lines, that"s why all blue in the end
            color[2] += 1./len(binned_avg_vectors['df'])
            color[2] = color[2] if color[2] <= 1. else 1.

        plt.ylabel('Angle (rad)')
        plt.xlabel('Modulation index')

    # TODO change angle by pressing keys, play a random corresponding soundstream (print the current angle to console)

    plt.show()



# 6
nsample = table_data_file.root.raw_da.shape[0]
raw_da = np.array(table_data_file.root.raw_da)
raw_df = np.array(table_data_file.root.raw_df)
raw_dazim = np.array(table_data_file.root.raw_dazim)

raw_da_ss = np.reshape(raw_da, [nsample, -1])
raw_df_ss = np.reshape(raw_df, [nsample, -1])
raw_dazim_ss = np.reshape(raw_dazim, [nsample, -1])

raw_da_avg = raw_da.mean(axis=-1)
raw_df_avg = raw_df.mean(axis=-1)
raw_dazim_avg = raw_dazim.mean(axis=-1)

shape_features = [0,1,2,3,4,5,6,7] #[2,3]
raw_da_avg_cvsg = raw_da_avg[:,shape_features,:].reshape([raw_da.shape[0], -1])
raw_df_avg_cvsg = raw_df_avg[:,shape_features,:].reshape([raw_da.shape[0], -1])
raw_dazim_avg_cvsg = raw_dazim_avg[:,shape_features,:].reshape([raw_da.shape[0], -1])

if ANAL['6']:
    # first label the table object test dataset
    imgs = np.array(table_data_file.root.gen_img)
    if ANAL['6/labeling']:
        img_labeling(imgs, test_set, 'table_labels')

    grid = int(math.sqrt(nsample)) + 1
    image_height = int(8192 / grid)  # tensorboard supports sprite images up to 8192 x 8192
    image_width = int(8192 / grid)
    sprite_img_path = 'table_sprite_image' + set_text + '.jpg'

    if ANAL['6/create_sprite']:
        create_sprite(imgs, grid, image_height, image_width, sprite_img_path)

    # embed features
    sound_features = np.concatenate([raw_da_ss, raw_df_ss, raw_dazim_ss], axis=-1)
    if ANAL['6/embed']:
        embed_features(sound_features, 'table_embedding', 'table_labels', sprite_img_path, image_height, image_width)

    # k-means cluster into two groups - see if the clustering corresponds to the labeling
    from sklearn.cluster import KMeans
    kmeans = KMeans(2, max_iter=1000).fit(sound_features)
    with open('table_labels' + set_text + '.csv', 'rt') as f:
        labels = np.array([int(row[0]) for row in csv.reader(f)])
    print(np.sum(labels == kmeans.labels_) / len(labels))

    # it seems, looking at all the sound features does not help
    # so select those soundstreams only that in average are more on the lest periphery,
    # where can vs gear distinction can be perceived
    indices = (raw_dazim_avg > 1.5).reshape([raw_dazim.shape[0], raw_dazim.shape[1], -1])
    # it looks as the first soundscape corresponds to the position, the rest is played mostly on the far left
    # remove the first soundscape, let's cluster the rest
    new_sound_features = np.concatenate([raw_da_avg_cvsg, raw_df_avg_cvsg], axis=-1)  # now leave dazim out of business it's the same anyways
    new_kmeans = KMeans(2, max_iter=1000).fit(new_sound_features)
    res = np.sum(labels == new_kmeans.labels_) / len(labels)
    print(res if res > 0.5 else 1.-res)
    # the correspondence to the labels increased! let's remove some more soundscapes above
    # there is a 61% correspondence with the soundscapes 2,3,4,5
    # let's see how that looks like in t-sne visualization
    if ANAL['6/embed']:
        embed_features(new_sound_features, 'table_constraint_embedding', 'table_labels', sprite_img_path, image_height, image_width)
    # and it worked



# 7) play sounds from each group: can vs gear
import simpleaudio as saudio
with open('table_labels' + set_text + '.csv', 'rt') as f:
    reader = csv.reader(f)
    indices = [int(row[0]) for row in reader]
can_indices = np.array(indices) == 0
gear_indices = np.array(indices) == 1

if ANAL['7']:
    sounds = np.array(table_data_file.root.soundscapes)
    imgs = np.array(table_data_file.root.gen_img)
    can_sounds, can_imgs = sounds[can_indices,:,:], imgs[can_indices,:,:]
    gear_sounds, gear_imgs = sounds[gear_indices,:,:], imgs[gear_indices,:,:]
    grouping = {0: [can_sounds, can_imgs], 1: [gear_sounds, gear_imgs]}

    group = 0
    playit = True
    while playit:
        # rand sound/img
        i = np.random.randint(0, len(grouping[group][0]), 1)
        p = saudio.play_buffer(grouping[group][0][i], 2, 2, 44100)
        img = grouping[group][1][i].reshape([can_imgs.shape[1], can_imgs.shape[2]])
        cv2.imshow("GenImgCam", img)
        c = cv2.waitKey(1)
        if c == 27:  # esc
            playit = False
        elif c == 99:
            group = 0
        elif c == 103:
            group = 1
        elif c != -1:
            print(c)
        p.wait_done()




# 8) plot distribution of table dataset sound properties: can vs gear
if ANAL['8']:
    # ~in the left azimuth, there is more oscillation for the cans than the gears
    # so, take soundstreams that are towards the left (+pi/2), check their distribution
    # according to anal/6, soundscapes 2,3,4,5 most likely contain information about can vs gear, p < .1^10
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    raw_da_avg_cvsg = raw_da_avg_cvsg.mean(-1)
    raw_df_avg_cvsg = raw_df_avg_cvsg.mean(-1)
    raw_dazim_avg_cvs = raw_dazim_avg_cvsg.mean(-1)

    ax.scatter(raw_da_avg_cvsg[can_indices], raw_dazim_avg_cvs[can_indices], raw_df_avg_cvsg[can_indices], c='C0', s=3)
    ax.scatter(raw_da_avg_cvsg[gear_indices], raw_dazim_avg_cvs[gear_indices], raw_df_avg_cvsg[gear_indices], c='C1', s=3)

    ax.set_xlabel('Mean Amplitude')
    ax.set_ylabel('Mean Azimuth (rad)')
    ax.set_zlabel('Mean Frequency (Hz)')
    ax.view_init(35, 50)
    # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    plt.legend(['Beer can', 'Gear'], frameon=True)

    # only amplitude and object type
    plt.figure()
    plt.scatter(raw_da_avg_cvsg[can_indices], raw_df_avg_cvsg[can_indices], c='C0', s=2)
    plt.scatter(raw_da_avg_cvsg[gear_indices], raw_df_avg_cvsg[gear_indices], c='C1', s=2)

    # draw uncertainty plots of modulations of amplitude and frequency of 2,3,4,5
    # amplitude in one plot, freq in a separate plot; with line plots but don't (maybe) connect the separate soundscapes
    sound_feature = raw_dazim
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    recs = [[], []]
    ax2.plot(np.arange(1), sound_feature.mean(), c='C0', linewidth=1)
    ax2.plot(np.arange(1), sound_feature.mean(), c='C1', linewidth=1)
    ax2.legend(['Beer can', 'Gear'], frameon=True)
    for rec_i in range(sound_feature.shape[0]):  # through samples
        label = indices[rec_i]
        rec = sound_feature[rec_i][shape_features].reshape([-1])
        ax2.plot(np.arange(len(rec)), rec, c='C' + str(label), alpha=0.01)
        #for soundscape_i in [2,3,4,5]:  # only the shape info soundscapes
            #ax2.plot()
            #for soundstream_i in range(raw_da.shape[2]):
        recs[label].append(rec)
    mean = [sum(rr) / len(rr) for rr in recs]
    ax2.plot(np.arange(len(mean[0])), mean[0], c='C0')
    ax2.plot(np.arange(len(mean[1])), mean[1], c='C1')
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    # ax2.set_yscale("log", nonposy='clip')
    plt.xlabel('Modulation Index')
    plt.ylabel('Frequency (Hz)')
    # plt.ylim([0, 10000])

    plt.show()
    # TODO make sound changing as you move on the grid experience; use the previously generated grid information

# bye
table_data_file.close()
hand_data_file.close()
