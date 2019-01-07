0. Record video of visual environment, producing an mp4 file
1. gen_img_from_all_vids.sh: ffmpeg -i nothing.mp4 -r 10 -s 160x120 -f image2 nothing/nothing-%04d.jpg
2. scale_imgs.py - turn them into black and white and scale them to 160x120 if necessary
3. run matlab script convertAllInFolder2DPar.m in parallel under matlab/faster_corf to extract contour
4. enrich dataset with mirror images if applicable: mirror_imgs.py
5. merge images into hdf5 file: merge_hdf5.py
