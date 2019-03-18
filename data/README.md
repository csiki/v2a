# How to generate a dataset from videos

0. Record videos of the visual environment, yielding mp4 files
1. Generate images: `./gen_imgs_from_vids.sh images/out/path/ wildcard/path/to/videos/*.mp4`
2. Turn images into black and white, scale them to 160x120, then apply edge detection if anything but CORF is used:
`prepare_imgs.py corf|sobel wildcard/to/images*.jpg path/to/output/`
3. If you chose CORF as your edge detection algorithm in the previous step, run Matlab script
`convertAllInFolder2DPar.m` in parallel under `matlab/faster_corf/` to extract contour
(see instruction in script), else just continue on with step 4
4. Enrich dataset with mirror images if applicable: run `mirror_imgs.py wildcard/to/images*.png`
5. Merge images into hdf5 file: `merge_imgs.py path/to/contour/imgs*.jpg output/path/something.hdf5`

The resulting hdf5 file should contain separate randomized training and test sets,
by a default 90-10% fold (specify in `merge_imgs.py` if needed).

Datasets used in the study are also available. Download, then place them in the `data` folder:

- [hand gesture dataset](https://ufile.io/xq28e),
- [table / reaching movement dataset](https://ufile.io/9uv24).
