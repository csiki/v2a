#How to generate a dataset from videos

0. Record videos of the visual environment, producing mp4 files
1. Generate images: `./gen_imgs_from_vids.sh images/out/path/ wildcard/path/to/videos/*.mp4`
2. Turn images into black and white and scale them to 160x120:
`scale_imgs.py wildcard/to/images*.jpg path/to/output/`
3. Run Matlab script `convertAllInFolder2DPar.m` in parallel under `matlab/faster_corf/` to extract contour
(see instruction in script)
4. Enrich dataset with mirror images if applicable: run `mirror_imgs.py wildcard/to/images*.png`
5. Merge images into hdf5 file: `merge_imgs.py path/to/contour/imgs*.jpg output/path/something.hdf5`

The resulting hdf5 file should contain separate randomized training and test sets,
by a default 90-10% fold (specify in `merge_imgs.py` if needed).
