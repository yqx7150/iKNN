# train
## forward
CUDA_VISIBLE_DEVICES=0 python 12chto4ch_kspace_train_compress.py --out_path="./exps/" --gamma --task=8to4
## backward
CUDA_VISIBLE_DEVICES=1 python 4chto12ch_kspace_train_backward.py --out_path="./exps/" --gamma --task=4chto12ch_cross_smooth_l1_backward

# test
CUDA_VISIBLE_DEVICES=1 python 12chto4ch_kspace_test_compress.py --task=test --out_path="./exps/" --ckpt="./exps/12chto3ch_cross_smooth_l1_forward/checkpoint/latest.pth"

CUDA_VISIBLE_DEVICES=1 python 12chto4ch_kspace_train_compress.py --out_path="./exps/" --gamma --task=12chto4ch_cross_smooth_l1 --resume

CUDA_VISIBLE_DEVICES=1 python 4chto12ch_kspace_test_backward.py --task=4to12_ch_cross_test --out_path="./exps/" --ckpt="./exps/4chto12ch_cross_smooth_l1_backward/checkpoint/latest.pth"

CUDA_VISIBLE_DEVICES=1 python test_weight_dc_iter_2.py  --out_path="./exps/" --ckpt_40="/zw/exps_zh/radial/radial_40/checkpoint/latest.pth" --ckpt_60="/zw/exps_zh/radial/radial_60/checkpoint/latest.pth" --ckpt_79="/zw/exps_zh/radial/radial_79/checkpoint/latest.pth" --ckpt_100="/zw/exps_zh/radial/100/checkpoint/latest.pth" --ckpt_20="/zw/exps_zh/radial_r5/checkpoint/latest.pth"

CUDA_VISIBLE_DEVICES=1 python test_weight_dc_iter.py  --out_path="./exps/" --ckpt_40="/zw/exps_zh/radial/radial_40/checkpoint/latest.pth" --ckpt_60="/zw/exps_zh/radial/radial_60/checkpoint/latest.pth" --ckpt_79="/zw/exps_zh/radial/radial_79/checkpoint/latest.pth" --ckpt_100="/zw/exps_zh/radial/100/checkpoint/latest.pth" --ckpt_10="/zw/exps_zh/radial_10/checkpoint/latest.pth"

CUDA_VISIBLE_DEVICES=1 python test_weight_dc_iter.py  --out_path="./exps/"  --ckpt_10="/zw/exps_av/random/random_10_REAL/checkpoint/latest.pth"  --ckpt_20="/zw/exps_av/random/random_20/checkpoint/latest.pth"