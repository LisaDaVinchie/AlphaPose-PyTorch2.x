set -x

CONFIG='./configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml' # 61.92 it/s 44.64
CKPT='./model_files/fast_res50_256x192.pth'
# CONFIG='./configs/halpe_coco_wholebody_136/resnet/256x192_res50_lr1e-3_2x-regression.yaml' # 55.67it/s
# CKPT='./model_files/multi_domain_fast50_regression_256x192.pth'
VIDEO='./data/video3.avi'
OUTDIR='./'

python -m scripts.demo_inference \
    --cfg ${CONFIG} \
    --checkpoint ${CKPT} \
    --video ${VIDEO} \
    --outdir ${OUTDIR} \
    --detector yolo \
    --save_video --sp --vis_fast \
    --detbatch 1 \
    --posebatch 8 \