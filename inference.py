import sys

sys.path.append('./parsing_reid/')
from datasets.datasets import VehicleReIDParsingDataset, get_preprocessing, get_validation_augmentation
from pathlib import Path
from utils import mkdir_p
from yacs.config import CfgNode
from logzero import logger
from models import ParsingReidModel, ParsingTripletLoss
from utils.math_tools import Clck_R1_mAP
from utils.iotools import merge_configs
from datasets import make_basic_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch
import logging
import argparse
import pickle
import cv2
import os
import numpy as np
import segmentation_models_pytorch as smp

ENCODER = 'se_resnext50_32x4d'
ENCODER_WEIGHTS = 'imagenet'
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

IMG2MASK = {}


def predict(model, test_dataset, test_dataset_vis, output_path):
    mkdir_p(output_path)
    for i in tqdm(range(len(test_dataset))):
        image = test_dataset[i]
        image_vis, extra = test_dataset_vis[i]

        # 重复图片直接用之前计算好的即可
        image_path = Path(extra["image_path"])
        if str(image_path) in IMG2MASK:
            extra["mask_path"] = str(IMG2MASK[str(image_path)])
            continue
        mask_path = output_path / f"{image_path.name.split('.')[0]}.png"

        x_tensor = torch.from_numpy(image).to("cuda").unsqueeze(0)
        with torch.no_grad():
            pr_mask = model.predict(x_tensor)
        pr_map = pr_mask.squeeze().cpu().numpy().round()
        pr_map = np.argmax(pr_map, axis=0)[:image_vis.shape[0], :image_vis.shape[1]]
        cv2.imwrite(str(mask_path), pr_map.astype(np.uint8))
        extra["mask_path"] = str(mask_path)

        IMG2MASK[str(image_path)] = str(mask_path)


def make_config():
    cfg = CfgNode()
    cfg.desc = ""  # 对本次实验的简单描述，用于为tensorboard命名
    cfg.stage = "train"  # train or eval or test
    cfg.device = "cpu"  # cpu or cuda
    cfg.device_ids = ""  # if not set, use all gpus
    cfg.output_dir = "/data/vehicle_reid/perspective_transform_feature/debug"
    cfg.debug = False

    cfg.train = CfgNode()
    cfg.train.epochs = 120

    cfg.data = CfgNode()
    cfg.data.name = "VeRi776"
    cfg.data.pkl_path = "../data_processing/veri776.pkl"
    cfg.data.train_size = (256, 256)
    cfg.data.valid_size = (256, 256)
    cfg.data.pad = 10
    cfg.data.re_prob = 0.5
    cfg.data.with_mask = True
    cfg.data.test_ext = ''

    cfg.data.sampler = 'RandomIdentitySampler'
    cfg.data.batch_size = 16
    cfg.data.num_instances = 4

    cfg.data.train_num_workers = 0
    cfg.data.test_num_workers = 0

    cfg.model = CfgNode()
    cfg.model.name = "resnet50"
    # If it is set to empty, we will download it from torchvision official website.
    cfg.model.pretrain_path = ""
    cfg.model.last_stride = 1
    cfg.model.neck = 'bnneck'
    cfg.model.neck_feat = 'after'
    cfg.model.pretrain_choice = 'imagenet'
    cfg.model.ckpt_period = 10

    cfg.optim = CfgNode()
    cfg.optim.name = 'Adam'
    cfg.optim.base_lr = 3.5e-4
    cfg.optim.bias_lr_factor = 1
    cfg.optim.weight_decay = 0.0005
    cfg.optim.momentum = 0.9

    cfg.loss = CfgNode()
    cfg.loss.losses = ["triplet", "id", "center", "local-triplet"]
    cfg.loss.triplet_margin = 0.3
    cfg.loss.normalize_feature = True
    cfg.loss.id_epsilon = 0.1

    cfg.loss.center_lr = 0.5
    cfg.loss.center_weight = 0.0005

    cfg.loss.tuplet_s = 64
    cfg.loss.tuplet_beta = 0.1

    cfg.scheduler = CfgNode()
    cfg.scheduler.milestones = [40, 70]
    cfg.scheduler.gamma = 0.1
    cfg.scheduler.warmup_factor = 0.0
    cfg.scheduler.warmup_iters = 10
    cfg.scheduler.warmup_method = "linear"

    cfg.test = CfgNode()
    cfg.test.feat_norm = True
    cfg.test.remove_junk = True
    cfg.test.period = 10
    cfg.test.device = "cuda"
    cfg.test.model_path = "/home/hanlanqian/PycharmProjects/PVEN/checkpoints/veri776_reid.pth"
    cfg.test.max_rank = 50
    cfg.test.rerank = False
    cfg.test.lambda_ = 0.0
    cfg.test.output_html_path = ""
    # split: When the CUDA memory is not sufficient,
    # we can split the dataset into different parts
    # for the computing of distance.
    cfg.test.split = 0

    cfg.logging = CfgNode()
    cfg.logging.level = "info"
    cfg.logging.period = 20
    cfg.infer_flag = False

    return cfg


def build_model(cfg, num_classes):
    # if cfg.MODEL.NAME == 'resnet50':
    #     model = Baseline(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT)
    model = ParsingReidModel(num_classes, cfg.model.last_stride, cfg.model.pretrain_path, cfg.model.neck,
                             cfg.model.neck_feat, cfg.model.name, cfg.model.pretrain_choice)
    return model


def eval_(model,
          device,
          valid_loader,
          query_length,
          feat_norm=True,
          remove_junk=True,
          max_rank=50,
          output_dir='',
          rerank=False,
          lambda_=0.5,
          split=0,
          output_html_path='',
          **kwargs):
    """实际测试函数

    Arguments:
        model {nn.Module}} -- 模型
        device {string} -- 设备
        valid_loader {DataLoader} -- 测试集
        query_length {int} -- 测试集长度

    Keyword Arguments:
        remove_junk {bool} -- 是否删除垃圾图片 (default: {True})
        max_rank {int} -- [description] (default: {50})
        output_dir {str} -- 输出目录。若为空则不输出。 (default: {''})

    Returns:
        [type] -- [description]
    """
    metric = Clck_R1_mAP(query_length, max_rank=max_rank, rerank=rerank, remove_junk=remove_junk, feat_norm=feat_norm,
                         output_path=output_dir, lambda_=lambda_)
    model.eval()
    with torch.no_grad():
        for batch in tqdm(valid_loader):
            for name, item in batch.items():
                if isinstance(item, torch.Tensor):
                    batch[name] = item.to("cuda")
            output = model(**batch)
            global_feat = output["global_feat"]
            local_feat = output["local_feat"]
            vis_score = output["vis_score"]
            metric.update((global_feat.detach().cpu(), local_feat.detach().cpu(), vis_score.cpu(), batch["id"].cpu(),
                           batch["cam"].cpu(), batch["image_path"]))

    print(f'Saving features to {output_dir}/test_features.pkl')
    metric.save(f'{output_dir}/test_features.pkl')

    print(f'Computing')
    metric_output = metric.compute(split=split, infer_flag=kwargs['infer_flag'])
    if not kwargs['infer_flag']:
        cmc = metric_output['cmc']
        mAP = metric_output['mAP']
        distmat = metric_output['distmat']
        all_AP = metric_output['all_AP']

        if output_html_path != '':
            from utils.visualize import reid_html_table
            query = valid_loader.dataset.meta_dataset[:query_length]
            gallery = valid_loader.dataset.meta_dataset[query_length:]
            # distmat = np.random.rand(query_length, len(valid_loader.dataset.meta_dataset)-query_length)
            reid_html_table(query, gallery, distmat, output_html_path, all_AP, topk=15)

        metric.reset()
        logger.info(f"mAP: {mAP:.2%}")
        for r in [1, 5, 10]:
            logger.info(f"CMC curve, Rank-{r:<3}:{cmc[r - 1]:.2%}")
        return cmc, mAP
    else:
        return metric_output.get('distmat')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="./checkpoints/parsing_model.pth")
    parser.add_argument("--reid-pkl-path", type=str, default='./veri776.pkl')
    parser.add_argument("--output-path", type=str, default='./outputs/')
    parser.add_argument("--config", type=str, default="./configs/my_dataset.yml")
    parser.add_argument("--cmd-config", default=(), nargs='?')
    args = parser.parse_args()
    model = torch.load(args.model_path)
    model = model.cuda()
    model.eval()

    with open(args.reid_pkl_path, "rb") as f:
        metas = pickle.load(f)
    output_path = Path(args.output_path).absolute()

    for phase in metas.keys():
        sub_path = output_path / phase
        mkdir_p(str(sub_path))
        dataset = VehicleReIDParsingDataset(metas[phase], augmentation=get_validation_augmentation(),
                                            preprocessing=get_preprocessing(preprocessing_fn))
        dataset_vis = VehicleReIDParsingDataset(metas[phase], with_extra=True)
        print('Predict mask to {}'.format(sub_path))
        predict(model, dataset, dataset_vis, sub_path)

    # Write mask path to pkl
    # with open(args.reid_pkl_path, "wb") as f:
    #     pickle.dump(metas, f)

    cfg = make_config()
    print(args.cmd_config)
    cfg = merge_configs(cfg, args.config, args.cmd_config)
    print(cfg)
    os.makedirs(cfg.output_dir, exist_ok=True)
    torch.cuda.empty_cache()
    reid_model = build_model(cfg, 1).to(cfg.device)
    state_dict = torch.load(cfg.test.model_path, map_location=cfg.device)
    remove_keys = []

    for key, value in state_dict.items():
        if 'classifier' in key:
            remove_keys.append(key)
    for key in remove_keys:
        del state_dict[key]
    reid_model.load_state_dict(state_dict, strict=False)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    logger.info(f"Load model {cfg.test.model_path}")
    train_dataset, valid_dataset, meta_dataset = make_basic_dataset(metas,
                                                                    cfg.data.train_size,
                                                                    cfg.data.valid_size,
                                                                    cfg.data.pad,
                                                                    test_ext=cfg.data.test_ext,
                                                                    re_prob=cfg.data.re_prob,
                                                                    with_mask=cfg.data.with_mask,
                                                                    infer_flag=cfg.infer_flag
                                                                    )
    valid_loader = DataLoader(valid_dataset,
                              batch_size=cfg.data.batch_size,
                              num_workers=cfg.data.test_num_workers,
                              pin_memory=True,
                              shuffle=False)

    query_length = meta_dataset.num_query_imgs
    outputs = eval_(reid_model, cfg.test.device, valid_loader, query_length,
                    feat_norm=cfg.test.feat_norm,
                    remove_junk=cfg.test.remove_junk,
                    max_rank=cfg.test.max_rank,
                    output_dir=cfg.output_dir,
                    lambda_=cfg.test.lambda_,
                    rerank=cfg.test.rerank,
                    split=cfg.test.split,
                    output_html_path=cfg.test.output_html_path,
                    infer_flag=cfg.infer_flag)
    print(outputs)