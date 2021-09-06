"""
Author: Guocheng Qian
Contact: guocheng.qian@kaust.edu.sa
"""

import argparse
from utils.wandb import Wandb
from utils.config import config
import os, sys, time, shortuuid, pathlib, json, logging, os.path as osp
from torch.utils.tensorboard import SummaryWriter


def parse_option():
    parser = argparse.ArgumentParser('S3DIS scene-segmentation training')
    parser.add_argument('--cfg', type=str, required=True, help='config file')
    # parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    args, opts = parser.parse_known_args()
    config.load(args.cfg, recursive=True)
    config.update(opts)

    return args, config


# ================ experiment folder ==================
def generate_exp_directory(config, expname=None, expid=None, logname=None):
    """Function to create checkpoint folder.

    Args:
        config:
        tags: tags for saving and generating the expname
        expid: id for the current run
        logname: the name for the current run. None if auto

    Returns:
        the expname, jobname, and folders into config
    """

    if logname is None:
        if expid is None:
            expid = time.strftime('%Y%m%d-%H%M%S-') + str(shortuuid.uuid())
        if isinstance(expname, list):
            expname = '-'.join(expname)
        logname = '-'.join([expname, expid])
    config.logname = logname
    config.log_dir = os.path.join(config.log_dir, config.logname)
    config.ckpt_dir = os.path.join(config.log_dir, 'checkpoint')
    config.code_dir = os.path.join(config.log_dir, 'code')
    pathlib.Path(config.ckpt_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(config.code_dir).mkdir(parents=True, exist_ok=True)


def setup_logger(config):
    """
    Configure logger on given level. Logging will occur on standard
    output and in a log file saved in model_dir.
    """
    loglevel = config.get('loglevel', 'INFO')  # Here, give a default value if there is no definition
    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: {}'.format(loglevel))

    log_format = logging.Formatter('%(asctime)s %(message)s')
    logger = logging.getLogger()
    logger.setLevel(numeric_level)

    file_handler = logging.FileHandler(osp.join(config.log_dir,
                                                '{}.log'.format(osp.basename(config.log_dir))))
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    file_handler = logging.StreamHandler(sys.stdout)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    logging.root = logger
    logging.info("save log, checkpoint and code to: {}".format(config.log_dir))


def main(config):
    logging.info("========= Start training =========")
    for epoch in range(config.epochs):
        logging.info(f"===> Epoch {epoch}")
        summary_writer.add_scalar('train/loss', 1.1**epoch, epoch)
    logging.info("========= Training End =========")
    logging.info("Congrats! Your loss has successfully exploded!")


if __name__ == "__main__":
    opt, config = parse_option()

    local_aggregation_cfg = config.model.sa_config.local_aggregation
    tags = [config.data.datasets,
            'train',
            opt.cfg.split('.')[-2].split('/')[-1],
            local_aggregation_cfg.type,
            local_aggregation_cfg.feature_type,
            local_aggregation_cfg.reduction,
            f'C{config.model.width}', f'L{local_aggregation_cfg.layers}', f'D{config.model.depth}',
            f'B{config.batch_size}', f'LR{config.optimizer.lr}',
            f'Epoch{config.epochs}', f'Seed{config.rng_seed}'
            ]
    generate_exp_directory(config, tags)
    config.wandb.tags = tags
    config.wandb.name = config.logname

    # dump the config to one file
    cfg_path = os.path.join(config.log_dir, "config.json")
    with open(cfg_path, 'w') as f:
        json.dump(vars(opt), f, indent=2)
        json.dump(vars(config), f, indent=2)
        os.system('cp %s %s' % (opt.cfg, config.log_dir))
    config.cfg_path = cfg_path

    # set up logging
    setup_logger(config)
    logging.info(config)

    # init wandb *FIRST*
    if config.wandb.use_wandb:
        assert config.wandb.entity is not None
        Wandb.launch(config, config.wandb.use_wandb)
        logging.info(f"Launch wandb, entity: {config.wandb.entity}")
    # then init tensorboard
    summary_writer = SummaryWriter(log_dir=config.log_dir)

    # run a toy example
    main(config)

