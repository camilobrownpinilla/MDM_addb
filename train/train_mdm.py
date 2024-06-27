
# This code is based on https://github.com/openai/guided-diffusion
"""
Train a diffusion model on images.
"""

import os
import json
from utils.fixseed import fixseed
# from utils.parser_util import train_args
from utils import dist_util
from utils import config
from train.training_loop import TrainLoop
from data.dataloaders.get_data import get_dataset_loader, get_dataset
from utils.model_util import create_model_and_diffusion
from train.train_platforms import ClearmlPlatform, TensorboardPlatform, NoPlatform  # required for the eval operation
from tqdm import tqdm

def main():
    # args = train_args()
    fixseed(config.SEED)
    train_platform_type = eval(config.TRAIN_PLATFORM_TYPE)
    train_platform = train_platform_type(config.SAVE_DIR)
    # train_platform.report_args(args, name='Args')

    if config.SAVE_DIR is None:
        raise FileNotFoundError('save_dir was not specified.')
    elif os.path.exists(config.SAVE_DIR) and not config.OVERWRITE:
        raise FileExistsError('save_dir [{}] already exists.'.format(config.SAVE_DIR))
    elif not os.path.exists(config.SAVE_DIR):
        os.makedirs(config.SAVE_DIR)
    # args_path = os.path.join(config.SAVE_DIR, 'args.json')
    # with open(args_path, 'w') as fw:
    #     json.dump(vars(args), fw, indent=4, sort_keys=True)

    dist_util.setup_dist(0)

    print("creating data loader...")
    data = tqdm(get_dataset_loader(name=config.DATASET, batch_size=config.BATCH_SIZE, window_size=config.WINDOW_SIZE, split='train'))

    print("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(get_dataset(name=config.DATASET, window_size=config.WINDOW_SIZE, split='train'))
    model.to(dist_util.dev())
    # model.rot2xyz.smpl_model.eval()

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters_wo_clip()) / 1000000.0))
    print("Training...")
    TrainLoop(train_platform, model, diffusion, data).run_loop()
    train_platform.close()

if __name__ == "__main__":
    main()
