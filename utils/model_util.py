import config
from model.uncond_mdm import MDM
from diffusion import guassian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps


class ModelArgs:

    def get_model_args():
        # SMPL defaults ##FIND ADDB DEFAULTS
        data_rep = 'rot6d'
        njoints = 25
        nfeats = 6 
        latent_dim = 256
        layers = 8
        dataset = 'addb'

        return {'modeltype': '', 'njoints': njoints, 'nfeats': nfeats,
            'translation': True, 'pose_rep': 'rot6d', 'glob': True, 'glob_rot': True,
            'latent_dim': latent_dim, 'ff_size': 1024, 'num_layers': layers, 'num_heads': 4,
            'dropout': 0.1, 'activation': "gelu", 'data_rep': data_rep,'dataset': dataset}



def create_model_and_diffusion():
    model = MDM(ModelArgs.get_model_args())
    diffusion = create_gaussian_diffusion()
    return model, diffusion

def create_gaussian_diffusion():
    # default params
    predict_xstart = True  # we always predict x_start (a.k.a. x0), that's our deal!
    steps = config.DIFFUSION_STEPS
    scale_beta = 1.  # no scaling
    timestep_respacing = ''  # can be used for ddim sampling, we don't use it.
    learn_sigma = False
    rescale_timesteps = False

    betas = gd.get_named_beta_schedule(config.NOISE_SCHEDULE, steps, scale_beta)
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not config.SIGMA_SMALL
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        lambda_vel=config.LAMBA_VEL,
        lambda_rcxyz=config.LAMBDA_RCXYZ,
        lambda_fc=config.LAMBA_FC,
    )
