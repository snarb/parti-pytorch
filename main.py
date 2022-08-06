from parti_pytorch import VitVQGanVAE, VQGanVAETrainer
from vx_config import *

vit_vae = VitVQGanVAE(
    dim = CROP_SIZE,               # dimensions
    image_size = CROP_SIZE,        # target image size
    patch_size = 8,
    #patch_size = 16,         # size of the patches in the image attending to each other
    num_layers = 3,           # number of layers
    discr_layers = 3
).cuda()

trainer = VQGanVAETrainer(
    vit_vae,
    folder = '/home/brans/repos/spaces_dataset-master/data/800',
    num_train_steps = 100000,
    lr = 3e-4,
    batch_size = 5,
    grad_accum_every = 8,
    amp = True
)

trainer.train()