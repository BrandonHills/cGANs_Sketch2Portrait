import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util import html

opt = TestOptions().parse()
print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
print("opt:", opt)

# opt = Namespace(aspect_ratio=1.0, batchSize=1, checkpoints_dir='./checkpoints', dataroot='./datasets/faces', dataset_mode='aligned', display_id=1, display_port=8097, display_winsize=256, fineSize=256, gpu_ids=[], how_many=50, init_type='normal', input_nc=3, isTrain=False, loadSize=286, max_dataset_size=inf, model='pix2pix', nThreads=2, n_layers_D=3, name='faces_pix2pix', ndf=64, ngf=64, no_dropout=False, no_flip=False, norm='batch', ntest=inf, output_nc=3, phase='test', resize_or_crop='resize_and_crop', results_dir='./results/', serial_batches=False, which_direction='BtoA', which_epoch='latest', which_model_netD='basic', which_model_netG='unet_256')

opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
# test
for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break
    model.set_input(data)
    model.test()
    visuals = model.get_current_visuals()
    img_path = model.get_image_paths()
    print('%04d: process image... %s' % (i, img_path))
    visualizer.save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio)

webpage.save()
