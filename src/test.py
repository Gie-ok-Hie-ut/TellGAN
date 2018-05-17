import os
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
from util import html
from data.video.transform.localizeface import LocalizeFace
from data.grid_loader import GRID
from torchvision import transforms
import skvideo.io
import numpy as np



if __name__ == '__main__':
    opt = TestOptions().parse()

    face_size=128
    frame_transforms = transforms.Compose([
        LocalizeFace(height=face_size,width=face_size),
        transforms.ToTensor()
    ])

    dataset = GRID(opt.dataroot, transform=frame_transforms)
    dataset_size = len(dataset)
    print('#training images = %d' % dataset_size)

    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip


    model = create_model(opt)
    visualizer = Visualizer(opt)
    # create website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))


    # test
    for vid_idx, video in enumerate(dataset):
        if vid_idx >= opt.how_many:
            break
        init_tensor = True
        vid_path = "/home/jake/classes/cs703/Project/dev/TellGAN/src/output/train/test_{0}.mp4"
        writer = skvideo.io.FFmpegWriter("/home/jake/classes/cs703/Project/dev/TellGAN/src/output/train/test_{0}.mp4".format(vid_idx))

        for frame_idx, frame in enumerate(video):
            (img, trans) = frame
            if img.size(1) is not face_size or img.size(2) is not face_size or trans is None:
                print("Incomplete Frame: {0} Size: {1} Word: {2}".format(frame_idx, img.size(), trans))
                init_tensor = True
                continue

            if frame_idx % 40 == 0:
                init_tensor = True

            model.set_input(frame)
            pred_frame = model.test(init_tensor)
            init_tensor=False

            writer.writeFrame(pred_frame)

        writer.close()
        visuals = model.get_current_visuals()
        #img_path = model.get_image_paths()
        print('%04d: process video... %s' % (vid_idx, vid_path))
        #visualizer.save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio)

    webpage.save()
