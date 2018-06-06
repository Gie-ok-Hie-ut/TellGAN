import os
import time
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
from util import html
from data.video.transform.localizeface import LocalizeFace, FeaturePredictor
from data.grid_loader import GRID
from torchvision import transforms
import skvideo.io
from PIL import Image
import numpy as np



if __name__ == '__main__':
    opt = TestOptions().parse()

    face_size = (128, 128)
    toTensor = transforms.ToTensor()
    face_predictor_path = './shape_predictor_68_face_landmarks.dat'
    localizer = LocalizeFace(height=face_size[0], width=face_size[1], predictor_path=face_predictor_path,
                             mouthonly=True)
    frame_transforms = transforms.Compose([
        # localizer,
        # LocalizeFace(height=face_size,width=face_size),
        # transforms.ToTensor()#,
        # normTransform
    ])

    # Number of landmarks to detect
    nFeaturePoints = 20

    landmarkSuite = FeaturePredictor(face_predictor_path)

    dataset = GRID(opt.dataroot, transform=frame_transforms)
    dataset_size = len(dataset)
    print('#training images = %d' % dataset_size)

    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip


    model = create_model(opt, landmarkSuite=landmarkSuite)
    visualizer = Visualizer(opt)
    # create website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

    # test
    for vid_idx, video in enumerate(dataset):
        vid_start_time = time.time()
        iter_data_time = time.time()
        if vid_idx >= opt.how_many:
            break
        init_tensor = True
        vid_path = "./output/test/test_{0}.mp4".format(vid_idx)
        writer = skvideo.io.FFmpegWriter(vid_path)
        last_word = None

        for frame_idx, frame in enumerate(video):
            iter_start_time = time.time()
            t_data = iter_start_time - iter_data_time

            (img, word) = frame

            #img = Image.open("/home/jake//classes/cs703/Project/choi.jpg")

            ############ Landmarks and localized frame ###############
            mat_img = landmarkSuite.pilToMat(img)
            localizedFrame, feat0 = localizer.localize(mat_img, mouthonly=False)

            # check if Features are detected
            exptected_nlmks = 68
            if feat0 is None or exptected_nlmks > feat0.shape[0]:
                shape = None if feat0 is None else feat0.shape
                print("Initializing State: {}".format(shape))
                init_tensor = True
                continue

            # We only want to use the mouth landmarks
            feat0 = landmarkSuite.extractMouthFeatures(feat0)

            # localizedFrame is PIL image
            pil_localizedFrame = landmarkSuite.matToPil(localizedFrame)
            imgT = toTensor(pil_localizedFrame)

            #######################################

            # Exception vs frame size
            if localizedFrame.shape[1] is not face_size[0] or localizedFrame.shape[0] is not face_size[1]:
                print("[Incomplete Frame] {0} Size: {1}".format(frame_idx, img.size))
                init_tensor = True
                continue

            # Exception - word
            if word == "sil":
                continue

            if init_tensor is True:
                last_word = word

            wordChange = word is not last_word

            input = (imgT, word, feat0)
            model.set_input(input)
            pred_frame = model.test(init_tensor, wordChage=wordChange)
            init_tensor=False

            writer.writeFrame(pred_frame)
            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(vid_idx, frame_idx, errors, t, t_data)

            if opt.display_id > 0:
                visualizer.plot_current_errors(vid_idx, float(frame_idx) / len(video), opt, errors)

            # Set up next frame
            last_word = word

        writer.close()
        visuals = model.get_current_visuals()
        #img_path = model.get_image_paths()
        print('%04d: process video... %s' % (vid_idx, vid_path))
        #visualizer.save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio)

    webpage.save()
