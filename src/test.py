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
import cv2
import skvideo.io
from PIL import Image
import numpy as np

"""
('img_init', img_init),
('img_cur', img_cur),
('img_predict', img_predict),
('lnmk_cur', lnmk_cur),
('lnmk_predict', lnmk_predict)])
"""
def gen_out_frame(imgs, word):
    init = imgs['img_init'].copy()
    gt = np.vstack((imgs['img_cur'].copy(), imgs['lnmk_cur'].copy()))
    pred = np.pad(np.vstack((imgs['img_predict'].copy(), imgs['lnmk_predict'].copy())),
                  pad_width=((0, 0), (0, 10), (0, 0)), mode='constant', constant_values=0)

    compare = np.hstack((pred,gt))

    padb = int(init.shape[0])

    padded_init = np.pad(init, pad_width=((0,padb),(0,10),(0,0)), mode='constant', constant_values=0)

    result = np.pad(np.hstack((padded_init,compare)), pad_width=((64,0),(0,0),(0,0)),
                    mode='constant', constant_values=0)
    # Put Word
    text_loc = (4, 256)
    cv2.putText(result, word, text_loc, cv2.FONT_HERSHEY_SIMPLEX,
                thickness=2, color=(255, 255, 255),  fontScale=1.3) \

    # Put Init label
    gt_loc = (18, 40)
    cv2.putText(result, "Initial", gt_loc, cv2.FONT_HERSHEY_SIMPLEX,
                thickness=1, color=(255, 255, 255),  fontScale=1)
    # Put Predict label
    gt_loc = (148, 40)
    cv2.putText(result, "Predict", gt_loc, cv2.FONT_HERSHEY_SIMPLEX,
                thickness=1, color=(255, 255, 255),  fontScale=1)

    # Put GT label
    pred_loc = (320, 40)
    cv2.putText(result, "GT", pred_loc, cv2.FONT_HERSHEY_SIMPLEX,
                thickness=1, color=(255, 255, 255), fontScale=1)

    return result


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
<<<<<<< HEAD
    outputdict = {
        '-vcodec': 'libx264',  # use the h.264 codec
        '-crf': '0',  # set the constant rate factor to 0, which is lossless
        '-preset': 'slow'  # the slower the better compression, in princple, try
        # other options see https://trac.ffmpeg.org/wiki/Encode/H.264
    }
=======

>>>>>>> 57b94183fdda84d12e8a36060211d08f417f161f
    out_frames = []
    # test
    for vid_idx, video in enumerate(dataset):
        vid_start_time = time.time()
        iter_data_time = time.time()
        if vid_idx >= opt.how_many:
            break
        init_tensor = True
        vid_path = "./output/test/test_{0}.mp4".format(vid_idx)
        last_word = None
<<<<<<< HEAD
        """
        writer = skvideo.io.FFmpegWriter(vid_path, outputdict={
            '-vcodec': 'libx264',  # use the h.264 codec
            '-crf': '0',  # set the constant rate factor to 0, which is lossless
            '-preset': 'slow'  # the slower the better compression, in princple, try
            # other options see https://trac.ffmpeg.org/wiki/Encode/H.264
        })
        """
        out_frames = []
        vid_error = []
=======

>>>>>>> 57b94183fdda84d12e8a36060211d08f417f161f
        for frame_idx, frame in enumerate(video):
            iter_start_time = time.time()
            t_data = iter_start_time - iter_data_time

<<<<<<< HEAD
            (img, align) = frame

            word = None
            word_nframes = 0
            if align is not None:
                word = align.word
                word_nframes = align.end - align.start

            #img = Image.open("/home/jake//classes/cs703/Project/choi.jpg")

            # If we have a problem with the video, reinitialize at next best frame
            if word is None:
                print("Incomplete Frame: {0} Size: {1} Word: {2}".format(frame_idx, img.size, word))
                init_tensor = True
                continue

=======
            (img, word) = frame

            #img = Image.open("/home/jake//classes/cs703/Project/choi.jpg")

>>>>>>> 57b94183fdda84d12e8a36060211d08f417f161f
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

<<<<<<< HEAD
            input = (imgT, word, feat0, word_nframes)
=======
            input = (imgT, word, feat0)
>>>>>>> 57b94183fdda84d12e8a36060211d08f417f161f
            model.set_input(input)
            pred_frame = model.test(init_tensor, wordChage=wordChange)
            init_tensor=False

            errors = model.get_current_errors()
<<<<<<< HEAD
            vid_error.append(errors['MSE'])
=======
>>>>>>> 57b94183fdda84d12e8a36060211d08f417f161f
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(vid_idx, frame_idx, errors, t, t_data)

            if opt.display_id > 0:
                visualizer.plot_current_errors(vid_idx, float(frame_idx) / len(video), opt, errors)

            # Set up next frame
            last_word = word
            visuals = model.get_current_visuals()
<<<<<<< HEAD
            #writer.writeFrame(gen_out_frame(visuals, word))
=======
>>>>>>> 57b94183fdda84d12e8a36060211d08f417f161f
            out_frames.append(gen_out_frame(visuals, word))

        #outputdata = np.expand_dims(np.array(out_frames), axis=3)
        outputdata = np.array(out_frames)
<<<<<<< HEAD
        skvideo.io.vwrite(vid_path.format(vid_idx), outputdata, outputdict=outputdict)
=======
        skvideo.io.vwrite(vid_path.format(vid_idx), outputdata)
>>>>>>> 57b94183fdda84d12e8a36060211d08f417f161f

        #writer.close()
        #img_path = model.get_image_paths()
        print('%04d: process video... %s' % (vid_idx, vid_path))
<<<<<<< HEAD

        print("MSELoss for Video {0}: {1}", sum(vid_error)/len(vid_error))
=======
>>>>>>> 57b94183fdda84d12e8a36060211d08f417f161f
        #visualizer.save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio)

    webpage.save()
