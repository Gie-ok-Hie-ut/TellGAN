import os
import time
from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
from data.video.transform.localizeface import LocalizeFace, FeaturePredictor
from data.grid_loader import GRID
from torchvision import transforms
import skvideo.io
import numpy as np



if __name__ == '__main__':
    opt = TrainOptions().parse()

    #normMean = [0.49139968, 0.48215827, 0.44653124]
    #normStd = [0.24703233, 0.24348505, 0.26158768]
    #normTransform = transforms.Normalize(normMean, normStd)
    face_size = (128, 128)
    toTensor=transforms.ToTensor()
    face_predictor_path = './shape_predictor_68_face_landmarks.dat'
    localizer = LocalizeFace(height=face_size[0], width=face_size[1], predictor_path=face_predictor_path,
                             mouthonly=True)

    frame_transforms = transforms.Compose([
    	#localizer,
        #LocalizeFace(height=face_size,width=face_size),
        #transforms.ToTensor()#,
        #normTransform
    ])

    # Number of landmarks to detect
    nFeaturePoints = 20

    landmarkSuite = FeaturePredictor(face_predictor_path)

    dataset = GRID(opt.dataroot, transform=frame_transforms)
    dataset_size = len(dataset)
    print('#training images = %d' % dataset_size)

    model = create_model(opt, landmarkSuite=landmarkSuite)
    visualizer = Visualizer(opt)
    total_steps = 0



    # Train
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for vid_idx in range(0,dataset_size): #13549,dataset_size):
            video = dataset[vid_idx]
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            visualizer.reset()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize

            init_tensor=True
            last_word = None
            # frame is a tuple (frame_img, frame_word)
            for frame_idx, frame in enumerate(video):
                (img, align) = frame

                word = None
                word_nframes = 0
                if align is not None:
                    word = align.word
                    word_nframes = align.end - align.start
                    #print(word_nframes)

                # If we have a problem with the video, reinitialize at next best frame
                if word is None:
                    print("Incomplete Frame: {0} Size: {1} Word: {2}".format(frame_idx, imgT.size(), word))
                    init_tensor = True
                    continue

                ############ Landmarks and localized frame ###############
                mat_img = landmarkSuite.pilToMat(img)
                localizedFrame, feat0 = localizer.localize(mat_img, mouthonly=False)

                # check if Features are detected
                exptected_nlmks = 68
                if feat0 is None or exptected_nlmks > feat0.shape[0]:
                    shape = None if feat0 is None else feat0.shape
                    print("Missing landmarks, Initializing State: {}".format(shape))
                    init_tensor=True
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
                    init_tensor=True
                    continue

                # Exception - word
                if word == "sil":
                    continue


                # Exception - dic size                
                if len(model.get_dic()) >= model.get_dic_size() and model.get_dic().get(word, -1) == -1:
                    print("[Dictionary Full] Frame: {0} Word: {1}".format(frame_idx, word))
                    init_tensor=True
                    continue

                if word is not last_word:
                    print("Learning: {0}".format(word))
                    init_tensor = True

                # Train
                input = (imgT, word, feat0, word_nframes)
                model.set_input(input)
                model.optimize_parameters(init_tensor)
                init_tensor=False

                #Set up next frame
                last_word = word


            if total_steps % opt.display_freq == 0:
                save_result = total_steps % opt.update_html_freq == 0
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_steps % opt.print_freq == 0:
                errors = model.get_current_errors()
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_errors(epoch, epoch_iter, errors, t, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_errors(epoch, float(epoch_iter) / dataset_size, opt, errors)

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d, video %d)' %
                      (epoch, total_steps, vid_idx))
                model.save('latest')

                iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d, video %d)' %
                  (epoch, total_steps, vid_idx))
            model.save('latest')
            model.save(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()

