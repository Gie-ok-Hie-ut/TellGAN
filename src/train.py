import time
from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
from data.video.transform.localizeface import LocalizeFace
from data.grid_loader import GRID
from torchvision import transforms

if __name__ == '__main__':
    opt = TrainOptions().parse()

    #normMean = [0.49139968, 0.48215827, 0.44653124]
    #normStd = [0.24703233, 0.24348505, 0.26158768]
    #normTransform = transforms.Normalize(normMean, normStd)
    face_size=128
    frame_transforms = transforms.Compose([
        LocalizeFace(height=face_size,width=face_size),
        transforms.ToTensor()#,
        #normTransform
    ])

    dataset = GRID(opt.dataroot, transform=frame_transforms)
    dataset_size = len(dataset)
    print('#training images = %d' % dataset_size)

    model = create_model(opt)
    visualizer = Visualizer(opt)
    total_steps = 0

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for vid_idx, video in enumerate(dataset):
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            visualizer.reset()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize

            init_tensor=True
            # frame is a tuple (frame_img, frame_word)
            for frame_idx, frame in enumerate(video):

                (img, trans) = frame
                if img.size(1) is not face_size or img.size(2) is not face_size or trans is None:
                    print "[Incomplete Frame came..]"
                    init_tensor=True
                    continue

                model.set_input(frame)
                model.optimize_parameters(init_tensor)
                init_tensor=False
                if frame_idx%20 == 0:
                    init_tensor=True

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
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save('latest')

                iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save('latest')
            model.save(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()

