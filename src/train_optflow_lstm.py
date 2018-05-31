from __future__ import print_function

import time
import numpy as np
import cv2
import dlib
import argparse
import os

import itertools
from options.train_options import TrainOptions
from data import CreateDataLoader
from models.networks import NextFrameConvLSTM, NLayerDiscriminator
from util.visualizer import Visualizer
from data.video.transform.localizeface import LocalizeFace
from data.grid_loader import GRID
import torch
import torch.nn as nn
from torchvision import transforms
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from PIL import Image
import skvideo.io


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Denoise Autoencoder.")
    parser.add_argument("--train", action="store_true", dest="isTrain", default=True,
                        help="Train model, save checkpoints (Default)")
    parser.add_argument("--test", action="store_false", dest="isTrain", default=True,
                        help="Test model, load checkpoints")
    return parser.parse_args()

args = get_arguments()

class OpticalFlow(object):
    def __init__(self, feature_model=None):
        self.feature_model = feature_model
        self.face_detector = dlib.get_frontal_face_detector()
        self.feature_detector = dlib.shape_predictor(self.feature_model)

        # Parameters for lucas kanade optical flow
        self.lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    def run(self, pil0, pil1, features0=None):
        #mat0 = self.pilToMat(pil0)
        gray0 = cv2.cvtColor(pil0, cv2.COLOR_BGR2GRAY)
        #mat1 = self.pilToMat(pil1)
        gray1 = cv2.cvtColor(pil1, cv2.COLOR_BGR2GRAY)
        features1 = None

        if features0 is None:
            features0 = self.getFeaturePoints(gray0)

        # May fail to get features, return None and blank mask
        if features0 is not None:
            features1, features0 = self.getFlow(gray0, gray1, features0)

        mask = self.create_mask(gray1, features1)

        features1 = features1.reshape(-1,1,2) if features1 is not None else None

        return features1, self.matToPil(mask)

    def getInit(self, pil0):
        #gray0 = cv2.cvtColor(self.pilToMat(pil0), cv2.COLOR_BGR2GRAY)
        gray0 = cv2.cvtColor(pil0, cv2.COLOR_BGR2GRAY)
        features0 = self.getFeaturePoints(gray0)

        mask = self.create_mask(gray0, features0)
        return features0, self.matToPil(mask)


    def getFlow(self, old_gray, frame_gray, features0):
        features1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, features0,
                                                      None, ** self.lk_params)
        # Select good points
        good_new = features1[st == 1]
        good_old = features0[st == 1]

        return good_new, good_old

    def create_mask(self, img, features):

        features = np.copy(features).astype(np.int)
        mask = np.zeros_like(img)
        if features is None:
            return mask
        mask[features[:,:,1], features[:,:,0]] = 255

        return mask


    def getFeaturePoints(self, img_grey):
        faces = self.face_detector(img_grey, 1)
        features = None

        for k, rect in enumerate(faces):
            # print("k: ", k)
            # print("rect: ", rect)
            # shape = self.predictor(img_gray, rect)
            features = self.feature_detector(img_grey, rect)
            break

        feat_points = np.asfarray([])
        if features is None:
            return None

        for i, part in enumerate(features.parts()):
            fpoint = np.asfarray([part.x, part.y])
            # filter if index values larger than image
            if (fpoint < 0).any() or fpoint[0] >= img_grey.shape[1] or fpoint[1] >= img_grey.shape[0]:
                print("ignoring point: {} | imgsize: {}".format(fpoint,img_grey.shape))
                continue
            if i is 0:
                feat_points = fpoint
            else:
                feat_points = np.vstack((feat_points, fpoint))
        feat_points = np.expand_dims(feat_points, axis=1)



        # print("face_points_shape: ", feat_points.shape)
        # print("feat_points: ", feat_points)
        return feat_points.astype(np.float32)

    def matToPil(self, mat_img):
        return Image.fromarray(mat_img)

    def pilToMat(self, pil_img):
        pil_image = pil_img.convert('RGB')
        open_cv_image = np.array(pil_image)
        # Convert RGB to BGR
        return open_cv_image  # [:, :, ::-1].copy()

def create_video(vid_path, vid_idx, save_freq):
    if vid_idx % save_freq != 0:
        return None

    return 1#skvideo.io.FFmpegWriter(vid_path)

# helper saving function that can be used by subclasses
def save_network(network, network_label, epoch_label):
    save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
    save_path = os.path.join(save_dir, save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available():
        network.cuda()


# helper saving function that can be used by subclasses
def save_network(network, network_label, epoch_label='latest', save_dir="./"):
    save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
    save_path = os.path.join(save_dir, save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available():
        network.cuda()

# helper loading function that can be used by subclasses
def load_network(network, network_label, epoch_label='latest', save_dir="./"):
    save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
    save_path = os.path.join(save_dir, save_filename)
    if os.path.isfile(save_path):
        print("Loading model: {}".format(save_path))
        network.load_state_dict(torch.load(save_path))
    else:
        print("Cannot Find Model: {}".format(save_path))
        exit(1)

def backwardG(fake, gt, word, optimizer_G, discriminator, crit_gan, critID, w_id=1):
    optimizer_G.zero_grad()
    # GAN Loss
    #loss_G = crit_gan(discriminator(fake), True)
    #fakeG = torch.cat((fake, word), 0)
    pred_fake = discriminator(fake.unsqueeze(0))
    true = torch.ones(pred_fake.shape).cuda()
    loss_G = crit_gan(pred_fake, true)

    # ID Loss
    loss_ID = critID(fake, gt)
    loss_G_total = loss_G + loss_ID * w_id
    loss_G_total.backward(retain_graph=True)
    optimizer_G.step()

    return loss_G_total, loss_G, loss_ID


def backwardD(fake, gt, word, optimizer_D, discriminator, crit_gan):
    # Train Discriminator
    optimizer_D.zero_grad()
    #gtD = torch.cat((gt, word), 0)
    pred_real = discriminator(gt.unsqueeze(0))
    true = torch.ones(pred_real.shape).cuda()
    loss_D_real = crit_gan(pred_real, true)

    # Fake
    #fakeD = torch.cat((fake, word), 0)
    pred_fake = discriminator(fake.detach().unsqueeze(0))
    false = torch.ones(pred_fake.shape).cuda()
    loss_D_fake = crit_gan(pred_fake, false)
    # Combined loss
    loss_D = (loss_D_real + loss_D_fake) * 0.5
    # backward
    loss_D.backward()# retain_graph=True)
    optimizer_D.step()
    return loss_D

if __name__ == '__main__':

    args = get_arguments()

    isTrain = args.isTrain
    dataroot = "/home/jake/classes/cs703/Project/data/grid/"
    if isTrain:
        print("Trainging...")
    else:
        print("Testing...")
        #dataroot = "/home/jake/classes/cs703/Project/data/grid_test/"

    save_dir = "./optflowGAN_chkpnts"

    mode = "train" if isTrain else "test"
    output_dir = './outGAN_{0}'.format(mode)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    face_size=(288, 360)
    save_freq=20

    face_predictor_path = '/home/jake/classes/cs703/Project/dev/TellGAN/src/assests/predictors/shape_predictor_68_face_landmarks.dat'

    toTensor = transforms.ToTensor()
    frame_transforms = transforms.Compose([
        #LocalizeFace(height=face_size,width=face_size),
        #toTensor#,
        #normTransform
    ])

    dataset = GRID(dataroot, transform=frame_transforms)
    dataset_size = len(dataset)
    print('#training images = %d' % dataset_size)

    model = NextFrameConvLSTM(input_size=face_size,input_dim=2,
                              num_layers=3,hidden_dim=[3,3,1],
                              kernel_size=(3,3), batch_first=True)
    discriminator = NLayerDiscriminator(input_nc=1)#, use_sigmoid=True)

    if isTrain is False:
        which_epoch = 'latest'
        load_network(model, 'OpticalFlowLSTM', epoch_label=which_epoch, save_dir=save_dir)
        #load_network(discriminator, 'OpticalFlow_D', epoch_label=which_epoch, save_dir=save_dir)

    model.cuda()
    discriminator.cuda()

    opticalFlow = OpticalFlow(face_predictor_path)
    embeds = nn.Embedding(100, 1)  # 100 words in vocab,  dimensional embeddings
    word_to_ix = {}

    crit_gan = nn.MSELoss()  # nn.BCEWithLogitsLoss() #GANLoss()
    crit_gan.cuda()

    critID = nn.MSELoss()  # nn.BCEWithLogitsLoss() #GANLoss()
    critID.cuda()

    if isTrain is True:
        optimizer_G = optim.Adam(itertools.chain(model.parameters()))
        scheduler_G = lr_scheduler.StepLR(optimizer_G, step_size=5, gamma=0.1)

        optimizer_D = optim.Adam(itertools.chain(discriminator.parameters()))
        scheduler_D = lr_scheduler.StepLR(optimizer_D, step_size=5, gamma=0.1)

    total_steps = 0

    vid_path = "%s/pred_mask_{0}.mp4" % output_dir

    for epoch in range(0, 100):

        if isTrain is True:
            scheduler_G.step()
            scheduler_D.step()
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for vid_idx, video in enumerate(dataset):
            iter_start_time = time.time()
            #if total_steps % opt.print_freq == 0:
            #    t_data = iter_start_time - iter_data_time

            #visualizer.reset()
            #total_steps += opt.batchSize
            #epoch_iter += opt.batchSize

            init_tensor=True
            prev_img_seq = None
            word_seq = None
            vid_loss = []
            #vidWriter = create_video(vid_path=vid_path.format(vid_idx), vid_idx=vid_idx, save_freq=20)
            sample_frames = []
            # frame is a tuple (frame_img, frame_word)
            for frame_idx, frame in enumerate(video):
                if isTrain is True:
                    optimizer_G.zero_grad()

                (img, trans) = frame
                imgT = toTensor(img)

                #if imgT.size(1) is not face_size[0] or imgT.size(2) is not face_size[1] or trans is None:
                if trans is None:
                    print("Incomplete Frame: {0} Size: {1} Word: {2}".format(frame_idx, imgT.size(), trans))
                    prev_img_seq=None
                    word_seq=None
                    init_tensor=True
                    continue

                if trans not in word_to_ix:
                    word_to_ix[trans] = len(word_to_ix)

                lookup_tensor = torch.LongTensor([word_to_ix[trans]])
                trans_embed = embeds(Variable(lookup_tensor))
                transT = trans_embed.repeat(imgT.size(1),imgT.size(2)).unsqueeze(0)

                #np_img = (img.permute(1,2,0).data.cpu().numpy()*255).astype(np.uint8)
                feat0, mask = opticalFlow.getInit(opticalFlow.pilToMat(img))
                #OpticalFlow.matToPil(mask).show()
                maskT = Variable(toTensor(mask))

                # INitialize the input with ground trouth only
                if (init_tensor == True):
                    prev_img_seq = maskT.unsqueeze(0)
                    init_tensor = False
                    if vid_idx % save_freq == 0:
                        sample_frames.append(np.concatenate((mask.copy(), mask.copy()), axis=1))
                    continue

                if word_seq is not None:
                    word_seq = torch.cat((word_seq, transT.unsqueeze(0)), 0)
                else:
                    word_seq = transT.unsqueeze(0)

                #Concat previous image and current word, add batch dim
                input = torch.cat((prev_img_seq, word_seq), 1).cuda()

                pred_maskT = model(input.detach())

                if vid_idx % save_freq == 0:
                    pred_mask = (pred_maskT.permute(1,2,0).data.cpu().numpy()*255).astype(np.uint8)
                    pil_pred_mask = opticalFlow.matToPil(np.squeeze(pred_mask, axis=2))
                    sample_frames.append(np.concatenate((mask.copy(), pil_pred_mask.copy()), axis=1))

                #mask.save("mask_{}.png".format(frame_idx))
                if isTrain:
                    prev_img_seq = torch.cat((prev_img_seq, maskT.unsqueeze(0)), 0)
                else:
                    prev_img_seq = torch.cat((prev_img_seq, pred_maskT.unsqueeze(0).cpu()), 0)

                # Train Generator (LSTM)
                if isTrain:
                    loss_G_total, loss_G, loss_ID = backwardG(pred_maskT, maskT.cuda(), transT.cuda(),
                                                              optimizer_G, discriminator, crit_gan, critID)
                    vid_loss.append(loss_ID.data.cpu().numpy())

                    loss_D = backwardD(pred_maskT, maskT.cuda(), transT.cuda(),
                                       optimizer_D, discriminator, crit_gan)

                    if frame_idx%20 == 0:
                        print("vid{0}: loss_G_total: {1} | loss_G: {2} | loss_ID: {3} | loss_D: {4}"
                              .format(vid_idx, loss_G_total, loss_G, loss_ID, loss_D))
                else:
                    loss = critID(pred_maskT, maskT.cuda())
                    vid_loss.append(loss.data.cpu().numpy())


                if frame_idx%25 == 0:
                    init_tensor=True
                    prev_img_seq=None
                    word_seq=None

            if vid_idx % save_freq == 0:
                outputdata = np.expand_dims(np.array(sample_frames), axis=3)
                skvideo.io.vwrite(vid_path.format(vid_idx), outputdata)

            if isTrain and vid_idx % 20 == 0:
                save_network(model, 'OpticalFlowLSTM', epoch_label='latest', save_dir=save_dir)
                save_network(model, 'OpticalFlowLSTM',
                             epoch_label="ep{0}_{1}".format(epoch,vid_idx),
                             save_dir=save_dir)

                save_network(discriminator, 'OpticalFlow_D', epoch_label='latest', save_dir=save_dir)
                save_network(discriminator, 'OpticalFlow_D',
                             epoch_label="ep{0}_{1}".format(epoch,vid_idx),
                             save_dir=save_dir)

            avg_loss = sum(vid_loss) / len(vid_loss)
            print("ep: {0}, video: {1}, Loss: {2}".format(epoch, vid_idx, avg_loss))
            print("===========================")

        if isTrain:
            save_network(model, 'OpticalFlowLSTM',
                         epoch_label="ep{0}".format(epoch),
                         save_dir=save_dir)
            save_network(discriminator, 'OpticalFlow_D',
                         epoch_label="ep{0}".format(epoch),
                         save_dir=save_dir)
        else:
            # If we are testing, no need to go through the dataset again for another epoch
            break
    if isTrain:
        save_network(model, 'OpticalFlowLSTM',
                     epoch_label="complete",
                     save_dir=save_dir)
        save_network(discriminator, 'OpticalFlowLSTM',
                     epoch_label="complete",
                     save_dir=save_dir)