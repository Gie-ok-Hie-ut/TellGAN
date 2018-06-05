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
from models.networks import NextFrameConvLSTM, NLayerDiscriminator, NextFeaturesForWord
from util.visualizer import Visualizer
from data.video.transform.localizeface import LocalizeFace, FeaturePredictor
from data.grid_loader import GRID
import torch
import torch.nn as nn
from torchvision import transforms
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from PIL import Image
import skvideo.io

class FullPaths(argparse.Action):
    """Expand user- and relative-paths"""
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, os.path.abspath(os.path.expanduser(values)))

def is_dir(dirname):
    """Checks if a path is an actual directory"""
    if not os.path.isdir(dirname):
        msg = "{0} is not a directory".format(dirname)
        raise argparse.ArgumentTypeError(msg)
    else:
        return dirname

def is_file(fname):
    """Checks if a path is an actual directory"""
    if not os.path.isfile(fname):
        msg = "{0} is not a file".format(fname)
        raise argparse.ArgumentTypeError(msg)
    else:
        return fname




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
    print("Saving... {}".format(save_path))
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
    parser.add_argument("--continue", action="store_true", dest="isContinue", default=False,
                        help="Continue Training, loads model, use with --ep-start and --vid-start")
    parser.add_argument("--mouth", action="store_true", dest="mouthonly", default=False,
                        help="localize mouth features only")
    parser.add_argument("--all-feat", action="store_true", dest="allFeatures", default=False,
                        help="Train and test using all features")
    parser.add_argument("--localize", action="store_true", dest="localize", default=False,
                        help="localize")
    parser.add_argument("--dataroot", action=FullPaths, type=is_dir, dest="dataroot", default="./",
                        help="data dir")
    parser.add_argument("--ckptdir", action=FullPaths, dest="chkptdir", default="./chkpt",
                        help="directory for loading/saving checkpoints")
    parser.add_argument("--outdir", action=FullPaths, dest="outdir", default="./out",
                        help="directory for ouput of videos and images from testing/training")
    parser.add_argument("--features-model",  dest="face_predictor_path",
                        default="./shape_predictor_68_face_landmarks.dat",
                        help="directory for loading/saving checkpoints")
    parser.add_argument("--ep-start", type=int, dest="ep_start",
                        default=0,
                        help="epoch for when to continue training")
    parser.add_argument("--vid-start", type=int, dest="vid_start",
                        default=0,
                        help="video for when to continue training")
    parser.add_argument("--save-freq", type=int, dest="saveFreq", default=500,
                        help="Frequency to save backup ceckpoints, latest points saved every 20, for training")
    parser.add_argument("--vid-freq", type=int, dest="vidFreq", default=20,
                        help="Frequency to save videos, for testing")
    parser.add_argument("--nosave", action="store_false", dest="isSave", default=True,
                        help="Do not save ceckpoints, for debug")
    return parser.parse_args()

def init_hidden(layers, hidden_dim):
    # Before we've done anything, we dont have any hidden state.
    # Refer to the Pytorch documentation to see exactly
    # why they have this dimensionality.
    # The axes semantics are (num_layers, minibatch_size, hidden_dim)
    return (torch.zeros(layers, 1, hidden_dim).cuda(),
            torch.zeros(layers, 1, hidden_dim).cuda())

def distLoss(fake, gt):
    loss = torch.sum(fake - gt)
    return loss



# Normalize and de-normalize
def normalize(data_points, img_size, scale_down=True):
    if(scale_down==True):
        data_points[:,0] = [[float(data[0])/img_size[1], float(data[1])/faceimg_size_size[0]] for data in data_points[:,0]]
    else:
        data_points[:] = [[int(data[0])*img_size[1], int(data[1])*img_size[0]] for data in data_points[:,0]]

if __name__ == '__main__':

    # Setup input args
    args = get_arguments()

    isTrain = args.isTrain
    dataroot = args.dataroot
    save_dir = args.chkptdir
    output_dir = args.outdir
    face_predictor_path = args.face_predictor_path
    ep_start = args.ep_start
    vid_start = args.vid_start
    isSave = args.isSave
    islocalize = args.localize
    isMouthOnly = args.mouthonly
    save_freq = args.saveFreq
    vid_save_freq = args.vidFreq

    #force continue to be false if not training
    isContinue = args.isContinue if isTrain is True else False

    if isTrain:
        print("Training(Continue:{0},ep:{1},vid{2})...".format(isContinue,ep_start,vid_start))
        if not isSave:
            print("WARNING: Not saving checkpoints!")
    else:
        print("Testing...")

    # Name output dir accoding to action
    mode = "train" if isTrain else "test"
    output_dir = '{0}_{1}'.format(output_dir,mode)

    # Initilize video save path
    vid_path = "%s/pred_mask_{0}.mp4" % output_dir

    # create missing dirs
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Default Face size
    face_size=(288, 360)

    # Face size changes depending on localization
    if islocalize:
        face_size = (128,128) if isMouthOnly is False else (50, 100)

    # TODO: Currently we always only use mouth landmars, add option to use all
    # number of xy feature points (xyxyxyxyxyxyxy...)
    nFeaturePoints = 20 #68 if isMouthOnly is False else 20

    # Init Image Transforms, localizer and Feature Detector
    toTensor = transforms.ToTensor()
    localizer = LocalizeFace(height=face_size[0], width=face_size[1],
                             predictor_path=face_predictor_path, mouthonly=isMouthOnly)

    landmarkSuite = FeaturePredictor(face_predictor_path)

    frame_transforms = transforms.Compose([
        #LocalizeFace(height=face_size,width=face_size),
        #toTensor#,
        #normTransform
    ])

    # Setup dataset
    dataset = GRID(dataroot, transform=frame_transforms)
    dataset_size = len(dataset)
    print('#training images = %d' % dataset_size)

    # Setup word embedding
    word_dim = nFeaturePoints*2
    embeds = nn.Embedding(100, word_dim)  # 100 words in vocab,  dimensional embeddings
    word_to_ix = {}

    # Initialize Models
    hidden_layers=3
    model = NextFeaturesForWord(input_size=(nFeaturePoints*2), hidden_size=nFeaturePoints*2, num_layers=hidden_layers)

    if isTrain is False or isContinue is True:
        which_epoch = 'latest'
        load_network(model, 'FeaturePointLSTM', epoch_label=which_epoch, save_dir=save_dir)

    model.cuda()

    # Initialize Loss and optimizers
    crit = nn.MSELoss()
    crit.cuda()

    if isTrain is True:
        optimizer_G = optim.Adam(itertools.chain(model.parameters()))
        scheduler_G = lr_scheduler.StepLR(optimizer_G, step_size=5, gamma=0.1)

    total_steps = 0

    for epoch in range(ep_start, 100):

        if isTrain is True:
            scheduler_G.step()
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for vid_idx in range(vid_start, dataset_size):
            # Get video
            video = dataset[vid_idx]
            iter_start_time = time.time()
            #if total_steps % opt.print_freq == 0:
            #    t_data = iter_start_time - iter_data_time

            init_tensor=True
            prev_feat_seq = None
            prev_mask = None
            prev_frame = None
            prev_pred_featT = None
            vid_loss = []

            sample_frames = []

            # frame is a tuple (frame_img, frame_word)
            last_word = None
            for frame_idx, frame in enumerate(video):
                mask = None

                if isTrain is True:
                    optimizer_G.zero_grad()

                (img, trans) = frame
                mat_img = landmarkSuite.pilToMat(img)
                imgT = toTensor(img)

                # If we have a problem with the video, reinitialize at next best frame
                if trans is None:
                    print("Incomplete Frame: {0} Size: {1} Word: {2}".format(frame_idx, imgT.size(), trans))
                    prev_feat_seq=None
                    word_seq=None
                    init_tensor=True
                    continue

                # Add Word to dictionary if not seen befor
                if trans not in word_to_ix:
                    word_to_ix[trans] = len(word_to_ix)

                # Init to None for checking success
                feat0 = None
                if islocalize:
                    # Localize/scale the face and return feature points for new image size
                    localizedFrame, feat0 = localizer.localize(mat_img, mouthonly=isMouthOnly)

                    # We only want to use the mouth landmarks
                    if not isMouthOnly:
                        feat0 = landmarkSuite.extractMouthFeatures(feat0)

                    # localizedFrame is PIL image
                    pil_localizedFrame = landmarkSuite.matToPil(localizedFrame)

                    # Create mask from features, convert to PIL
                    mask = landmarkSuite.matToPil(
                        landmarkSuite.create_mask(size=localizedFrame.shape[0:2],
                                                  features=feat0)
                    )

                    #mask.show()
                    #frame.show()
                else:
                    # Not localizing, just return mask and features unchanged
                    feat0, mask = landmarkSuite.getFeatureMask(mat_img, mouthonly=isMouthOnly)

                if feat0 is None or nFeaturePoints > feat0.shape[0]:
                    shape = None if feat0 is None else feat0.shape
                    print("Initializing State: {}".format(shape))
                    init_tensor=True
                    prev_feat_seq=None
                    word_seq=None
                    continue

                #normalize between 0 and 1
                feat0_norm = feat0.copy()
                feat0_norm[:,:,1] /= face_size[0]
                feat0_norm[:,:,0] /= face_size[1]

                # Flatten 2D (x,y) coords to 1-D [xyxyxy]
                featTB4 = Variable(torch.from_numpy(feat0_norm))
                featT = featTB4.view(featTB4.numel())

                # Create word tensor from embedding
                lookup_tensor = torch.LongTensor([word_to_ix[trans]])
                transT= embeds(Variable(lookup_tensor))

                # INitialize the input with ground truth only
                # Training: always initilize with GT
                # Testing: TRY to always initilize with predicted (except errors or frist frame)
                if init_tensor == True or last_word is not trans:

                    # If we are training or there is no previous predected lmks, initialize with GT
                    # Initilize with predicted otherwise
                    if isTrain or prev_feat_seq is None:
                        init_feature = featT.unsqueeze(0)
                        prev_feat_seq = torch.cat((transT, init_feature), 0)

                        if vid_idx % vid_save_freq == 0:
                            sample_frames.append(np.concatenate((mask.copy(), mask.copy()), axis=1))
                        last_word = trans
                        init_tensor = False
                        continue
                    else:
                        init_feature = prev_feat_seq[-1].unsqueeze(0)

                        prev_feat_seq = torch.cat((transT, init_feature), 0)

                # Input is created over time, create batch dim and use GPU
                input = prev_feat_seq.unsqueeze(1).cuda()

                # Detach input so it is not back-propigated since we use it over time
                pred_featT = model(input.detach())

                # Loss Predicted - Groundtruth
                loss = crit(pred_featT, featT.unsqueeze(0).cuda())

                # Running Video loss
                vid_loss.append(loss.clone().data.cpu().numpy())

                # If we are training we want to back-propigate the weights
                if isTrain:
                    loss.backward()
                    optimizer_G.step()

                # Add GT (Training) or Predicted (Testing) landmarks to LSTM input
                if isTrain:
                    prev_feat_seq = torch.cat((prev_feat_seq, featT.unsqueeze(0)), 0)
                else:
                    prev_feat_seq = torch.cat((prev_feat_seq, pred_featT.cpu()), 0)

                # Save predicted frames to video
                if vid_idx % vid_save_freq == 0:
                    pred_featT2d = pred_featT.clone().view(nFeaturePoints,1,2)
                    #pred_featT2d = pred_featT.clone().view(nFeaturePoints, 1, 2)
                    # denormalize
                    pred_featT2d[:,:,1] *= (face_size[0])
                    pred_featT2d[:,:,0] *= (face_size[1])
                    pred_feat = pred_featT2d.data.cpu().numpy()
                    pred_mask = landmarkSuite.create_mask(size=(mask.size[1], mask.size[0]),
                                                          features=pred_feat)

                    pil_pred_mask = landmarkSuite.matToPil(pred_mask)
                    sample_frames.append(np.concatenate((mask.copy(), pil_pred_mask.copy()), axis=1))

                # Setup next iteration
                last_word = trans

            # Save Video
            if vid_idx % vid_save_freq == 0:
                outputdata = np.expand_dims(np.array(sample_frames), axis=3)
                skvideo.io.vwrite(vid_path.format(vid_idx), outputdata)

            if isTrain and isSave:
                if vid_idx % 20 == 0:
                    save_network(model, 'FeaturePointLSTM', epoch_label='latest', save_dir=save_dir)

                if vid_idx % save_freq == 0:
                    save_network(model, 'FeaturePointLSTM',
                                 epoch_label="ep{0}_{1}".format(epoch,vid_idx),
                                 save_dir=save_dir)
            avg_loss = 0
            if len(vid_loss) > 1:
                avg_loss = sum(vid_loss) / len(vid_loss)

            print("ep: {0}, video: {1}, Loss: {2}".format(epoch, vid_idx, avg_loss))
            print("===========================")

        if isTrain and isSave:
            save_network(model, 'FeaturePointLSTM',
                         epoch_label="ep{0}".format(epoch),
                         save_dir=save_dir)
        else:
            # If we are testing, no need to go through the dataset again for another epoch
            break
    if isTrain and isSave:
        save_network(model, 'FeaturePointLSTM',
                     epoch_label="complete",
                     save_dir=save_dir)