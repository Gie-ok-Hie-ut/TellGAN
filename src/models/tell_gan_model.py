import torch
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import numpy as np


class TellGANModel(BaseModel):
    def name(self):
        return 'TellGANModel'

    def initialize(self, opt):

        BaseModel.initialize(self, opt)

        self.feature_size = 64
        self.lstm_in_dim = (self.feature_size, self.feature_size)
        self.lstm_in_nc = 257
        self.lstm_out_nc = [256]
        self.lstm_nlayers = 1
        self.lstm_kernel_size = (3, 3)
        self.netImgEncoder = networks.define_ImgEncoder(opt.input_nc, opt.output_nc, opt.ngf, 'resnet_3blocks_enc',
                                                        opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.netImgLSTM = networks.define_ConvLSTM(self.lstm_in_dim, self.lstm_in_nc, self.lstm_nlayers,
                                                   self.lstm_out_nc, self.lstm_kernel_size, self.gpu_ids)
        # self.netWordEmbed = networks.define_WordEmbed(64,64,1)
        self.netImgDecoder = networks.define_ImgDecoder(opt.input_nc, opt.output_nc, opt.ngf, 'resnet_3blocks_dec',
                                                        opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)

        # Load Dictionary
        self.dic_size = 500
        try:
            self.dictionary = np.load('grid_embedding.npy').item()
            print "[Dictionary] Loading Existing Embedding Dictionary"
        except IOError as e:
            self.dictionary = {'default': 0}
            print "[Dictionary] Building New Word Embedding Dictionary"


        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.which_model_netD, opt.n_layers_D, opt.norm,
                                          use_sigmoid, opt.init_type, self.gpu_ids)

        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netImgEncoder, 'ImgEncoder', which_epoch)
            self.load_network(self.netImgLSTM, 'ImgLSTM', which_epoch)
            # self.load_network(self.netWordEmbed, 'WordEmbed', which_epoch)
            self.load_network(self.netImgDecoder, 'ImgDecoder', which_epoch)

            if self.isTrain:
                self.load_network(self.netD, 'Discriminator', which_epoch)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionIdt = torch.nn.L1Loss()  # L1 Loss Okay?

            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(
                itertools.chain(self.netImgEncoder.parameters(), self.netImgLSTM.parameters(),
                                self.netImgDecoder.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netImgEncoder)
        networks.print_network(self.netImgLSTM)
        # networks.print_network(self.netWordEmbed)
        networks.print_network(self.netImgDecoder)
        if self.isTrain:
            networks.print_network(self.netD)
        print('-----------------------------------------------')

    def set_input(self, input):
        (input_frame, input_transcription) = input

        if len(self.gpu_ids) > 0:
            input_frame = input_frame.cuda(self.gpu_ids[0], async=True)

        self.input_frame = input_frame
        self.input_transcription = input_transcription

    def forward(self):
        self.img_input = Variable(self.input_frame)
        self.word_input = Variable(self.input_transcription)


    def Word2Tensor(self, word, width, height):
        word_cur = word

        # Update unseen word
        if self.dictionary.get(word_cur, -1) == -1:
            self.dictionary.update({word_cur: float((len(self.dictionary) + 1)) / self.dic_size})

        # Make Tensor
        vec2np = np.full((width, height), self.dictionary[word_cur])
        np2tensor = torch.from_numpy(vec2np).cuda().float()

        # Add One dim
        np2tensor_sq1 = np2tensor.unsqueeze(0)
        np2tensor_sq2 = np2tensor_sq1.unsqueeze(0)

        return np2tensor_sq2 # 1*1*38*38 (b*c*w*h)

    def test(self, init_tensor):

        img_input = Variable(self.input_frame, volatile=True)
        word_input = Variable(self.input_transcription, volatile=True)

        if init_tensor == True:
            self.img_init = img_input
            self.word_init = word_input

            # Different from Training.
            self.img_enc_stack = 0
            self.word_enc_stack = 0

            # Refresh All saved data
            self.convlstm_input = 0
            self.convlstm_output = 0

            # Redundant
            self.img_predict = img_input

        else:
            # Predict Current Img
            self.word_cur = self.word_input

            self.img_cur_enc = self.netImgEncoder(self.img_predict.unsqueeze(0))
            self.word_cur_enc = self.Word2Tensor(self.word_cur, self.feature_size, self.feature_size)

            self.img_enc_stack = torch.stack((self.img_enc_stack, self.img_cur_enc), 0)
            self.word_enc_stack = torch.stack((self.word_enc_stack, self.word_cur_enc), 0)

        self.convlstm_input = torch.cat((self.img_enc_stack, self.word_enc_stack), 1)
        self.convlstm_output = self.netImgLSTM(self.convlstm_input)

        self.img_predict = self.netImgDecoder(self.img_init.unsqueeze(0), self.convlstm_output.unsqueeze(0))

        self.img_init_save = self.img_init.data
        self.img_cur_save = img_input.data
        self.img_predict_save = self.img_predict.data


    # get image paths
    def get_image_paths(self):
        return self.image_paths


    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real.unsqueeze(0))
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D


    def backward_D(self):
        loss_D = self.backward_D_basic(self.netD, self.img_cur, self.img_predict)
        self.loss_D = loss_D.data[0]


    def backward_G_init(self):
        self.img_init = self.img_input
        self.word_init = self.word_input

        self.img_cur_enc = self.netImgEncoder(self.img_init.unsqueeze(0))
        self.word_cur_enc = self.Word2Tensor(self.word_init, self.feature_size, self.feature_size)

        self.word_enc_flag = True
        self.img_enc_stack = self.img_cur_enc
        self.word_enc_stack = 0  # Is it okay to use?

        # Refresh All saved data
        self.convlstm_input = 0
        self.convlstm_output = 0

        # Save (Just for the exception case)
        self.img_init_save = self.img_init.data
        self.img_cur_save = self.img_init.data
        self.img_predict_save = self.img_init.data
        self.loss_G = 0
        self.loss_idt = 0


    def backward_G(self):

        # Setting
        self.img_cur = self.img_input
        self.word_cur = self.word_input

        self.img_cur_enc = self.netImgEncoder(self.img_cur.unsqueeze(0))     # Temporalily Added to make a form (1 * 3 * 150 * 150) instead of (3 * 150 * 150)
        self.word_cur_enc = self.Word2Tensor(self.word_cur, self.feature_size, self.feature_size)

        #Input 
        # as (1 * 3 * 150 * 150) instead of (3 * 150 * 150)
        #Output
        # img_cur_enc => 1 * 256 * 38 * 38 (b * c * w * h)

        # Stack Before
        self.img_enc_stack
        if self.word_enc_flag == True:
            self.word_enc_stack = self.word_cur_enc
            self.word_enc_flag = False
        else:
            self.word_enc_stack = torch.cat((self.word_enc_stack, self.word_cur_enc), 0)

        # Lstm
        self.convlstm_input = torch.cat((self.img_enc_stack, self.word_enc_stack), 1)  # Stack Input
        self.convlstm_output = self.netImgLSTM(self.convlstm_input)

        # Stack After
        self.img_enc_stack = torch.cat((self.img_enc_stack, self.img_cur_enc), 0)
        self.word_enc_stack

        # Final
        self.img_predict = self.netImgDecoder(self.img_init.unsqueeze(0), self.convlstm_output.unsqueeze(0))

        print "Prediction Output"
        print self.img_predict

        # Loss Weight
        weight_idt = 1
        weight_G = 1

        self.loss_G = self.criterionGAN(self.netD(self.img_predict), True) * weight_G
        #self.loss_idt = self.criterionIdt(self.img_cur, self.img_predict) * weight_idt
        self.loss_idt = torch.mean(torch.abs(self.img_cur - self.img_predict)) * weight_idt # Not Sure About this...

        loss_total = self.loss_G + self.loss_idt
        loss_total.backward()

        # Save
        self.img_init_save = self.img_init.data
        self.img_cur_save = self.img_cur.data
        self.img_predict_save = self.img_predict.data
        self.loss_G = self.loss_G.data[0]
        self.loss_idt = self.loss_idt.data[0]

    def optimize_parameters(self, init_tensor = True):
        self.forward()

        if init_tensor == True:
            print "[First Frame Initialization] True"
            self.backward_G_init()
        else:
            # G
            print "[First Frame Initialization] False"
            self.optimizer_G.zero_grad()
            self.backward_G()
            self.optimizer_G.step()

            # D
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()


    def get_current_errors(self):
        ret_errors = OrderedDict([('D', self.loss_D), ('G', self.loss_G), ('Idt', self.loss_idt)])
        return ret_errors


    def get_current_visuals(self):
        img_init = util.tensor2im(self.img_init_save)
        img_cur = util.tensor2im(self.img_cur_save)
        img_predict = util.tensor2im(self.img_predict_save)

        ret_visuals = OrderedDict([('img_init', img_init), ('img_cur', img_cur), ('img_predict', img_predict)])
        return ret_visuals


    def save(self, label):
        self.save_network(self.netImgEncoder, 'ImgEncoder', label, self.gpu_ids)
        self.save_network(self.netImgLSTM, 'ImgLSTM', label, self.gpu_ids)
        # self.save_network(self.netWordEmbed, 'WordEmbed', label, self.gpu_ids)
        self.save_network(self.netImgDecoder, 'ImgDecoder', label, self.gpu_ids)
        self.save_network(self.netD, 'Discriminator', label, self.gpu_ids)

        np.save('grid_embedding.npy', self.dictionary)
