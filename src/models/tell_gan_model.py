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

        self.feature_size = 32
        self.lstm_in_dim = (self.feature_size, self.feature_size)
        self.lstm_in_nc = 257
        #self.lstm_out_nc = [257, 256]
        #self.lstm_nlayers = 2
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
        self.dic_size = 300
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

            # 256 + 256 + 1 = 513 channels on input
            dspeak_input_nc = 513
            self.netD_speak = networks.define_D(dspeak_input_nc, opt.ndf, opt.which_model_netD, opt.n_layers_D, opt.norm,
                                                use_sigmoid, opt.init_type, self.gpu_ids)

        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netImgEncoder, 'ImgEncoder', which_epoch)
            self.load_network(self.netImgLSTM, 'ImgLSTM', which_epoch)
            # self.load_network(self.netWordEmbed, 'WordEmbed', which_epoch)
            self.load_network(self.netImgDecoder, 'ImgDecoder', which_epoch)

            if self.isTrain:
                self.load_network(self.netD, 'Discriminator', which_epoch)
                self.load_network(self.netD_speak, 'DiscriminatorSpeak', which_epoch)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionIdt = torch.nn.MSELoss()  # L1 Loss Okay?

            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(
                itertools.chain(self.netImgEncoder.parameters(), self.netImgLSTM.parameters(),
                                self.netImgDecoder.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_speak = torch.optim.Adam(self.netD_speak.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_D_speak)

            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))
        else:
             self.criterionTest = torch.nn.MSELoss()

        print('---------- Networks initialized -------------')
        networks.print_network(self.netImgEncoder)
        networks.print_network(self.netImgLSTM)
        # networks.print_network(self.netWordEmbed)
        networks.print_network(self.netImgDecoder)
        if self.isTrain:
            networks.print_network(self.netD)
            networks.print_network(self.netD_speak)
        print('-----------------------------------------------')

    def set_input(self, input):
        (input_frame, input_transcription) = input

        if len(self.gpu_ids) > 0:
            input_frame = input_frame.cuda(self.gpu_ids[0], async=True)

        self.input_frame = input_frame
        self.input_transcription = input_transcription

    def forward(self):
        self.img_input = Variable(self.input_frame)

        self.word_tensor = self.Word2Tensor(self.input_transcription)
        self.word_input = Variable(self.word_tensor)

    def Word2Tensor(self, word):
        word_cur = word

        # Update unseen word
        if self.dictionary.get(word_cur, -1) == -1:
            self.dictionary.update({word_cur: float((len(self.dictionary) + 1)) / self.dic_size})

        # Make Tensor
        vec2np = np.full((1, 1), self.dictionary[word_cur])
        np2tensor = torch.from_numpy(vec2np).cuda().float()

        return np2tensor

    def ExpandTensor(self,tensor,width,height):
        tensor_cur = tensor

        np2tensor = tensor_cur.repeat(width,height)

        # Add One dim
        np2tensor_sq1 = np2tensor.unsqueeze(0)
        np2tensor_sq2 = np2tensor_sq1.unsqueeze(0)

        return np2tensor_sq2 # 1*1*38*38 (b*c*w*h)

    def test(self, init_tensor):

        with torch.no_grad():
            self.img_input = Variable(self.input_frame, volatile=True)

            self.word_tensor = self.Word2Tensor(self.input_transcription)
            self.word_input = Variable(self.word_tensor, volatile=True)

            if init_tensor == True:
                '''
                ' When tensor is initialized, we just reset the encoder sequence for the LSTM
                ' to the init frame and return the given frame, as no prediction is needed
                '''
                self.img_init = self.img_input
                self.img_cur = self.img_input
                self.word_init = self.word_input

                self.img_cur_enc = self.netImgEncoder(self.img_init.unsqueeze(0))
                self.word_cur_enc = self.ExpandTensor(self.word_init, self.feature_size, self.feature_size)

                self.word_enc_flag = True
                self.img_enc_stack = self.img_cur_enc
                self.word_enc_stack = 0  # Is it okay to use?

                # Refresh All saved data
                self.convlstm_input = 0
                self.convlstm_output = 0

                # Redundant
                self.img_predict = self.img_input.unsqueeze(0)
                self.img_predict_save = self.img_predict.data

            else:
                '''
                ' When Tensor has already been initialized, we must predict the next frame with the
                ' initialized tensors. First we stack the word encodings to tell the network the word 
                ' to be uttered predict the frame. Then concatinate the init image and word encondings
                ' are fed to the LSTM which predicts the next frame encoding. The predicted encoding
                ' is then stacked onto the image encoding stack for the next iteration. The predicted 
                ' encoding is also fed into the decoder to produce the acutual frame.
                '''
                # Predict Current Img
                self.img_cur = self.img_input
                self.word_cur = self.word_input

                # Temporalily Added to make a form (1 * 3 * 150 * 150) instead of (3 * 150 * 150)
                self.img_cur_enc = self.netImgEncoder(self.img_cur.unsqueeze(0))
                self.word_cur_enc = self.ExpandTensor(self.word_cur, self.feature_size, self.feature_size)

                # Stack Before
                #self.img_enc_stack
                if self.word_enc_flag == True:
                    self.word_enc_stack = self.word_cur_enc
                    self.word_enc_flag = False
                else:
                    self.word_enc_stack = torch.cat((self.word_enc_stack, self.word_cur_enc), 0)

                self.convlstm_input = torch.cat((self.img_enc_stack, self.word_enc_stack), 1)  # Stack Input
                self.convlstm_output = self.netImgLSTM(self.convlstm_input)

                # Stack predicted image encoding After
                self.img_enc_stack = torch.cat((self.img_enc_stack, self.convlstm_output.unsqueeze(0)), 0)
                # self.word_enc_stack


                self.img_predict = self.netImgDecoder(self.img_init.unsqueeze(0), self.convlstm_output.unsqueeze(0))

                self.img_init_save = self.img_init.data
                self.img_cur_save = self.img_input.data
                self.img_predict_save = self.img_predict.data

            self.loss_test = self.criterionTest(self.img_predict, self.img_cur.unsqueeze(0)).data[0]
            return util.tensor2im(self.img_predict_save)


    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)

        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward(retain_graph=True)
        return loss_D


    def backward_D(self):
        loss_D = self.backward_D_basic(self.netD, self.img_cur.unsqueeze(0), self.img_predict)
        self.loss_D = loss_D.data[0]

    def backward_D_speak(self):
        real = torch.cat((self.img_cur_enc, self.img_cur_enc, self.word_cur_enc), 1)
        # self.fake_dspeak_enc defined in backward_G and should be called before
        fake = self.fake_dspeak_enc
        loss_D_speak = self.backward_D_basic(self.netD_speak, real, fake)
        self.loss_D_speak = loss_D_speak.data[0]

    def backward_G_init(self):
        self.img_init = self.img_input
        self.word_init = self.word_input

        self.img_cur_enc = self.netImgEncoder(self.img_init.unsqueeze(0))
        self.word_cur_enc = self.ExpandTensor(self.word_init, self.feature_size, self.feature_size)

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
        self.loss_G_speak = 0
        self.loss_idt = 0


    def backward_G(self):

        # Setting
        self.img_cur = self.img_input
        self.word_cur = self.word_input

        self.img_cur_enc = self.netImgEncoder(self.img_cur.unsqueeze(0))     # Temporalily Added to make a form (1 * 3 * 150 * 150) instead of (3 * 150 * 150)
        self.word_cur_enc = self.ExpandTensor(self.word_cur, self.feature_size, self.feature_size)
        #self.word_cur_enc = self.word_cur

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

        self.convlstm_input = torch.cat((self.img_enc_stack, self.word_enc_stack), 1)  # Stack Input
        self.convlstm_output = self.netImgLSTM(self.convlstm_input)

        # Stack After
        self.img_enc_stack = torch.cat((self.img_enc_stack, self.img_cur_enc), 0)
        self.word_enc_stack

        # Final
        self.img_predict = self.netImgDecoder(self.img_init.unsqueeze(0), self.convlstm_output.unsqueeze(0))

        # Loss Weight
        weight_idt = 100
        weight_G = 1

        self.fake_dspeak_enc = torch.cat((self.convlstm_output.unsqueeze(0), self.netImgEncoder(self.img_predict), self.word_cur_enc), 1)
        self.loss_G = self.criterionGAN(self.netD(self.img_predict), True) * weight_G
        self.loss_G_speak = self.criterionGAN(self.netD_speak(self.fake_dspeak_enc), True) * weight_G
        #self.loss_idt = self.mse_loss(self.img_cur, self.img_predict.squeeze(0)) * weight_idt
        self.loss_idt = self.criterionIdt(self.img_predict, self.img_cur.unsqueeze(0)) * weight_idt
        #self.loss_idt = torch.mean(torch.abs(self.img_cur - self.img_predict)) * weight_idt # Not Sure About this...

        loss_total = self.loss_G + self.loss_G_speak + self.loss_idt
        loss_total.backward(retain_graph=True)

        # Save
        self.img_init_save = self.img_init.data
        self.img_cur_save = self.img_cur.data
        self.img_predict_save = self.img_predict.data
        self.loss_G = self.loss_G.data[0]
        self.loss_G_speak = self.loss_G_speak.data[0]
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

            # D_speak
            self.optimizer_D_speak.zero_grad()
            self.backward_D_speak()
            self.optimizer_D_speak.step()


    def get_current_errors(self):
        if self.isTrain:
            ret_errors = OrderedDict([
                ('D', self.loss_D),
                ('Ds', self.loss_D_speak),
                ('G', self.loss_G),
                ('Gs', self.loss_G_speak),
                ('Idt', self.loss_idt)
            ])
        else:
            ret_errors = OrderedDict([('L', self.loss_test),('MSE', self.loss_test)])
        return ret_errors


    def get_current_visuals(self):
        img_init = util.tensor2im(self.img_init_save.unsqueeze(0))
        img_cur = util.tensor2im(self.img_cur_save.unsqueeze(0))
        img_predict = util.tensor2im(self.img_predict_save)

        #ret_visuals = OrderedDict([('img_init', img_init), ('img_cur', img_cur), ('img_predict', img_predict)])
        ret_visuals = OrderedDict([('img_predict', img_predict)])
        return ret_visuals


    def save(self, label):
        self.save_network(self.netImgEncoder, 'ImgEncoder', label, self.gpu_ids)
        self.save_network(self.netImgLSTM, 'ImgLSTM', label, self.gpu_ids)
        # self.save_network(self.netWordEmbed, 'WordEmbed', label, self.gpu_ids)
        self.save_network(self.netImgDecoder, 'ImgDecoder', label, self.gpu_ids)
        self.save_network(self.netD, 'Discriminator', label, self.gpu_ids)
        self.save_network(self.netD_speak, 'DiscriminatorSpeak', label, self.gpu_ids)

        np.save('grid_embedding.npy', self.dictionary)

    def mse_loss(self, input, target):
        return torch.sum((input - target)**2) / input.data.nelement()