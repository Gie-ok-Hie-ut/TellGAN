import torch
from collections import OrderedDict
from torch.autograd import Variable
from torchvision import transforms
import itertools
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import numpy as np


class TellGANModel(BaseModel):
    def name(self):
        return 'TellGANModel'

    def initialize(self, opt, landmarkSuite=None):

        BaseModel.initialize(self, opt)

        #utilities
        self.landmarkSuite = landmarkSuite
        self.toTensor = transforms.ToTensor()

        ###### Basic Parameters ######
        self.feature_size = 20
        self.lstm_in_dim = (self.feature_size, self.feature_size)
        self.lstm_in_nc = 257
        #self.lstm_out_nc = [257, 256]
        #self.lstm_nlayers = 2
        self.lstm_out_nc = [256]
        self.lstm_nlayers = 3 # too shallow?
        self.lstm_kernel_size = (3, 3)
        hidden_layers=3

        ###### Network setting ######
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'WordUnet',opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.netPredictor = networks.NextFeaturesForWord(input_size=(self.feature_size*2),
                                                         hidden_size=self.feature_size*2,
                                                         num_layers=self.lstm_nlayers)
        #self.netImgEncoder = networks.define_ImgEncoder(opt.input_nc, opt.output_nc, opt.ngf, 'resnet_3blocks_enc',opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        #self.netImgLSTM = networks.define_ConvLSTM(self.lstm_in_dim, self.lstm_in_nc, self.lstm_nlayers, self.lstm_out_nc, self.lstm_kernel_size, self.gpu_ids)
        #self.netImgDecoder = networks.define_ImgDecoder(opt.input_nc, opt.output_nc, opt.ngf, 'resnet_3blocks_dec',opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)

        self.netPredictor.cuda()#self.gpu_ids)
        self.netG.cuda()

        # Load Dictionary
        self.dic_size = 30
        try:
            self.dictionary = np.load('grid_embedding.npy').item()
            print "[Dictionary] Loading Existing Embedding Dictionary"
        except IOError as e:
            self.dictionary = {'default': 0}
            print "[Dictionary] Building New Word Embedding Dictionary"


        if self.isTrain:
            use_sigmoid = opt.no_lsgan


            #self.netD_lstm = networks.define_D(opt.output_nc, opt.ndf, 'lstm_dis', opt.n_layers_D, opt.norm,use_sigmoid, opt.init_type, self.gpu_ids)
            self.netD_lstm = networks.SimpleLSTMDiscriminator(input_size=(self.feature_size*2), hidden_size=self.feature_size*2, num_layers=hidden_layers)
            self.netD_pair = networks.define_D(4, opt.ndf, 'basic', opt.n_layers_D, opt.norm,use_sigmoid, opt.init_type, self.gpu_ids)

            self.netD_lstm.cuda()
            self.netD_pair.cuda()
            #self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.which_model_netD, opt.n_layers_D, opt.norm,use_sigmoid, opt.init_type, self.gpu_ids)

            # 256 + 256 + 1 = 513 channels on input
            #dspeak_input_nc = 513
            #self.netD_speak = networks.define_D(dspeak_input_nc, opt.ndf, opt.which_model_netD, opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)

        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG, 'WordUnet', which_epoch)
            self.load_network(self.netPredictor, 'Predictor', which_epoch)
            #self.load_network(self.netImgEncoder, 'ImgEncoder', which_epoch)
            #self.load_network(self.netImgLSTM, 'ImgLSTM', which_epoch)
            #self.load_network(self.netImgDecoder, 'ImgDecoder', which_epoch)
            # self.load_network(self.netWordEmbed, 'WordEmbed', which_epoch)

            if self.isTrain: 
                self.load_network(self.netD_lstm, 'Discriminator_lstm', which_epoch)
                self.load_network(self.netD_pair, 'Discriminator_pair', which_epoch)
                #self.load_network(self.netD_speak, 'DiscriminatorSpeak', which_epoch)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionGAN.cuda()
            self.criterionIdt = torch.nn.MSELoss()  # L1 Loss Okay?
            self.criterionGAN.cuda()

            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(
                itertools.chain(self.netG.parameters(), self.netPredictor.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
                #itertools.chain(self.netG.parameters(), self.netImgEncoder.parameters(), self.netImgLSTM.parameters(), self.netImgDecoder.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
                #itertools.chain(self.netImgEncoder.parameters(), self.netImgLSTM.parameters(),self.netImgDecoder.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_lstm = torch.optim.Adam(self.netD_lstm.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_pair = torch.optim.Adam(self.netD_pair.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            #self.optimizer_D_speak = torch.optim.Adam(self.netD_speak.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D_lstm)
            self.optimizers.append(self.optimizer_D_pair)

            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))
        else:
             self.criterionTest = torch.nn.MSELoss()

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        networks.print_network(self.netPredictor)
        #networks.print_network(self.netImgEncoder)
        #networks.print_network(self.netImgLSTM)
        #networks.print_network(self.netImgDecoder)
        # networks.print_network(self.netWordEmbed)
        if self.isTrain:
            networks.print_network(self.netD_lstm)
            networks.print_network(self.netD_pair)
            #networks.print_network(self.netD_speak)
        print('-----------------------------------------------')

    def set_input(self, input):
        (input_frame, input_transcription, input_landmark) = input

        if len(self.gpu_ids) > 0:
            input_frame = input_frame.cuda(self.gpu_ids[0], async=True)

        self.input_frame = input_frame
        self.input_transcription = input_transcription
        self.input_landmark = input_landmark

    def landmarksToTensor(self, landmarks, size=(256,256)):
        #Flatten and normilize landmarks
        feat0_norm = landmarks.copy()
        feat0_norm[:, :, 1] /= size[0]
        feat0_norm[:, :, 0] /= size[1]

        # Flatten 2D (x,y) coords to 1-D [xyxyxy]
        featTB4 = Variable(torch.from_numpy(feat0_norm))
        featT = featTB4.view(featTB4.numel())

        return featT

    def forward(self):
        #nLmks =  self.input_landmark.shape[0]
        self.img_input = Variable(self.input_frame)

        self.word_tensor = self.Word2Tensor(self.input_transcription, self.feature_size*2)
        self.word_input = Variable(self.word_tensor)

        # format landmarks
        featT = self.landmarksToTensor(landmarks=self.input_landmark,
                                            size=(self.input_frame.size(1), self.input_frame.size(2)))

        self.lnmk_input = featT

    def Word2Tensor(self, word, dim=1):
        word_cur = word

        # Update unseen word
        if self.dictionary.get(word_cur, -1) == -1:
            self.dictionary.update({word_cur: float((len(self.dictionary) + 1)) / self.dic_size})

        # Make Tensor
        vec2np = np.full((1, dim), self.dictionary[word_cur])
        np2tensor = torch.from_numpy(vec2np).float()

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

            self.word_tensor = self.Word2Tensor(self.input_transcription, dim=self.feature_size*2)
            self.word_input = Variable(self.word_tensor, volatile=True)

            if init_tensor == True:
                '''
                ' When tensor is initialized, we just reset the encoder sequence for the LSTM
                ' to the init frame and return the given frame, as no prediction is needed
                '''
                self.img_init = self.img_input
                self.img_cur = self.img_input
                self.word_init = self.word_input
                self.lnmk_cur = self.lnmk_input

                if self.lstm_stack is not None and (self.lstm_stack[0] - self.word_init != 0):
                    self.lstm_stack = torch.cat((self.word_init, self.lstm_stack[-1].unsqueeze(0)), 0)
                else:
                    self.lstm_stack = torch.cat((self.word_init, self.lnmk_cur.unsqueeze(0)), 0)




                self.img_enc_stack = self.img_cur_enc
                self.word_stack = 0  # Is it okay to use?


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
                if self.word_flag == True:
                    self.word_stack = self.word_cur_enc
                    self.word_flag = False
                else:
                    self.word_stack = torch.cat((self.word_stack, self.word_cur_enc), 0)

                self.convlstm_input = torch.cat((self.img_enc_stack, self.word_stack), 1)  # Stack Input
                self.convlstm_output = self.netImgLSTM(self.convlstm_input)

                # Stack predicted image encoding After
                self.img_enc_stack = torch.cat((self.img_enc_stack, self.convlstm_output.unsqueeze(0)), 0)
                # self.word_stack


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

    def backward_D_word(self):
        real = self.dis_word_real
        fake = self.dis_word_fake

        loss_D_word = self.backward_D_basic(self.netD_lstm, real, fake)
        self.loss_D_word = loss_D_word.data[0]


    def backward_D_lnmk(self):
        real = self.dis_lnmk_real.cuda()
        fake = self.dis_lnmk_fake.cuda()

        loss_D_lnmk = self.backward_D_basic(self.netD_lstm, real, fake)
        self.loss_D_lnmk = loss_D_lnmk.data[0]

    def backward_D_pair(self):
        real = self.dis_pair_real.cuda()
        fake = self.dis_pair_fake.cuda()

        loss_D_pair = self.backward_D_basic(self.netD_pair, real, fake)
        self.loss_D_pair = loss_D_pair.data[0]


    def backward_G_init(self):
        self.img_init = self.img_input
        self.word_init = self.word_input
        self.lnmk_init = self.lnmk_input

        #self.img_cur_enc = self.netImgEncoder(self.img_init.unsqueeze(0))
        self.lnmk_cur = self.lnmk_init
        self.word_cur = self.word_init

        self.lstm_stack = torch.cat((self.word_cur, self.lnmk_cur.unsqueeze(0)),0)

        self.word_flag = True
        self.word_stack = 0  # Is it okay to use?

        # Save (Just for the exception case)
        self.img_init_save = self.img_init.data
        self.img_cur_save = self.img_init.data
        self.img_predict_save = self.img_init.data
        self.lnmk_cur_save = self.lnmk_init.data
        self.lnmk_predict_save = self.lnmk_init.data
        self.loss_G_lnmk = 0
        self.loss_G_pair = 0
        self.loss_img_idt = 0
        self.loss_lnmk_idt = 0

    def backward_G_finetune(self): # Fine-tune to the initialization Frame
        self.lnmk_cur_imgT, self.lnmk_cur_img = self.landmarkToImg(self.lnmk_cur, size=(self.img_init.size(1), self.img_init.size(2)))
        self.img_predict = self.netG(self.img_init.unsqueeze(0).cuda(), self.lnmk_cur_imgT.unsqueeze(0).cuda()) # Train focus on Face Generator

        self.loss_img_idt = self.criterionIdt(self.img_predict, self.img_init.unsqueeze(0)) * 1
        self.loss_img_idt.backward(retain_graph=True)


        self.img_predict_save = self.img_predict.data
        self.loss_img_idt = self.loss_img_idt.data[0]


    def landmarkToImg(self, lnmk, size=(128,128)):
        if lnmk.size(-1) < 40:
            print lnmk.size
        lmkT2d = lnmk.clone().view(self.feature_size,1,2)

        # denormalize
        lmkT2d[:,:,1] *= size[0]
        lmkT2d[:,:,0] *= size[1]
        np_lmk = lmkT2d.data.cpu().numpy()
        lnmk_img = self.landmarkSuite.create_mask(size, np_lmk)
        pil_lnmk_img = self.landmarkSuite.matToPil(lnmk_img)

        pil_lnmk_imgT = self.toTensor(pil_lnmk_img)

        return pil_lnmk_imgT, pil_lnmk_img


    def backward_G(self):
        # Setting
        self.img_cur = self.img_input
        self.word_cur = self.word_input
        self.lnmk_cur = self.lnmk_input

        # Prediction
        lstm_input = self.lstm_stack.unsqueeze(1).cuda()
        self.lnmk_predict = self.netPredictor(lstm_input.detach())

        # Final
        self.lnmk_cur_imgT, self.lnmk_cur_img = self.landmarkToImg(self.lnmk_cur, size=(self.img_cur.size(1), self.img_cur.size(2)))
        self.lnmk_predict_imgT, self.lnmk_predict_img = self.landmarkToImg(self.lnmk_predict, size=(self.img_cur.size(1), self.img_cur.size(2)))
        self.img_predict = self.netG(self.img_init.unsqueeze(0).cuda(), self.lnmk_cur_imgT.unsqueeze(0).cuda()) # Train focus on Face Generator
        #self.img_predict = self.netG(self.img_init.unsqueeze(0), self.lnmk_predict.unsqueeze(0))

        # Stack After
        self.lstm_stack2 = torch.cat((self.lstm_stack, self.lnmk_predict.cpu()),0)
        self.lstm_stack = torch.cat((self.lstm_stack, self.lnmk_cur.unsqueeze(0)), 0)





        # For Discriminator
        # 1. Discriminator_lnmk - fake landmark added
        self.dis_lnmk_real = self.lstm_stack.unsqueeze(1)
        self.dis_lnmk_fake = self.lstm_stack2.unsqueeze(1)

        # 2. Discriminator_word - fake word added
        #self.dis_word_real = torch.cat((self.lnmk_stack, self.word_stack2_fake), 1) # Should be changed
        #self.dis_word_fake = torch.cat((self.lnmk_stack, self.word_stack), 1)

        # 3. Discriminator_img - fake img added
        self.dis_pair_real = torch.cat((self.lnmk_cur_imgT.unsqueeze(0).cuda(), self.img_cur.unsqueeze(0)), 1)
        self.dis_pair_fake = torch.cat((self.lnmk_cur_imgT.unsqueeze(0).cuda(), self.img_predict), 1)

        # Loss Weight
        weight_G_word = 1
        weight_G_lnmk = 1
        weight_G_img = 1
        weight_img_idt = 1
        weight_lnmk_idt = 2

        # Loss Calculate
        #self.fake_dspeak_enc = torch.cat((self.lnmk_predict.unsqueeze(0), self.netImgEncoder(self.img_predict), self.word_cur_enc), 1)
        #self.fake_dspeak_enc = torch.cat((self.lnmk_predict.unsqueeze(0), self.img_predict), 1)
        
        #self.loss_G_word = self.criterionGAN(self.netD_lstm(self.dis_word_fake), True) * weight_G_word
        self.loss_G_lnmk = self.criterionGAN(self.netD_lstm(self.dis_lnmk_fake.cuda()), True) * weight_G_lnmk
        self.loss_G_pair = self.criterionGAN(self.netD_pair(self.dis_pair_fake.cuda()), True) * weight_G_img

        #self.loss_G = self.criterionGAN(self.netD(self.img_predict), True) * weight_G
        #self.loss_G_speak = self.criterionGAN(self.netD_speak(self.fake_dspeak_enc), True) * weight_G
        #self.loss_idt = self.mse_loss(self.img_cur, self.img_predict.squeeze(0)) * weight_idt
        self.loss_img_idt = self.criterionIdt(self.img_predict, self.img_cur.unsqueeze(0)) * weight_img_idt
        self.loss_lnmk_idt = self.criterionIdt(self.lnmk_predict, self.lnmk_cur.cuda().unsqueeze(0)) * weight_lnmk_idt

        #loss_total = self.loss_G_word + self.loss_G_lnmk + self.loss_G_pair + self.loss_img_idt + self.loss_lnmk_idt
        loss_total = self.loss_G_lnmk + self.loss_G_pair + self.loss_img_idt + self.loss_lnmk_idt
        loss_total.backward(retain_graph=True)






        # Save
        self.img_init_save = self.img_init.data
        self.img_cur_save = self.img_cur.data
        self.img_predict_save = self.img_predict.data
        self.lnmk_cur_save = self.lnmk_cur_img
        self.lnmk_predict_save = self.lnmk_predict_img

        self.loss_G_lnmk = self.loss_G_lnmk.data[0]
        self.loss_G_pair = self.loss_G_pair.data[0]
        self.loss_img_idt = self.loss_img_idt.data[0]
        self.loss_lnmk_idt = self.loss_lnmk_idt.data[0]
        #self.loss_G_speak = self.loss_G_speak.data[0]
        #self.loss_idt = self.loss_idt.data[0]

    def optimize_parameters(self, init_tensor = True):
        self.forward()

        if init_tensor == True:
            #print("[First Frame Initialization] {0} [Word] {1}".format(init_tensor, self.input_transcription))
            self.backward_G_init()

            for lap in range(0,20):
                self.optimizer_G.zero_grad()
                self.backward_G_finetune()
                self.optimizer_G.step()
        else:
            # G
            self.optimizer_G.zero_grad()
            self.backward_G()
            self.optimizer_G.step() 

            # D_word
            #self.optimizer_D_lstm.zero_grad()
            #self.backward_D_word
            #self.optimizer_D_lstm.step()

            # D_lnmk
            self.optimizer_D_lstm.zero_grad()
            self.backward_D_lnmk()
            self.optimizer_D_lstm.step()

            # D_img
            self.optimizer_D_pair.zero_grad()
            self.backward_D_pair()
            self.optimizer_D_pair.step()

            # D
            #self.optimizer_D.zero_grad()
            #self.backward_D()
            #self.optimizer_D.step()

            # D_speak
            #self.optimizer_D_speak.zero_grad()
            #self.backward_D_speak()
            #self.optimizer_D_speak.step()



    def get_current_errors(self):
        if self.isTrain:
            ''' self.loss_G_lnmk = self.loss_G_lnmk.data[0]
        self.loss_G_pair = self.loss_G_pair.data[0]
        self.loss_img_idt = self.loss_img_idt.data[0]
        self.loss_lnmk_idt = self.loss_lnmk_idt.data[0]'''
            ret_errors = OrderedDict([
                ('D_lnmk', self.loss_D_lnmk),
                ('D_pair', self.loss_D_pair),
                ('G_lnmk', self.loss_G_lnmk),
                ('G_pair', self.loss_G_pair),
                ('img_idt', self.loss_img_idt),
                ('lnmk_idt', self.loss_lnmk_idt)
            ])
        else:
            ret_errors = OrderedDict([('L', self.loss_test),('MSE', self.loss_test)])
        return ret_errors


    def get_current_visuals(self):
        img_init = util.tensor2im(self.img_init_save.unsqueeze(0))
        img_cur = util.tensor2im(self.img_cur_save.unsqueeze(0))
        img_predict = util.tensor2im(self.img_predict_save)
        lnmk_cur = self.landmarkSuite.pilToMat(self.lnmk_cur_save)
        lnmk_predict = self.landmarkSuite.pilToMat(self.lnmk_predict_save)

        ret_visuals = OrderedDict([('img_init', img_init), ('img_cur', img_cur), ('img_predict', img_predict), ('lnmk_cur', lnmk_cur), ('lnmk_predict', lnmk_predict)])
        return ret_visuals


    def save(self, label):
        self.save_network(self.netG, 'WordUnet', label, self.gpu_ids)
        self.save_network(self.netPredictor, 'Predictor', label, self.gpu_ids)
        #self.save_network(self.netImgEncoder, 'ImgEncoder', label, self.gpu_ids)
        #self.save_network(self.netImgLSTM, 'ImgLSTM', label, self.gpu_ids)
        #self.save_network(self.netImgDecoder, 'ImgDecoder', label, self.gpu_ids)
        # self.save_network(self.netWordEmbed, 'WordEmbed', label, self.gpu_ids)

        self.save_network(self.netD_lstm, 'Discriminator_lstm', label, self.gpu_ids)
        self.save_network(self.netD_pair, 'Discriminator_pair', label, self.gpu_ids)

        np.save('grid_embedding.npy', self.dictionary)

    def mse_loss(self, input, target):
        return torch.sum((input - target)**2) / input.data.nelement()

    def get_dic(self):
        return self.dictionary

    def get_dic_size(self):
        return self.dic_size