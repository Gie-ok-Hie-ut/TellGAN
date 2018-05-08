import torch
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


class TellGANModel(BaseModel):
    def name(self):
        return 'TellGANModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        # Just the basic format of function
        self.netImgEncoder = networks.define_ImgEncoder(opt.input_nc, opt.output_nc, opt.ngf, opt.which_model_netG,
                                                        opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.netImgLSTM = networks.define_ConvLSTM(opt.input_nc, opt.output_nc, opt.ngf, opt.which_model_netG, opt.norm,
                                                   not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.netWordEmbed = networks.define_WordEmbed(opt.input_nc, opt.output_nc, opt.ngf, opt.which_model_netG,
                                                      opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.netImgDecoder = networks.define_ImgDecoder(opt.input_nc, opt.output_nc, opt.ngf, opt.which_model_netG,
                                                        opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.which_model_netD, opt.n_layers_D, opt.norm,
                                          use_sigmoid, opt.init_type, self.gpu_ids)

        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netImgEncoder, 'ImgEncoder', which_epoch)
            self.load_network(self.netImgLSTM, 'ImgLSTM', which_epoch)
            self.load_network(self.netWordEmbed, 'WordEmbed', which_epoch)
            self.load_network(self.netImgDecoder, 'ImgDecoder', which_epoch)

            if self.isTrain:
                self.load_network(self.netD, 'Discriminator', which_epoch)

        if self.isTrain:
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)

            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionIdt = torch.nn.L1Loss()  # L1 Loss Okay?

            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(
                itertools.chain(self.netImgEncoder.parameters(), self.netImgLSTM.parameters(),
                                self.netWordEmbed.parameters(), self.netImgDecoder.parameters()), lr=opt.lr,
                betas=(opt.beta1, 0.999))
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
        networks.print_network(self.netWordEmbed)
        networks.print_network(self.netImgDecoder)
        if self.isTrain:
            networks.print_network(self.netD)
        print('-----------------------------------------------')

    def set_input(self, input):

        # We may change
        input_initial = input['Initial']
        input_frame = input['Frame']
        input_frame_next = input['Frame_Next']
        input_transcription_next = input['Transcription']

        if len(self.gpu_ids) > 0:
            input_frame = input_frame.cuda(self.gpu_ids[0], async=True)

        self.input_initial = input_initial
        self.input_frame = input_frame
        self.input_frame_next = input_frame_next
        self.input_transcription_next = input_transcription_next

    def forward(self):
        # self.real_A = Variable(self.input_A)
        self.img_init = Variable(self.input_initial)
        self.img_cur = Variable(self.input_frame)
        self.img_next = Variable(self.input_frame_next)
        self.word_next = Variable(self.input_transcription_next)

    def test(self, Init_Net):
        ##### 'test' doesn't need input_frame. (initial_frame and input_transcription only)

        ## Input
        self.img_init = Variable(self.input_initial, volatile=True)
        self.word_next = Variable(self.input_transcription_next, volatile=True)

        ## Generator

        # Encode Tensor
        if Init_Net == True:
            self.img_cur_enc = self.netImgEncoder(img_init)
        else:
            self.img_cur_enc = self.netImgEncoder(img_next_fake)

        self.word_next_enc = self.netWordEmbed(self.word_next)

        # ImgWord = [img, word]
        self.tensor_imgword_cur = torch.cat((self.img_cur_enc, self.word_next_enc), 1)

        # Previous = [ImgWord1, ImgWord2 ... , ImgWord_cur]
        if Init_Net == True:
            self.tensor_previous = tensor_imgword_cur
        else:
            self.tensor_previous = torch.cat((self.tensor_previous, self.tensor_imgword_cur), 1)

        # Feed to LSTM
        self.tensor_lstm_enc = self.netImgLSTM(self.tensor_previous)

        # Feed to Decoder
        img_next_fake = self.netImgDecoder(self.img_init, self.tensor_lstm_enc)

        self.img_next_fake = img_next_fake.data

    # Original Code
    # real_A = Variable(self.input_A, volatile=True)
    # fake_B = self.netG_A(real_A)
    # self.rec_A = self.netG_B(fake_B).data
    # self.fake_B = fake_B.data

    # real_B = Variable(self.input_B, volatile=True)
    # fake_A = self.netG_B(real_B)
    # self.rec_B = self.netG_A(fake_A).data
    # self.fake_A = fake_A.data

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
        loss_D.backward()
        return loss_D

    def backward_D(self):
        #######?
        img_next_fake = self.img_next_fake
        loss_D = self.backward_D_basic(self.netD, self.img_next, img_next_fake)
        self.loss_D = loss_D.data[0]

        # fake_B = self.fake_B_pool.query(self.fake_B)
        # loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)
        # self.loss_D_A = loss_D_A.data[0]

    def backward_G(self, Init_Net):

        # Tell GAN

        self.img_init
        self.img_cur
        self.img_next
        self.word_next
        self.tensor_previous
        self.img_next_fake

        ### Generator ###

        # Encode Tensor
        if Init_Net == True:
            self.img_cur_enc = self.netImgEncoder(self.img_init)
        else:
            self.img_cur_enc = self.netImgEncoder(self.img_cur)
        self.word_next_enc = self.netWordEmbed(self.word_next)

        # ImgWord = [img, word]
        self.tensor_imgword_cur = torch.cat((self.img_cur_enc, self.word_next_enc), 1)

        # Previous = [ImgWord1, ImgWord2 ... , ImgWord_cur]
        if Init_Net == True:
            self.tensor_previous = self.tensor_imgword_cur
        else:
            # We may delete old tensor by changin this form
            self.tensor_previous = torch.cat((self.tensor_previous, self.tensor_imgword_cur), 1)

        # Feed to LSTM
        self.tensor_lstm_enc = self.netImgLSTM(self.tensor_previous)

        # Feed to Decoder
        self.img_next_fake = self.netImgDecoder(self.img_init, self.tensor_lstm_enc)

        ### Loss ###

        # Loss Weight
        weight_idt = 1
        weight_G = 1

        # GAN Loss
        self.loss_G = self.criterionGAN(self.netD(self.img_next_fake), True) * weight_G

        # Identity Loss
        self.loss_idt = self.criterionIdt(self.img_next, self.img_next_fake) * weight_idt

        loss_total = self.loss_G + loss_idt
        loss_total.backward()

        self.img_next_fake = img_next_fake.data
        self.loss_G = loss_G.data[0]
        self.loss_idt = loss_idt.data[0]


def optimize_parameters(self, Init_Net):
    # forward
    self.forward()
    # G
    self.optimizer_G.zero_grad()
    self.backward_G(Init_Net)
    self.optimizer_G.step()
    # D
    self.optimizer_D.zero_grad()
    self.backward_D()
    self.optimizer_D.step()


def get_current_errors(self):
    ret_errors = OrderedDict([('D', self.loss_D), ('G', self.loss_G), ('Idt', self.loss_idt)])
    return ret_errors


def get_current_visuals(self):
    img_init = util.tensor2im(self.img_init)  # GT
    img_cur = util.tensor2im(self.img_cur)  # GT
    img_next = util.tensor2im(self.img_next)  # GT
    img_next_fake = util.tensor2im(self.img_next_fake)  # Generated

    ret_visuals = OrderedDict([('real_init', img_init), ('real_cur', img_cur), ('real_next', img_next),
                               ('fake_next', img_next)])
    return ret_visuals


def save(self, label):
    self.save_network(self.netImgEncoder, 'ImgEncoder', label, self.gpu_ids)
    self.save_network(self.netImgLSTM, 'ImgLSTM', label, self.gpu_ids)
    self.save_network(self.netWordEmbed, 'WordEmbed', label, self.gpu_ids)
    self.save_network(self.netImgDecoder, 'ImgDecoder', label, self.gpu_ids)
    self.save_network(self.netD, 'Discriminator', label, self.gpu_ids)
