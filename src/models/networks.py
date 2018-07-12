from __future__ import division
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
from torch.optim import lr_scheduler
import torchvision

from .convlstm import ConvLSTM
###############################################################################
# Functions
###############################################################################


def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


# Define Network
def define_ImgEncoder(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False, init_type='normal', gpu_ids=[]):
    netImgEncoder = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())

    if which_model_netG == 'resnet_3blocks_enc':
        netImgEncoder = ResnetGenerator_Encoder(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=3, gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('ImgEncoder model name [%s] is not recognized' % which_model_netG)

    if len(gpu_ids) > 0:
        netImgEncoder.cuda(gpu_ids[0])

    init_weights(netImgEncoder, init_type=init_type)

    return netImgEncoder

# Define Network
def define_ImgDecoder(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False, init_type='normal', gpu_ids=[]):
    netImgDecoder = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())

    which_model_netG = 'resnet_3blocks_dec'

    if which_model_netG == 'resnet_3blocks_dec':
        netImgDecoder = ResnetGenerator_Decoder(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=3, gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('ImgEncoder model name [%s] is not recognized' % which_model_netG)

    if len(gpu_ids) > 0:
        netImgDecoder.cuda(gpu_ids[0])

    init_weights(netImgDecoder, init_type=init_type)

    return netImgDecoder

class NextFrameConvLSTM(ConvLSTM):
    def forward(self, input, hidden_state=None):

        # ConvLSTM takes (b, t, c, h, w), encoder out/decoder in uses (t, h, w, c)
        # Using pytorch to Permute dimensions  (b, t, h, w, c) -> (t, c, h, w)

        # Add batch dim with unsqueeze and Rearrange dim
        seq_input = input.unsqueeze(0)#.permute(0, 1, 4, 2, 3)

        # Feet into LSTM
        output, hidden_state = super(NextFrameConvLSTM, self).forward(seq_input, hidden_state)

        # Permute dimensions back to (b, t, c, h, w) -> (b, t, h, w, c)
        # and remove batch dim -> (t, h, w, c)
        #convlstm_output = output.permute(0, 1, 3, 4, 2).squeeze(0)
        convlstm_output = output.squeeze(0)


        predicted_change = convlstm_output[-1]

        # I don't think we need weights (hidden_state)
        return predicted_change #, hidden_state

def define_ConvLSTM(input_size,input_dim,num_layers,hidden_dim,kernel_size,gpu_ids=[]):

    convLSTM = None
    use_gpu = len(gpu_ids) > 0

    if use_gpu:
        assert (torch.cuda.is_available())

    convLSTM = NextFrameConvLSTM(input_size=input_size,input_dim=input_dim,
                            num_layers=num_layers,hidden_dim=hidden_dim,
                            kernel_size=kernel_size, batch_first=True)

    if len(gpu_ids) > 0:
        convLSTM.cuda(gpu_ids[0])

    # Hidden State initialized if not given
    # hidden_state = netImgLSTM.get_init_states()

    return convLSTM




def define_G(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False, init_type='normal', gpu_ids=[]):
    netG = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())

    if which_model_netG == 'resnet_9blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, gpu_ids=gpu_ids)
    elif which_model_netG == 'resnet_6blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6, gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_128':
        netG = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_256':
        netG = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    elif which_model_netG == 'LandmarkUnet':
        netG = LandmarkUnetGenerator(input_nc, output_nc)
    elif which_model_netG == 'LandmarkUnet2':
        netG = UnetGenerator(input_nc+1, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    if len(gpu_ids) > 0:
        netG.cuda(gpu_ids[0])
    init_weights(netG, init_type=init_type)
    return netG


def define_D(input_nc, ndf, which_model_netD,n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', gpu_ids=[]):
    netD = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())
    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'pixel':
        netD = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'lstm_dis':

        L_in_size = (128,128) # ?
        L_dim_in = 257
        L_n_layers = 1
        L_dim_out = 256
        L_kernel_size = (3,3)
        D_ndf = ndf
        D_n_layers = n_layers_D
        D_norm_layer = norm_layer
        D_use_sigmoid = use_sigmoid
        gpu_ids = gpu_ids

        netD = LSTMDiscriminator(L_in_size, L_dim_in, L_n_layers, L_dim_out, L_kernel_size, D_ndf=64, D_n_layers=3, D_norm_layer=nn.BatchNorm2d, D_use_sigmoid=False, gpu_ids=[])
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD)
    if use_gpu:
        netD.cuda(gpu_ids[0])

    if not which_model_netD == 'lstm_dis': # I'm not sure
        init_weights(netD, init_type=init_type)
    return netD

# On Network.py
def define_F(gpu_ids, use_bn=False):
    tensor = torch.cuda.FloatTensor if gpu_ids else torch.FloatTensor
    # pytorch pretrained VGG19-54, before ReLU.
    if use_bn:
        feature_layer = 49
    else:
        feature_layer = 34
    netF = VGGFeatureExtractor(feature_layer=feature_layer, use_bn=use_bn, use_input_norm=True, tensor=tensor)
    if gpu_ids:
        netF = nn.DataParallel(netF).cuda()
    netF.eval()  # No need to train
    return netF

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


##############################################################################
# Classes
##############################################################################


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)

class ResnetGenerator_Decoder_Summation(torch.nn.Module):
    def __init__(self,input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=4, gpu_ids=[], padding_type='reflect'):
        super(ResnetGenerator_Decoder_Summation, self).__init__()
        #Size should be changed
        self.Decoder = ResnetGenerator_Decoder(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=3, gpu_ids=gpu_ids)

    def forward(self, input_init, input_output):
        output_transition = self.Decoder.forward(input_output)

        # Added batch 3 * 150 * 150 -> 1 * 3 * 150 * 150
        input_init_sq = input_init

        output_result = input_init_sq + output_transition
        #output_result = input_init_sq

        return output_result


#Currently 256*256*3 -> 64 * 64 * 256
class ResnetGenerator_Encoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=3, gpu_ids=[], padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator_Encoder, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc # Not Used
        self.ngf = ngf
        self.gpu_ids = gpu_ids

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]


        self.model_en = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model_en, input, self.gpu_ids)
        else:
            return self.model_en(input)

class ResnetGenerator_Decoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=4, gpu_ids=[], padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator_Decoder, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = []

        n_downsampling = 2

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]



        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)] #Why not ConvTranspose2d?

        #Delete this line for Affine Transformation
        #model += [nn.Tanh()]

        self.model_de = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model_de, input, self.gpu_ids)
        else:
            return self.model_de(input)


# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, gpu_ids=[], padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

# Based on this code
# https://github.com/milesial/Pytorch-UNet/blob/master/unet

class LandmarkUnetGenerator(nn.Module):
    def __init__(self,dim_in,dim_out):
        super(LandmarkUnetGenerator,self).__init__()

        self.same1 = conv_same(dim_in, 16) #3/256/256 -> 16/256/256
        self.down1 = conv_down(16, 32)
        self.down2 = conv_down(32, 64)
        self.down3 = conv_down(64, 128)
        #self.up1 = conv_up(160, 64, 32)
        self.up1 = conv_up(128,64,1,64)
        self.up2 = conv_up(64,32,1,32)
        self.up3 = conv_up(32,16,1,32)
        self.same2 = conv_same(32 + 3, dim_out)

        self.pool = self.pool_layer()

    def pool_layer(self):
        layer = [nn.MaxPool2d(2,stride=2)]
        return nn.Sequential(*layer)

    def forward(self, img, lm):
        # Preprocess
        lm1=self.pool(lm)  # 256/256/3 -> 128/128/3
        lm2=self.pool(lm1) # 128/128/3 -> 64/64/3
        #lm3=pool(lm2) # 64/64/3 -> 32/32/3

        # Encoder
        x1 = self.same1(img) # 256/256/3 -> 256/256/16
        x2 = self.down1(x1) # 256/256/16 -> 128/128/32
        x3 = self.down2(x2) # 128/128/32 -> 64/64/64
        x4 = self.down3(x3) # 64/64/64 -> 32/32/128
        #x5 = self.down4(x4) # 32/32/128 -> 16/16/256

        #x5 = torch.cat((x4, lm3), 0)  # 32/32/128, 32/32/3 -> 32/32/131

        # Decoder
        #x6 = self.up1(x5,x4,lm3) # 16/16/256, 32/32/128, 32/32/3 -> 32
        x5 = self.up1(x4,x3,lm2) # 32/32/128, 64/64/64, 64/64/3 -> 64/64/64
        x6 = self.up2(x5,x2,lm1) # 64/64/64, 128/128/32, 128/128/3 -> 128/128/32
        x7 = self.up3(x6,x1,lm) # 128/128/32, 256/256/16, 256/256/3 -> 256/256/32

        x8 =  torch.cat((x7, img),1)

        x9 = self.same2(x8)

        return x9


class conv_double(nn.Module): # same size W/H
    def __init__(self, in_ch, out_ch):
        super(conv_double, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class conv_same(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_same, self).__init__()
        self.conv = conv_double(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x

class conv_down(nn.Module): # size be half
    def __init__(self, in_ch, out_ch):
        super(conv_down, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3,stride=2, padding=1),
            #norm_layer(out_ch),
            nn.ReLU(True),
            conv_double(out_ch, out_ch)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class conv_up(nn.Module):
    def __init__(self, in_ch1, in_ch2, in_ch3, out_ch):
        super(conv_up, self).__init__()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv1 = conv_double(in_ch1 + in_ch2 + in_ch3, out_ch)
        self.conv2 = conv_double(out_ch, out_ch)
        #self.aug = nn.Sequential(nn.Conv2d(in_ch, aug_ch, kernel_size=1,stride=1, padding=0))

    def forward(self, x1, x2, x3):
        x1 = self.up(x1)
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(UnetGenerator, self).__init__()
        self.gpu_ids = gpu_ids

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)
# Defines the PatchGAN discriminator with the specified arguments.
class LSTMDiscriminator(nn.Module):
    def __init__(self,L_in_size, L_dim_in, L_n_layers, L_dim_out, L_kernel_size, D_ndf=64, D_n_layers=3, D_norm_layer=nn.BatchNorm2d, D_use_sigmoid=False, gpu_ids=[]):
        super(LSTMDiscriminator, self).__init__()

        self.ConvLSTM = define_ConvLSTM(L_in_size, L_dim_in, L_n_layers, L_dim_out, L_kernel_size, gpu_ids)
        self.NLayerDis =  NLayerDiscriminator(L_dim_out, D_ndf, D_n_layers, D_norm_layer, D_use_sigmoid, gpu_ids)

    def forward(self, input):
        x = self.ConvLSTM(input)
        x = self.NlayerDis(x)

        return x

# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[]):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[]):
        super(PixelDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        if use_sigmoid:
            self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.net, input, self.gpu_ids)
        else:
            return self.net(input)

class SimpleLSTMDiscriminator(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super(SimpleLSTMDiscriminator, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.input_seq = [
            nn.Linear(input_size,input_size)
        ]

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.in_layer = nn.Sequential(*self.input_seq)

        self.hidden = self.init_hidden()

        self.output_seq = [
            nn.Linear(hidden_size,hidden_size)
        ]

        self.out_layer = nn.Sequential(*self.output_seq)

        
    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(self.num_layers, 1, self.hidden_size).cuda(),
                torch.zeros(self.num_layers, 1, self.hidden_size).cuda())

    def forward(self, input):
            lstm_in = self.in_layer(input)
            pred_seq, self.hidden = self.lstm(lstm_in, self.init_hidden())
            out = self.out_layer(pred_seq[-1])
            return out


    '''
class NextFeaturesForWord(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=None, num_layers=5):
        super(NextFeaturesForWord, self).__init__()

        output_size =  output_size if output_size is not None else hidden_size

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        #self.input_seq = [
        #    nn.Linear(input_size, hidden_size),
        #    nn.Dropout(0.5),
        #    nn.Linear(hidden_size, hidden_size)
            #nn.ReLU(True),
            #nn.Sigmoid()
        #]

        self.output_seq = [
            nn.Linear(hidden_size, output_size)
            #nn.ReLU(True),
            #nn.Sigmoid()
        ]

        #self.in_layer = nn.Sequential(*self.input_seq)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.out_layer = nn.Sequential(*self.output_seq)

        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(self.num_layers, 1, self.hidden_size).cuda(),
                torch.zeros(self.num_layers, 1, self.hidden_size).cuda())

    def forward(self, input):
        pred_seq, self.hidden = self.lstm(input, self.init_hidden())
        pred_latent = pred_seq[-1]
        out = self.out_layer(pred_latent)
        return out

    '''
class NextFeaturesForWord(nn.Module):
    def __init__(self, input_size, hidden_size, dic_size, output_size=None, num_layers=5):
        super(NextFeaturesForWord, self).__init__()

        output_size =  output_size if output_size is not None else hidden_size

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.input_seq = [
            nn.Linear(input_size, hidden_size),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, hidden_size)
        ]

        self.output_seq = [
            nn.Linear(hidden_size,output_size)
        ]

        self.in_layer = nn.Sequential(*self.input_seq)
        self.lstm = nn.LSTM(hidden_size+dic_size, hidden_size, num_layers)
        self.out_layer = nn.Sequential(*self.output_seq)

        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (Variable(torch.zeros(self.num_layers, 1, self.hidden_size).cuda()),
                Variable(torch.zeros(self.num_layers, 1, self.hidden_size).cuda()))

    def forward(self, lmks, word):
        lmk_enc = None
        for seq in range(0,lmks.size(0)):
            seq_enc = self.in_layer(lmks[seq]).unsqueeze(0)
            if lmk_enc is None:
                lmk_enc = seq_enc
            else:
                lmk_enc = torch.cat((lmk_enc, seq_enc),0)

        # Concat word to encodings
        lstm_in = torch.cat((lmk_enc, word), 2)

        pred_seq, self.hidden = self.lstm(lstm_in, self.init_hidden())
        pred_latent = pred_seq[-1]
        out = self.out_layer(pred_latent)
        return out



class VGGFeatureExtractor(nn.Module):
    def __init__(self,
                 feature_layer=34,
                 use_bn=False,
                 use_input_norm=True,
                 tensor=torch.FloatTensor):
        super(VGGFeatureExtractor, self).__init__()
        if use_bn:
            model = torchvision.models.vgg19_bn(pretrained=True)
        else:
            model = torchvision.models.vgg19(pretrained=True)
        self.use_input_norm = use_input_norm
        if self.use_input_norm:
            mean = Variable(tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), requires_grad=False)
            # [0.485-1, 0.456-1, 0.406-1] if input in range [-1,1]
            std = Variable(tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), requires_grad=False)
            # [0.229*2, 0.224*2, 0.225*2] if input in range [-1,1]
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        self.features = nn.Sequential(*list(model.features.children())[:(feature_layer + 1)])
        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = self.features(x)
        return output