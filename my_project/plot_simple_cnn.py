import sys
sys.path.append('../')
from pycore.tikzeng import *
from pycore.blocks import *

def plot_simple_cnn():
    arch = [
        to_head('..'),
        to_cor(),
        to_begin(),

        # Input
        to_input('input.png'),

        # Encoder
        to_Conv("conv1", 224, 64, offset="(0,0,0)", to="(0,0,0)", height=64, depth=64, width=2),
        to_Pool("pool1", offset="(0,0,0)", to="(conv1-east)"),
        to_Conv("conv2", 112, 128, offset="(1,0,0)", to="(pool1-east)", height=32, depth=32, width=2),
        to_Pool("pool2", offset="(0,0,0)", to="(conv2-east)"),
        to_Conv("conv3", 56, 128, offset="(1,0,0)", to="(pool2-east)", height=25, depth=25, width=2),
        to_Pool("pool3", offset="(0,0,0)", to="(conv3-east)"),
        
        # Fully Connected Layers
        to_ConvSoftMax("fc1", s_filer=512, offset="(2,0,0)", to="(pool3-east)", width=1, height=25, depth=25, caption="FC 512"),
        to_ConvSoftMax("fc2", s_filer=4096, offset="(2,0,0)", to="(fc1-east)", width=1, height=25, depth=25, caption="FC 4096"),

        # Decoder
        to_UnPool("unpool1", offset="(1,0,0)", to="(fc2-east)"),
        to_Conv("uconv1", 16, 32, offset="(0,0,0)", to="(unpool1-east)", height=16, depth=16, width=2),
        to_UnPool("unpool2", offset="(1,0,0)", to="(uconv1-east)"),
        to_Conv("uconv2", 32, 64, offset="(0,0,0)", to="(unpool2-east)", height=32, depth=32, width=2),
        to_UnPool("unpool3", offset="(1,0,0)", to="(uconv2-east)"),
        to_Conv("uconv3", 64, 128, offset="(0,0,0)", to="(unpool3-east)", height=64, depth=64, width=2),
        to_UnPool("unpool4", offset="(1,0,0)", to="(uconv3-east)"),
        to_Conv("uconv4", 128, 128, offset="(0,0,0)", to="(unpool4-east)", height=128, depth=128, width=2),
        to_Conv("uconv6", 128, 4, offset="(1,0,0)", to="(uconv4-east)", height=128, depth=128, width=2),

        # Output
        to_ConvSoftMax("output", 4, "(1,0,0)", "(uconv6-east)"),

        to_end()
    ]

    namefile = 'SimpleCNN_architecture'
    to_generate(arch, namefile + '.tex')

if __name__ == '__main__':
    plot_simple_cnn()