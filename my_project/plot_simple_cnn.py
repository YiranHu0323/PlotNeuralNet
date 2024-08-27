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
        to_input('input.jpg', to='(0,0,0)', width=8, height=8, name="input"),
        to_ConvConvRelu(name='ccr_b1', s_filer=64, n_filer=(64,64), offset="(0,0,0)", to="(0,0,0)", width=(2,2), height=40, depth=40, caption="Convolutional Layers"),
        to_Pool(name="pool_b1", offset="(0,0,0)", to="(ccr_b1-east)", width=1, height=32, depth=32, opacity=0.5, caption="Max Pooling"),

        # Encoder
        to_Conv("conv2", 112, 128, offset="(2,0,0)", to="(pool_b1-east)", height=32, depth=32, width=2, caption="Conv 128"),
        to_Pool("pool2", offset="(0,0,0)", to="(conv2-east)", height=28, depth=28, width=1, opacity=0.5, caption="Max Pooling"),
        to_Conv("conv3", 56, 128, offset="(2,0,0)", to="(pool2-east)", height=25, depth=25, width=2, caption="Conv 128"),
        to_Pool("pool3", offset="(0,0,0)", to="(conv3-east)", height=21, depth=21, width=1, opacity=0.5, caption="Max Pooling"),
        
        # Fully Connected Layers
        to_ConvSoftMax("fc1", s_filer=512, offset="(2,0,0)", to="(pool3-east)", width=1, height=25, depth=25, caption="FC 512"),
        to_ConvSoftMax("fc2", s_filer=4096, offset="(2,0,0)", to="(fc1-east)", width=1, height=25, depth=25, caption="FC 4096"),

        # Decoder
        to_UnPool("unpool1", offset="(2,0,0)", to="(fc2-east)", width=1, height=25, depth=25, opacity=0.5, caption="Upsampling"),
        to_Conv("uconv1", 16, 32, offset="(0,0,0)", to="(unpool1-east)", height=25, depth=25, width=2, caption="Conv 32"),
        to_UnPool("unpool2", offset="(2,0,0)", to="(uconv1-east)", width=1, height=32, depth=32, opacity=0.5, caption="Upsampling"),
        to_Conv("uconv2", 32, 64, offset="(0,0,0)", to="(unpool2-east)", height=32, depth=32, width=2, caption="Conv 64"),
        to_UnPool("unpool3", offset="(2,0,0)", to="(uconv2-east)", width=1, height=40, depth=40, opacity=0.5, caption="Upsampling"),
        to_Conv("uconv3", 64, 128, offset="(0,0,0)", to="(unpool3-east)", height=40, depth=40, width=2, caption="Conv 128"),
        to_UnPool("unpool4", offset="(2,0,0)", to="(uconv3-east)", width=1, height=64, depth=64, opacity=0.5, caption="Upsampling"),
        to_Conv("uconv4", 128, 128, offset="(0,0,0)", to="(unpool4-east)", height=64, depth=64, width=2, caption="Conv 128"),
        to_Conv("uconv6", 128, 4, offset="(2,0,0)", to="(uconv4-east)", height=64, depth=64, width=2, caption="Conv 4"),

        # Output
        to_ConvSoftMax("output", 4, "(2,0,0)", "(uconv6-east)", caption="Output"),

        to_end()
    ]

    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex')

if __name__ == '__main__':
    plot_simple_cnn()