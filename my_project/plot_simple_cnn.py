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
        to_input('input.jpg', caption="Input"),
        
        # Encoder
        to_Conv("conv1", 224, 64, offset="(0,0,0)", to="(0,0,0)", height=64, depth=64, width=2, caption="Conv1 3x3 + BN"),
        to_Pool("pool", offset="(0,0,0)", to="(conv1-east)", height=32, depth=32, width=1, opacity=0.5, caption="MaxPool"),
        
        to_Conv("conv2", 112, 128, offset="(2,0,0)", to="(pool-east)", height=32, depth=32, width=2, caption="Conv2 3x3 + BN"),
        
        to_Conv("conv3", 112, 128, offset="(2,0,0)", to="(conv2-east)", height=32, depth=32, width=2, caption="Conv3 3x3 + BN"),
        
        # Fully Connected Layers
        to_ConvSoftMax("fc1", s_filer=512, offset="(2,0,0)", to="(conv3-east)", width=1, height=40, depth=40, caption="FC 512"),
        to_ConvSoftMax("fc2", s_filer=4096, offset="(2,0,0)", to="(fc1-east)", width=1, height=40, depth=40, caption="FC 4096"),

        # Decoder
        to_UnPool("unpool1", offset="(2,0,0)", to="(fc2-east)", width=1, height=32, depth=32, opacity=0.5, caption="Upsample"),
        to_Conv("uconv1", 32, 32, offset="(0,0,0)", to="(unpool1-east)", height=32, depth=32, width=2, caption="Conv 3x3"),
        
        to_UnPool("unpool2", offset="(2,0,0)", to="(uconv1-east)", width=1, height=40, depth=40, opacity=0.5, caption="Upsample"),
        to_Conv("uconv2", 64, 64, offset="(0,0,0)", to="(unpool2-east)", height=40, depth=40, width=2, caption="Conv 3x3"),
        
        to_UnPool("unpool3", offset="(2,0,0)", to="(uconv2-east)", width=1, height=48, depth=48, opacity=0.5, caption="Upsample"),
        to_Conv("uconv3", 128, 128, offset="(0,0,0)", to="(unpool3-east)", height=48, depth=48, width=2, caption="Conv 3x3"),
        
        to_UnPool("unpool4", offset="(2,0,0)", to="(uconv3-east)", width=1, height=56, depth=56, opacity=0.5, caption="Upsample"),
        to_Conv("uconv4", 256, 128, offset="(0,0,0)", to="(unpool4-east)", height=56, depth=56, width=2, caption="Conv 3x3"),
        
        to_Conv("uconv6", 256, 4, offset="(2,0,0)", to="(uconv4-east)", height=56, depth=56, width=2, caption="Conv 3x3"),

        # Output
        to_ConvSoftMax("output", 4, "(2,0,0)", "(uconv6-east)", caption="Output"),

        to_end()
    ]

    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex')

if __name__ == '__main__':
    plot_simple_cnn()