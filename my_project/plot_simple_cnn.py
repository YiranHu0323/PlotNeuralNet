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
        to_input('input.jpg', width=13, height=13),
        
        # Encoder
        to_Conv("conv1", 224, 64, offset="(0,0,0)", to="(0,0,0)", height=64, depth=64, width=2, caption="Conv1 + BN"),
        to_Pool("pool1", offset="(0,0,0)", to="(conv1-east)", height=32, depth=32, width=1, opacity=0.5, caption="MaxPool"),
        
        to_Conv("conv2", 112, 128, offset="(2,0,0)", to="(pool1-east)", height=32, depth=32, width=2, caption="Conv2 + BN"),
        to_Pool("pool2", offset="(0,0,0)", to="(conv2-east)", height=16, depth=16, width=1, opacity=0.5, caption="MaxPool"),
        
        to_Conv("conv3", 56, 128, offset="(2,0,0)", to="(pool2-east)", height=16, depth=16, width=2, caption="Conv3 + BN"),
        to_Pool("pool3", offset="(0,0,0)", to="(conv3-east)", height=8, depth=8, width=1, opacity=0.5, caption="MaxPool"),
        
        # Fully Connected Layers
        to_SoftMax("fc1", 512, offset="(2,0,0)", to="(pool3-east)", width=1, height=1, depth=40, caption="FC1"),
        to_SoftMax("fc2", 4096, offset="(2,0,0)", to="(fc1-east)", width=1, height=1, depth=40, caption="FC2"),

        # Reshape
        to_Conv("reshape", 64, 8, offset="(2,0,0)", to="(fc2-east)", width=1, height=8, depth=8, caption="Reshape"),

        # Decoder
        to_UnPool("unpool1", offset="(2,0,0)", to="(reshape-east)", width=1, height=16, depth=16, opacity=0.5),
        to_Conv("uconv1", 16, 32, offset="(0,0,0)", to="(unpool1-east)", height=16, depth=16, width=2, caption="Upsample + Uconv1"),
        
        to_UnPool("unpool2", offset="(2,0,0)", to="(uconv1-east)", width=1, height=32, depth=32, opacity=0.5),
        to_Conv("uconv2", 32, 64, offset="(0,0,0)", to="(unpool2-east)", height=32, depth=32, width=2, caption="Upsample + Uconv2"),
        
        to_UnPool("unpool3", offset="(2,0,0)", to="(uconv2-east)", width=1, height=48, depth=48, opacity=0.5),
        to_Conv("uconv3", 64, 128, offset="(0,0,0)", to="(unpool3-east)", height=48, depth=48, width=2, caption="Upsample + Uconv3"),
        
        to_UnPool("unpool4", offset="(2,0,0)", to="(uconv3-east)", width=1, height=56, depth=56, opacity=0.5),
        to_Conv("uconv4", 128, 128, offset="(0,0,0)", to="(unpool4-east)", height=56, depth=56, width=2, caption="Upsample + Uconv4"),
        
        to_Conv("uconv6", 128, 4, offset="(2,0,0)", to="(uconv4-east)", height=56, depth=56, width=2, caption="Uconv6"),

        # Output
        # to_Conv("output", 128, 4, offset="(2,0,0)", to="(uconv6-east)", height=56, depth=56, width=2, caption="Output"),

        to_end()
    ]

    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex')

if __name__ == '__main__':
    plot_simple_cnn()