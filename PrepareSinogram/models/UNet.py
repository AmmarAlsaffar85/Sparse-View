import torch
import torch.nn as nn

class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.bn2(x)
        

        return x

class conv_block_down(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = nn.Conv2d(in_c, out_c, kernel_size=3,stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_c)

        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.relu(x)
        x = self.bn(x)
        


        return x
        
class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = conv_block(in_c, out_c)
        self.conv_down = conv_block_down(out_c, out_c)
        

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.conv_down(x)

        return x, p

class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=3, stride=2, padding=1,output_padding=1)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()
        self.conv = conv_block(out_c+out_c, out_c)

    def forward(self, inputs, skip):
        #print('input size=',inputs.size())
        x = self.up(inputs)
        x = self.relu(x)
        x =self.bn(x)
        
        #print('from decoder block x and skip size=',x.size(),skip.size())
        x = torch.cat([x, skip], axis=1)
       
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        """ Encoder """
        self.e1 = encoder_block(1, 64)
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)
        self.e4 = encoder_block(256, 512)

        """ Bottleneck """
        self.b = conv_block(512, 1024)

        """ Decoder """
        self.d1 = decoder_block(1024, 512)
        self.d2 = decoder_block(512, 256)
        self.d3 = decoder_block(256, 128)
        self.d4 = decoder_block(128, 64)

        """ Classifier """
        self.outputs = nn.Conv2d(64, 1, kernel_size=1, padding=0)

    def forward(self, inputs):
        """ Encoder """
        s1, p1 = self.e1(inputs)
        #print('s1 size=',s1.size(),p1.size())
        s2, p2 = self.e2(p1)
        #print('s2 size=',s2.size(),p2.size())
        s3, p3 = self.e3(p2)
        #print('s3 size=',s3.size(),p3.size())
        s4, p4 = self.e4(p3)
        #print('s3 size=',s4.size(),p4.size())
        """ Bottleneck """
        b = self.b(p4)
        #print('b size=',b.size())
        """ Decoder """
        d1 = self.d1(b, s4)
        #print('d1 size=',d1.size())
        d2 = self.d2(d1, s3)
        #print('d2 size=',d2.size())
        d3 = self.d3(d2, s2)
        #print('d3 size=',d3.size())
        #print('d3 and s1=',d3.size(),s1.size())
        d4 = self.d4(d3, s1)
        #print('d4 size=',d4.size())

        outputs = self.outputs(d4)

        return outputs

if __name__ == "__main__":
    x = torch.randn((2, 3, 512, 512))
    f = build_unet()
    y = f(x)
    print(y.shape)
