import torch
from torch import nn
from torch.nn import functional as F

class NonLocalBlock3D(nn.Module):
    def __init__(self, in_channels, dimension=3, sub_sample=True):
        super(NonLocalBlock3D, self).__init__()

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels

        self.inter_channels = in_channels // 2
        if self.inter_channels == 0:
            self.inter_channels = 1

        max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
       
        #============================================================
        #make self.g , self.theta, self.phi
        #these are nn.Conv3d, 1x1x1, stride=1, padding=0
        #============================================================
        self.g = nn.Conv3d(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)
        
        self.theta = nn.Conv3d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        
        self.phi = nn.Conv3d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)
        #============================================================

        #============================================================
        #make self.W
        #in this part, self.W.weight and self.W.bias must initialize to 0
        #============================================================
        self.W = nn.Conv3d(in_channels=self.inter_channels, out_channels=self.in_channels,
                         kernel_size=1, stride=1, padding=0)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)
        #============================================================

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''
        print(x.shape)
        batch_size = x.size(0)
        
        #============================================================
        #1. use self.g(x)
        #2. use self.theta(x)
        #3. use self.phi(x)
        #4. several matrix multiplication between previous return value
        #5. use self.W(y)
        #6. make z with x and self.W(y)
        #============================================================
        print(self.g(x).shape)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)
        print(g_x.shape)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        print(theta_x.shape)
        
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        print(phi_x.shape)

        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)
        print(f.shape)


        y = torch.matmul(f_div_C, g_x)
        print(y.shape)

        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x
        #============================================================

        return z


if __name__ == '__main__':
    from torch.autograd import Variable
    import torch

    sub_sample = False

    img = Variable(torch.randn(2, 3, 10, 20, 20))
    net = NonLocalBlock3D(3, sub_sample=sub_sample)
    out = net(img)
    print(out.size())