import torch.nn as nn

class RegionDecoderBlock(nn.Module):
    def __init__(self,dim):
        super(RegionDecoderBlock, self).__init__()

        self.down_block1 = nn.Sequential(  
            nn.Conv2d(int(dim),int(dim/2),3,1,1),
            nn.BatchNorm2d(int(dim/2)),
            nn.ReLU6())

        self.down_block2 = nn.Sequential(
            nn.Conv2d(int(dim/2),int(dim/4),3,1,1),
            nn.BatchNorm2d(int(dim/4)),
            nn.ReLU6())

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(int(dim/4),int(dim/8),3,1,1),
            nn.BatchNorm2d(int(dim/8)),
            nn.ReLU6())

        self.out = nn.Conv2d(int(dim/8),2,3,1,1) # 32 → 2

    def forward(self,corr_map):
        """q is x_feat, kv is x corrmap

        Args:
            corr_map (tensor): (b,c,h,w)
        """
        corr_map = self.down_block1(corr_map)      # (b,128,32,32) (b,64,32,32)
    
        corr_map = self.down_block2(corr_map) 
    
        corr_map = self.conv_block3(corr_map) 
        corr_map = self.out(corr_map)
        return corr_map

def build_seek_target(dim):
    return RegionDecoderBlock(dim)