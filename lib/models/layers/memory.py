
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import trunc_normal_
from lib.models.layers.cbam import CBAM
def softmax_w_top(x, top):
    values, indices = torch.topk(x, k=top, dim=1)
    x_exp = values.exp_()

    x_exp /= torch.sum(x_exp, dim=1, keepdim=True)
    x.zero_().scatter_(1, indices, x_exp.type(x.dtype)) # B * THW * HW

    return x

class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        '''
            Args:
                x (torch.Tensor): (B, L, C), input tensor
            Returns:
                torch.Tensor: (B, L, C), output tensor
        '''
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class mask_convblock(nn.Module):
    def __init__(self, in_dim,out_dim,expand):
        super().__init__()
        self.mask_conv = nn.Sequential(nn.Conv2d(in_dim+1,in_dim*expand,kernel_size=3,stride=1,padding=1),
                                    nn.BatchNorm2d(in_dim * expand),
                                    nn.LeakyReLU(),
                                    nn.Conv2d(in_dim*expand,out_dim,kernel_size=1,stride=1,padding=0))
        self.res = nn.Conv2d(in_dim+1,out_dim,1,1,0)

    def forward(self,x):
        x = self.res(x) + self.mask_conv(x)
        return x
    
class MemoryEecoder(nn.Module):
    def __init__(self,in_dim,out_dim,expand=2,pool_size=8):
        super(MemoryEecoder,self).__init__()
        # d_feat avage pool for depth-wise correlation operation
        self.conv_0 = mask_convblock(in_dim,in_dim // 2,expand)
        self.conv_1 = mask_convblock(in_dim // 2,in_dim // 2,expand)
        self.conv_2 = mask_convblock(in_dim // 2 ,out_dim,expand)
        self.cbam = CBAM(gate_channels=out_dim,pool_types=['lse','avg','max'])
        self.adp_pool = nn.AdaptiveMaxPool2d(pool_size)
        self.pool_size = pool_size
    def forward(self,x,mask,state='infer'):
        x = self.conv_0(torch.cat([x,mask],dim=1))  # b c h w
        if state != 'initial':
            x = self.adp_pool(x)
            mask = F.interpolate(mask,self.pool_size)
        x = self.conv_1(torch.cat([x,mask],dim=1))
        x = self.conv_2(torch.cat([x,mask],dim=1))
        x = self.cbam(x)
        return x
    
class MemoryDecoder(nn.Module):
    def __init__(self, memory_max_num,dim,expand,num_heads=8,pool_size=8):
        super().__init__()
        self.max_num = memory_max_num
        hidden_dim = dim * expand
        self.conv_0 = nn.Sequential(nn.Conv3d(dim,hidden_dim,kernel_size=(memory_max_num,3,3),padding=(0,1,1)),
                                    nn.Flatten(1,2),
                                    nn.BatchNorm2d(hidden_dim),
                                    nn.LeakyReLU())
        self.conv_1 = nn.Sequential(nn.Conv3d(dim,hidden_dim,kernel_size=(memory_max_num,5,5),padding=(0,2,2)),
                                    nn.Flatten(1,2),
                                    nn.BatchNorm2d(hidden_dim),
                                    nn.LeakyReLU())
        self.conv_res = nn.Sequential(nn.Conv3d(dim,dim,kernel_size=(memory_max_num,1,1)),
                                    nn.Flatten(1,2),
                                    nn.BatchNorm2d(dim),
                                    nn.LeakyReLU())
        self.mlp = Mlp(hidden_dim,hidden_features=hidden_dim*2,out_features=hidden_dim)
        self.uni_linear = nn.Linear(hidden_dim,dim)
        self.attn = nn.MultiheadAttention(hidden_dim,num_heads,batch_first=True)
        self.norm0 = nn.LayerNorm(hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mem_pos = nn.Parameter(torch.zeros(1,pool_size*pool_size,hidden_dim))
        trunc_normal_(self.mem_pos,std=0.02)
        self.cbam = CBAM(gate_channels=dim,pool_types=['lse','avg','max'])
        # self.adp_pool = nn.AdaptiveMaxPool2d(7)

    def forward(self,memory):
        memory = self.fill_memory(memory)
        B,C,N,H,W = memory.shape
        q = self.norm0(rearrange(self.conv_0(memory),'b c h w -> b (h w) c'))  
        kv = self.norm1(rearrange(self.conv_1(memory),'b c h w -> b (h w) c'))
        res = rearrange(self.conv_res(memory),'b c h w -> b (h w) c')
        q = q + self.attn(q+self.mem_pos,kv+self.mem_pos,kv,attn_mask=None,key_padding_mask=None)[0]
        q = q + self.mlp(self.norm2(q))
        memory = res + self.uni_linear(q)
        memory = rearrange(memory,'b (h w) c -> b c h w',h=H,w=W)
        kernel = self.cbam(memory)
        # kernel = self.adp_pool(memory)
        return kernel
    
    def fill_memory(self,memory):
        """## utilize the spitial information and reduce the computation source for decoder

        ### Args:
            - `memory (tensor)`: n b c h w (h:3, w:3)
        """
        memory_len = len(memory)
        if memory_len < self.max_num:
            lack_num = self.max_num - memory_len
            repeat_memory = memory[0].clone().repeat(lack_num,1,1,1,1)
            memory = torch.cat([repeat_memory,memory],dim=0)
        memory = rearrange(memory,'n b c h w -> b c n h w')
        return memory
    
class MemoryFusion(nn.Module):
    def __init__(self,in_dim,down_dim,num_heads,pool_size):
        super().__init__()
        hidden_dim = in_dim // 3
        self.search_conv_0 = nn.Conv2d(in_dim,in_dim,kernel_size=3,stride=1,padding=1,groups=in_dim)
        self.memory_conv_0 = nn.Conv2d(down_dim,down_dim*2,kernel_size=1)
        self.attn = nn.MultiheadAttention(hidden_dim,num_heads,batch_first=True)
        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_kv = nn.LayerNorm(hidden_dim)
        self.kernel_conv = nn.Sequential(nn.Conv2d(hidden_dim,hidden_dim,kernel_size=3,padding=1,stride=1,groups=hidden_dim),
                                      nn.BatchNorm2d(hidden_dim),
                                      nn.LeakyReLU()) 
        self.identity = nn.Conv2d(hidden_dim,hidden_dim,kernel_size=3,padding=1,stride=1,groups=hidden_dim)
        self.pos_down = nn.Linear(in_dim,hidden_dim)
        self.mem_pos = nn.Parameter(torch.zeros(1,pool_size*pool_size,hidden_dim))
        trunc_normal_(self.mem_pos,std=0.02)
        self.out = nn.Sequential(nn.Conv2d(in_dim,in_dim,kernel_size=1),
                                 nn.BatchNorm2d(in_dim))
    def forward(self,memory,x_feat,x_pos):
        """## _summary_

        ### Args:
            - `memory (_type_)`: b c h w c:256 h:12
            - `search (_type_)`: b c h w c:768 h:24
            - `search_pos (_type_)`: b l c
        ### Raises:
            - `NotImplementedError`: _description_

        ### Returns:
            - `_type_`: _description_
        """
        _,_,H,W = x_feat.shape
        x1,x2,x3 = self.search_conv_0(x_feat).chunk(3,dim=1)
        mem1,mem2 = self.memory_conv_0(memory).chunk(2,dim=1)
        # attention
        x_pos = self.pos_down(x_pos)
        x1 = rearrange(x1,'b c h w -> b (h w) c')
        q = self.norm_q(x1) + x_pos
        kv = self.norm_kv(rearrange(mem1,'b c h w -> b (h w) c'))
        x1 = x1 + self.attn(q,kv+self.mem_pos,kv,attn_mask=None,key_padding_mask=None)[0]
        x1 = rearrange(x1,'b (h w) c -> b c h w',h=H,w=W)
        # kernel
        x2 = F.interpolate(xcorr_depthwise(x2,mem2),(H,W),mode='bilinear')
        x2 = self.kernel_conv(x2)
        # identity
        x3 = self.identity(x3)
        return self.out(torch.cat([x1,x2,x3],dim=1))
    
class MemoryNetwork(nn.Module):
    def __init__(self,memory_num,memory_max_num,in_dim,out_dim,expand,num_heads,tz,stride):
        super().__init__()
        pool_size = tz // stride
        self.memory_encoder = MemoryEecoder(in_dim,out_dim,expand,pool_size)
        self.memory_decoder = MemoryDecoder(memory_max_num,out_dim,num_heads=num_heads,pool_size=pool_size,expand=expand)
        self.memory_fusion = MemoryFusion(in_dim,out_dim,num_heads,pool_size)
        # self.hw = tz // stride
        # lz = self.hw * self.hw
        # self.center = [lz//2 -1,lz//2+1]
        self.memory_num = memory_num
        self.memory  = None # n,b,c,h,w

    def initialize(self,z_feat,mask,x_feat,x_pos):
        """## initialize the memory with template features

        ### Args:
            - `memory (tensor)`: template features
        """
        bs = z_feat.shape[0]
        memory = self.memory_encoder(z_feat,mask,state='initial')
        self.memory = memory.unsqueeze(0)    # n,b c h w
        score = torch.zeros(bs,self.memory_num).to(memory) 
        score[:,0] = 1
        self.score = score
        memory = self.memory_decoder(self.memory)
        x_feat = self.memory_fusion(memory,x_feat,x_pos)
        return x_feat
    
    def addMemory(self,score_ve,in_memorys:torch.Tensor,state='infer',update_list=None,replace_index=None):
        """## add memory to memory block

        ### Args:
            - `score_ve (tensor)`: b,lz,lx   the similarity of between the x features and the center of z features
            - 'in_memory (tensor)': b c h w
        """
        if state == 'train':
            if len(self.memory) < self.memory_num:
                self.memory = torch.cat([self.memory,in_memorys.unsqueeze(0)],dim=0)

            else:
                for bs in range(self.memory.shape[1]):
                    if update_list[bs]:
                        index = replace_index[bs]
                        self.memory[index:index+1,bs] = in_memorys[bs:bs+1]
        else:
            self.update = False
            score = score_ve.max(-1)[0] # b
            # score = 0
            min_score = torch.min(self.score,dim=-1)[0]   # b
            in_memory = in_memorys
            index = sorted(torch.where(self.score == min_score)[1])[0]
            if len(self.memory) < self.memory_num:
                memory = in_memory.unsqueeze(0) # n b c h w
                self.memory = torch.cat([self.memory,memory],dim=0)
                self.score[:,index] = score
                self.update = True
            # no add new memory
            # else:
            #     self.update = False
            # adaptively add new memory
            elif score >= min_score:
                self.memory[index] = in_memory
                self.score[:,index] = score
                self.update = True
            
    
    def decode_memory(self,x_feat,x_pos,memory=None):
        """## _summary_

        ### Args:
            - `z_f (tensor)`: b c h w
            - `d_f (tensor)`: b c h w
            - `z_pos (tensor)`: position embeding of template features 
            - `d_pos (tensor)`: position embeding of dynamic template features 
            - `kernel (tensor)`: kernel of memory template features b c 3 3 

        ### Returns:
            - `tensor`: new template features with b c h w
        """
        if memory is not None:
            memory = self.memory_decoder(memory)
        else:
            memory = self.memory_decoder(self.memory)
        x_feat = self.memory_fusion(memory,x_feat,x_pos)
        return x_feat
    
    def encode_memory(self,feat,mask,state=None):
        x = self.memory_encoder(feat,mask,state)
        return x
    
    def forward(self, mode, *args, **kwargs):
        if mode == 'encode':
            return self.encode_memory(*args, **kwargs)
        elif mode == 'decode':
            return self.decode_memory(*args, **kwargs)
        elif mode == 'initialize':
            return self.initialize(*args, **kwargs)
        elif mode == 'addMemory':
            return self.addMemory(*args, **kwargs)
        else:
            raise NotImplementedError

def build_memory_network(cfg,backbone_dim):
    memory_num = cfg.MODEL.MEMORYBLOCK.MEMORY_NUM
    memory_max_num = cfg.MODEL.MEMORYBLOCK.MEMORY_MAX_NUM
    expand = cfg.MODEL.MEMORYBLOCK.EXPAND
    num_heads = cfg.MODEL.MEMORYBLOCK.NUM_HEADS
    in_dim = backbone_dim
    out_dim = cfg.MODEL.MEMORYBLOCK.OUT_DIM
    tz = cfg.DATA.TEMPLATE.SIZE
    stride = cfg.MODEL.BACKBONE.STRIDE 
    return MemoryNetwork(memory_num,memory_max_num,in_dim,out_dim,expand,num_heads,tz,stride)


def xcorr_depthwise(x, kernel):
    """depthwise cross correlation
        x (b,c,h,w,)
        kernel(b,c,h,w,)
    """
    batch = kernel.size(0)
    channel = kernel.size(1)
    # x = x.view(1, batch*channel, x.size(2), x.size(3))
    x = rearrange(x,'b c h w -> (b c) h w').unsqueeze(0)
    # kernel = kernel.view(batch*channel, 1, kernel.size(2), kernel.size(3))
    kernel = rearrange(kernel,'b c h w -> (b c) h w').unsqueeze(1)
    out = F.conv2d(x, kernel, groups=batch*channel)
    out = out.view(batch, channel, out.size(2), out.size(3))
    # out = rearrange(out,'n (b c) h w -> (n b) c h w',n=1,b=batch)
    return out