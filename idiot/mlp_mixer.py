import torch
import torch.nn as nn

from torch.nn.functional import gelu as gelu


class MLP(nn.Module):
    def __init__(self, num_hidden_dim, dropout, mlp_dim_factor):
        """
        Single MLP, as defined in the paper, contains two fully-connected layers
        and a GELU nonlinearity.
        
        num_hidden_dim * mlp_dim_factor: hidden mlp width in token mixing/channel mixing.
        """
        super().__init__()
        self.fc1 = nn.Linear(num_hidden_dim, num_hidden_dim * mlp_dim_factor)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(num_hidden_dim * mlp_dim_factor, num_hidden_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout1(gelu(self.fc1(x)))
        x = self.dropout2(self.fc2(x))
        return x


class TokenMixer(nn.Module):
    def __init__(self, num_hidden_dim, num_patches, dropout, mlp_dim_factor):
        super().__init__()
        self.layer_norm = nn.LayerNorm(num_hidden_dim)
        self.mlp = MLP(num_patches, dropout, mlp_dim_factor)

    def forward(self, x):
        residual = x # (B, S, D)
        x = self.layer_norm(x)
        x = x.transpose(1, 2) # (B, D, S)
        x = self.mlp(x) # (B, D, S)
        x = x.transpose(1, 2) # (B, S, D)
        out = x + residual
        return out


class ChannelMixer(nn.Module):
    def __init__(self, num_hidden_dim, num_patches, dropout, mlp_dim_factor):
        super().__init__()
        self.layer_norm = nn.LayerNorm(num_hidden_dim)
        self.mlp = MLP(num_hidden_dim, dropout, mlp_dim_factor)

    def forward(self, x):
        residual = x # (B, S, D) 
        x = self.layer_norm(x)
        x = self.mlp(x) # (B, S, D)
        out = x + residual
        return out


class MixerLayer(nn.Module):
    def __init__(self, num_hidden_dim, num_patches, dropout, mlp_dim_factor):
        super().__init__()
        self.token_mixer = TokenMixer(num_hidden_dim, num_patches, dropout, mlp_dim_factor)
        self.channel_mixer = ChannelMixer(num_hidden_dim, num_patches, dropout, mlp_dim_factor)

    def forward(self, x):
        x = self.token_mixer(x) # (B, S, D)
        x = self.channel_mixer(x) # (B, S, D)
        return x
    

class MLPMixer(nn.Module):
    def __init__(self, image_shape, patch_size, num_channels, num_hidden_dim, 
                 num_layers, num_classes, dropout, mlp_dim_factor):
        super().__init__()
        
        # check if a HxH image can be split to S PxP patches
        quotient, modulo = divmod(image_shape[0] * image_shape[1], patch_size**2)
        if modulo:
            raise ValueError("Cannot divide the image into (patch_size, patch_size) patches.")
        else: 
            self.num_patches = quotient
            print(f"Divided image of dimensions {image_shape} into {self.num_patches} {patch_size}x{patch_size} patches")
        
        # Strided conv2 equals: patching the image + embedding every patch to a vector.
        self.patch_embedd = nn.Conv2d(num_channels, num_hidden_dim, kernel_size=patch_size, stride=patch_size)
        
        self.mixer_layers = nn.Sequential(
            *[MixerLayer(num_hidden_dim, self.num_patches, dropout, mlp_dim_factor) for _ in range(num_layers)])
        self.logit_generator = nn.Linear(num_hidden_dim, num_classes)
        
    def forward(self, x):
        patches = self.patch_embedd(x)
        B, D, _, _ = patches.shape
        patches = patches.permute(0, 2, 3, 1)
        patches = patches.view(B, -1, D) # (B, S, D)
        assert patches.shape[1] == self.num_patches
        output = self.mixer_layers(patches) # (B, S, D)
        output = output.mean(dim = 1) # (B, D)
        
        return self.logit_generator(output)
        
