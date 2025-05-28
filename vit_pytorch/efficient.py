import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from linformer import Linformer
class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, transformer, pool = 'cls', channels = 3):
        super().__init__()
        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = transformer

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        img = img.squeeze()
        x = self.to_patch_embedding(img)
        # print(x.shape)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        # print(x.shape)
        x += self.pos_embedding[:, :(n + 1)]

        x = self.transformer(x)
        print(x.shape)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)

if __name__ == '__main__':
    img_size = 15
    x = torch.rand(2, 1, 200, img_size, img_size)
    efficient_transformer = Linformer(
        dim=1024,
        seq_len=25 + 1,  # 7x7 patches + 1 cls-token
        depth=12,
        heads=8,
        k=64
    )
    model = ViT(
        dim=1024,
        image_size=15,
        patch_size=3,
        num_classes=16,
        transformer=efficient_transformer,
        channels=200,
    )

    from thop import profile
    flops, params = profile(model, (x,))

    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')
    repetitions = 100
    total_time = 0
    optimal_batch_size = 2
    with torch.no_grad():
        for rep in range(repetitions):
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()
            _ = model(x)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender) / 1000
            total_time += curr_time
    Throughput = (repetitions * optimal_batch_size) / total_time
    print("FinalThroughput:", Throughput)
    print("The training time is: **********", total_time)
