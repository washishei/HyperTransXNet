import math
import torch
from torch import nn
from torch.nn import functional as F
from .blocks import MaskedConv1D, Scale, LayerNorm, DeepInterpolator, MoEBlock
from .UMMA_backbones import ConvHRLRFullResSelfAttTransformerBackboneRevised
from .conv2d import TESTEtAl
from .UMMA_necks import FPN1D
import collections
import numpy as np


class PtTransformerClsHead(nn.Module):   # fpn_output is x level tensor,like[torch.Size([10, 768, 16],torch.Size([10, 384, 16])......)


    """
    1D Conv heads for classification
    """
    def __init__(
        self,
        input_dim,
        feat_dim,
        num_classes,
        prior_prob=0.01,
        num_layers=3,
        kernel_size=3,
        act_layer=nn.ReLU,
        with_ln=False,
        empty_cls = []
    ):
        super().__init__()
        self.act = act_layer()
        # build the head
        self.head = nn.ModuleList()
        self.norm = nn.ModuleList()
        for idx in range(num_layers-1):
            if idx == 0:
                in_dim = input_dim
                out_dim = feat_dim
            else:
                in_dim = feat_dim
                out_dim = feat_dim
            self.head.append(
                MaskedConv1D(
                    in_dim, out_dim, kernel_size,
                    stride=1,
                    padding=kernel_size//2,
                    bias=(not with_ln)
                )
            )
            if with_ln:
                self.norm.append(LayerNorm(out_dim))
            else:
                self.norm.append(nn.Identity())

        # classifier
        self.cls_head = MaskedConv1D(
                feat_dim, num_classes, kernel_size,
                stride=1, padding=kernel_size//2
            )

        # use prior in model initialization to improve stability
        # this will overwrite other weight init
        if prior_prob > 0:
            bias_value = -(math.log((1 - prior_prob) / prior_prob))
            torch.nn.init.constant_(self.cls_head.conv.bias, bias_value)

        # a quick fix to empty categories:
        # the weights assocaited with these categories will remain unchanged
        # we set their bias to a large negative value to prevent their outputs
        if len(empty_cls) > 0:
            bias_value = -(math.log((1 - 1e-6) / 1e-6))
            for idx in empty_cls:
                torch.nn.init.constant_(self.cls_head.conv.bias[idx], bias_value)

    def forward(self, fpn_feats, fpn_masks):
        assert len(fpn_feats) == len(fpn_masks)

        # apply the classifier for each pyramid level
        out_logits = tuple()
        for _, (cur_feat, cur_mask) in enumerate(zip(fpn_feats, fpn_masks)):
            cur_out = cur_feat
            for idx in range(len(self.head)):
                cur_out, _ = self.head[idx](cur_out, cur_mask)
                cur_out = self.act(self.norm[idx](cur_out))
            cur_logits, _ = self.cls_head(cur_out, cur_mask)
            out_logits += (cur_logits, )

        # fpn_masks remains the same
        return out_logits

class fpn_fc(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(fpn_fc, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, feature_list):
        # device = torch.device("cuda:0")
        # feature_list = feature_list.to(device)
        # 将6个特征张量连接起来
        concatenated_features = torch.cat(feature_list, dim=1)  # 沿着第二维度连接
        #print(concatenated_features.shape)
        # 通过全连接层进行线性变换
        # print(concatenated_features.is_cuda)
        output = self.fc(concatenated_features)

        return output

class UMMA(nn.Module):

    def __init__(
        self,
        # x_input,
        padding_val,
        input_dim,             # input feat dim
        embd_dim,              # output feat channel of the embedding network
        n_classes,
        device_umma,
        max_seq_len =768,
        fpn_out_c = 256,
        num_fpn_level = 2,
        head_dim = 256
        # device = torch.device(0),
    ):
        super().__init__()
        self.device = device_umma
        self.num_fpn_level = num_fpn_level
        FPN_in_c = [embd_dim] * num_fpn_level # default: (backbone_arch[-1] + 1),  6 = len(feats_backbone_output) , level of fpn
        # # image_height, image_width = pair(image_size)
        # # patch_height, patch_width = pair(patch_size)
        # in_channels = channels
        self.max_seq_len = max_seq_len
        self.n_classes = n_classes
        self.interpolator = DeepInterpolator(1, embd_dim, norm=False)
        self.backbone = ConvHRLRFullResSelfAttTransformerBackboneRevised(n_in = 1, n_embd = embd_dim, max_len=max_seq_len)
        # self.device = device
        self.convnet2d = TESTEtAl(200, 1024, patch_size=15)
        self.neck = FPN1D(in_channels=FPN_in_c, out_channel=fpn_out_c)
        # self.fpn_level_class = PtTransformerClsHead(input_dim=fpn_out_c, feat_dim=head_dim, num_classes=n_classes)
        # self.final_class = nn.Linear(1024, n_classes)
        self.conv_class = nn.Sequential(collections.OrderedDict([
            # ('conv_out_linear', nn.Linear(45000, 1024)),
            ('conv_class_linear', nn.Linear(256, n_classes)),
        ]))
        self.norm_class = nn.Sequential(collections.OrderedDict([
            ('norm_out_linear', nn.Linear(172800, 1024)),
            ('norm_class_linear', nn.Linear(1024, n_classes)),
        ]))
        # self.final_class = nn.Sequential(collections.OrderedDict([
        #   ('out_linear',    nn.Linear(316104, 1024)),
        #   ('class_linear',  nn.Linear(1024, n_classes)),
        # # ]))
        self.final_fpn_class = nn.Sequential(collections.OrderedDict([
            ('out_linear', nn.Linear(294912, 1024)),
            ('class_linear', nn.Linear(1024, n_classes)),
        ]))
        self.fpn1_class = nn.Sequential(collections.OrderedDict([
            ('out_linear', nn.Linear(196608, 1024)),
            ('class_linear', nn.Linear(1024, n_classes)),
        ]))
        self.fpn2_class = nn.Sequential(collections.OrderedDict([
            ('out_linear', nn.Linear(98304, 1024)),
            ('class_linear', nn.Linear(1024, n_classes)),
        ]))

        self.fpn3_class = nn.Sequential(collections.OrderedDict([
            ('out_linear', nn.Linear(49152, 1024)),
            ('class_linear', nn.Linear(1024, n_classes)),
        ]))
        # self.DCAE_class = nn.Sequential(collections.OrderedDict([
        #     ('out_linear', nn.Linear(1, 1024)),
        #     ('class_linear', nn.Linear(1024, n_classes)),
        # ]))


        self.moe1 = MoEBlock(dim=256, heads=4,num_experts=8,num_experts_per_token=2)
        self.moe2 = MoEBlock(dim=768, heads=4, num_experts=32, num_experts_per_token=2)
        self.moe3 = MoEBlock(dim=384, heads=4, num_experts=16, num_experts_per_token=2)
        self.moe4 = MoEBlock(dim=768, heads=4, num_experts=32, num_experts_per_token=2)
        self.moe5 = MoEBlock(dim=192, heads=4, num_experts=4, num_experts_per_token=2)
        # self.moe4 = MoEBlock(dim=294912, heads=4, num_experts=8, num_experts_per_token=2)
        # self.moe4 = MoEBlock(dim=414408, heads=4, num_experts=8, num_experts_per_token=2)

    def conv2d2videolist(self, conv2d_output):
        feats_list =[]
        # feats_dir = {}
        # 定义新的形状
        b, c, t = conv2d_output.size()
        # 将特征张量重新排列成 b*c*t 的形状
        # x = conv2d_output.view(b, c, t)
        # 为每个视频生成特征
        for level in range(b):
            feat = conv2d_output[level]
            feat_list = {'feats': feat.detach()}
            feats_list.append(feat_list)
        return feats_list

    def preprocessing(self, video_list, padding_val=0.0):
        """
            Generate batched features and masks from a list of dict items
        """
        feats = [x['feats'] for x in video_list]
        # feats = video_list
        feats_lens = torch.as_tensor([feat.shape[-1] for feat in feats])
        max_len = feats_lens.max(0).values.item()

        # if self.training:
        if 1:  #using train model
            assert max_len <= self.max_seq_len, "Input length must be smaller than max_seq_len during training"
            # set max_len to self.max_seq_len
            max_len = self.max_seq_len
            # batch input shape B, C, T
            batch_shape = [len(feats), feats[0].shape[0], max_len]
            batched_inputs = feats[0].new_full(batch_shape, padding_val)
            for feat, pad_feat in zip(feats, batched_inputs):
                # pad_feat = feat.clone()
                pad_feat[..., :feat.shape[-1]].copy_(feat)
        else:
            assert len(video_list) == 1, "Only support batch_size = 1 during inference"
            # input length < self.max_seq_len, pad to max_seq_len
            if max_len <= self.max_seq_len:
                max_len = self.max_seq_len
            else:
                # pad the input to the next divisible size
                stride = self.max_div_factor
                max_len = (max_len + (stride - 1)) // stride * stride
            padding_size = [0, max_len - feats_lens[0]]
            batched_inputs = F.pad(
                feats[0], padding_size, value=padding_val).unsqueeze(0)

        # generate the mask
        batched_masks = torch.arange(max_len)[None, :] < feats_lens[:, None]

        # push to device(?)
        # batched_inputs = batched_inputs.to(self.device)
        # batched_masks = batched_masks.unsqueeze(1).to(self.device)
        batched_inputs = batched_inputs
        batched_masks = batched_masks.unsqueeze(1)
        return batched_inputs, batched_masks

    def forward(self, x_input):
        # batch the video list into feats (B, C, T) and masks (B, 1, T)
        # video_list = video_list.squeeze()
        # x = x.squeeze()
        # print(x_input.shape)
        conv2d_output = self.convnet2d(x_input)
        conv2d_output_moe = self.moe1(conv2d_output)
        # print(conv2d_output.shape)
        # conv2d_output =self.moe1(conv2d_output_raw)
        # print(conv2d_output.shape)
        # print(conv2d_output.shape)
        # skip = conv2d_output_moe
        video_list = self.conv2d2videolist(conv2d_output_moe)
        batched_inputs, batched_masks = self.preprocessing(video_list)
        # forward the network (backbone -> neck -> heads)
        norm_inputs, reco_result, cls_scores = self.interpolator(batched_inputs, batched_masks)  # DCAE

        # norm_inputs = self.moe2(norm_inputs)
        # print(norm_inputs.shape)
        # print(norm_inputs.size(), reco_result.size(), cls_scores)
        feats, masks = self.backbone(batched_inputs, norm_inputs, reco_result, batched_masks)
        # feats[0] =self.moe4(feats[0])
        # feats[1] = self.moe5(feats[1])
        # print("shape of bacbone_output:",len(feats[0]),len(feats[1]),len(feats[2]),len(feats))
        fpn_feats, fpn_masks = self.neck(feats, masks)
        # print(fpn_feats[0].shape)
        # print(fpn_feats[1].shape)
        # print("finish_neck")
        # out_cls_logits = self.fpn_level_class(fpn_feats, fpn_masks)
        # # print(out_cls_logits[0].shape)
        # out_cls_logits = [x.permute(0, 2, 1) for x in out_cls_logits]
        # # print(out_cls_logits[0].shape)
        # #
        # # #fpn_fc 5.25
        # level_fc_list = []  #降维后的out_cls_logits
        # # #将每层维度降维：B,X,Z --->B,(X*Z)
        # #

        # level_fc_list.append(conv2d_output_moe.flatten(1))
        # output_fpn = torch.cat(level_fc_list, dim=1)


        # output_fpn = self.moe4(output_fpn)
        # class_input_dim = sum([x.shape[1] for x in level_fc_list])  # 计算输入维度
        # # print('class_input_channel',class_input_dim)
        # # print(level_fc_list[0].shape)
        # # output_fpn = torch.cat(level_fc_list, dim=1)
        # # classifier = fpn_fc(input_dim= class_input_dim, output_dim= self.n_classes)
        # # output = classifier(level_fc_list)


        # 5.8 use final_fpn_level
        # output = fpn_feats[0].flatten(1)
        # print(output.shape)
        # output = self.final_class(output)

        # # 5.8 use three feats
        # level_feat_list = []
        # level_feat_list.append(conv2d_output.flatten(1))
        # level_feat_list.append(norm_inputs.flatten(1))
        # level_feat_list.append(fpn_feats[0].flatten(1))
        # output = torch.cat(level_feat_list, dim=1)
        # output = self.final_class(output)

        #5.9 use 3 MOE
        # conv2d_output_moe = self.moe1(conv2d_output)
        # norm_inputs_moe =self.moe2(norm_inputs)
        # fpn_final_level_moe = self.moe3(fpn_feats[0])
        # # print(fpn_final_level_moe == fpn_feats[0])
        # level_feat_list = []
        # level_feat_list.append(conv2d_output_moe.flatten(1))
        # level_feat_list.append(norm_inputs_moe.flatten(1))
        # level_feat_list.append(fpn_final_level_moe.flatten(1))
        # output = torch.cat(level_feat_list, dim=1)
        # output = self.final_class(output)
        # conv2d_output = self.conv_class(conv2d_output_moe.flatten(1))


        # # 5.9 use 3 MOE在特征生成后
        # level_feat_list = []
        # level_feat_list.append(conv2d_output.flatten(1))
        # level_feat_list.append(norm_inputs.flatten(1))
        # fpn_final_level_moe = self.moe3(fpn_feats[0])
        # level_feat_list.append(fpn_final_level_moe.flatten(1))
        # output = torch.cat(level_feat_list, dim=1)
        # output = self.final_class(output)
        # # print(output.shape)

        # # 5.10 use 3loss
        # level_feat_list = []
        # conv2d_output = conv2d_output.flatten(1)
        # level_feat_list.append(conv2d_output)
        # level_feat_list.append(norm_inputs.flatten(1))
        # level_feat_list.append(fpn_feats[0].flatten(1))
        # output = torch.cat(level_feat_list, dim=1)
        # output = self.final_class(output)
        # conv2d_output = self.conv_class(conv2d_output)
        # norm_inputs = norm_inputs.flatten(1)
        # norm_output = self.norm_class(norm_inputs)
        # # print(output.shape)
        # norm_output = self.norm_class(norm_inputs.flatten(1))

        # # 5.20 use 2 loss+ moe-middle
        # level_feat_list = []
        # fpn_final_level_moe = self.moe3(fpn_feats[1])
        # level_feat_list.append(conv2d_output_moe.flatten(1))
        # level_feat_list.append(norm_inputs.flatten(1))
        # level_feat_list.append(fpn_final_level_moe.flatten(1))
        #
        # output = torch.cat(level_feat_list, dim=1)
        # output = self.final_class(output)
        # conv2d_output_moe = self.conv_class(conv2d_output_moe.flatten(1))

        # 5.20
        # output_fpn_moe = self.moe4(output_fpn)

        # output = self.final_fpn_class(output_fpn)
        # fpn1_moe = self.moe2(fpn_feats[0])
        fpn2_moe = self.moe3(fpn_feats[1])
        # fpn3_moe = self.moe5(fpn_feats[2])
        # fpn1_oupt = self.fpn1_class(fpn1_moe.flatten(1))

        # level_feat_list = []
        # level_feat_list.append(fpn1_moe.flatten(1))
        # level_feat_list.append(fpn2_moe.flatten(1))
        # # fpn_out = torch.cat(level_feat_list, dim=1)
        # level_feat_list = []
        # level_feat_list.append(fpn2_moe.flatten(1))
        # level_feat_list.append(conv2d_output.flatten(1))
        # fpn_out = torch.cat(level_feat_list, dim=1)
        # # fpn_out = self.moe2(fpn_out)
        # fpn_out = self.final_fpn_class(fpn_out)
        fpn2_oupt = self.fpn2_class(fpn2_moe.flatten(1))
        # fpn3_moe = self.fpn3_class(fpn3_moe.flatten(1))
        # norm_inputs_moe = self.moe4(norm_inputs)
        conv2d_output = self.conv_class(conv2d_output_moe.flatten(1))
        # norm_output = self.norm_class(norm_inputs_moe.flatten(1))
        # cls_scores_output = self.DCAE_class(cls_scores)
        return conv2d_output,fpn2_oupt

if __name__ == '__main__':
    #
    # x =[
    # {'feats': torch.randn(1024,200)},  # 第一个视频的特征
    # # {'feats': torch.randn(4096,120)},  # 第二个视频的特征
    # # 可能还有更多的视频特征...
    # ]
    # print(x)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img_size = 15
    x = torch.rand(10, 1, 200, img_size, img_size)
    x = x.to(device)
    model = UMMA(padding_val=0, input_dim=225, embd_dim=256, n_classes=17,device_umma=0)
    model = model.to(device)
    y = model(x)
    model.eval()
    from thop import profile
    flops, params = profile(model, (x,))

    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')