# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from .swin_transformer import SwinTransformer
from .swin_transformer_v2 import SwinTransformerV2
from .swin_transformer_g import SwinTransformerG
from .swin_transformer_s import SwinTransformerS


def build_model(config):
    model_type = config.MODEL.TYPE

    # accelerate layernorm
    if config.FUSED_LAYERNORM:
        try:
            import apex as amp
            layernorm = amp.normalization.FusedLayerNorm
        except:
            layernorm = None
            print("To use FusedLayerNorm, please install apex.")
    else:
        import torch.nn as nn
        layernorm = nn.LayerNorm

    if model_type == 'swin':
        model = SwinTransformer(img_size=config.DATA.IMG_SIZE,
                                patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                in_chans=config.MODEL.SWIN.IN_CHANS,
                                num_classes=config.MODEL.NUM_CLASSES,
                                embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                depths=config.MODEL.SWIN.DEPTHS,
                                num_heads=config.MODEL.SWIN.NUM_HEADS,
                                window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                qk_scale=config.MODEL.SWIN.QK_SCALE,
                                drop_rate=config.MODEL.DROP_RATE,
                                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                ape=config.MODEL.SWIN.APE,
                                norm_layer=layernorm,
                                patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                fused_window_process=config.FUSED_WINDOW_PROCESS)
    elif model_type == 'swinv2':
        model = SwinTransformerV2(img_size=config.DATA.IMG_SIZE,
                                  patch_size=config.MODEL.SWINV2.PATCH_SIZE,
                                  in_chans=config.MODEL.SWINV2.IN_CHANS,
                                  num_classes=config.MODEL.NUM_CLASSES,
                                  embed_dim=config.MODEL.SWINV2.EMBED_DIM,
                                  depths=config.MODEL.SWINV2.DEPTHS,
                                  num_heads=config.MODEL.SWINV2.NUM_HEADS,
                                  window_size=config.MODEL.SWINV2.WINDOW_SIZE,
                                  mlp_ratio=config.MODEL.SWINV2.MLP_RATIO,
                                  qkv_bias=config.MODEL.SWINV2.QKV_BIAS,
                                  drop_rate=config.MODEL.DROP_RATE,
                                  drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                  ape=config.MODEL.SWINV2.APE,
                                  patch_norm=config.MODEL.SWINV2.PATCH_NORM,
                                  use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                  pretrained_window_sizes=config.MODEL.SWINV2.PRETRAINED_WINDOW_SIZES)
    elif model_type == 'swin_g':
        model = SwinTransformerG(img_size=config.DATA.IMG_SIZE,
                                 patch_size=config.MODEL.SWIN_G.PATCH_SIZE,
                                 in_chans=config.MODEL.SWIN_G.IN_CHANS,
                                 num_classes=config.MODEL.NUM_CLASSES,
                                 embed_dim=config.MODEL.SWIN_G.EMBED_DIM,
                                 depths=config.MODEL.SWIN_G.DEPTHS,
                                 num_heads=config.MODEL.SWIN_G.NUM_HEADS,
                                 window_size=config.MODEL.SWIN_G.WINDOW_SIZE,
                                 mlp_ratio=config.MODEL.SWIN_G.MLP_RATIO,
                                 qkv_bias=config.MODEL.SWIN_G.QKV_BIAS,
                                 qk_scale=config.MODEL.SWIN_G.QK_SCALE,
                                 drop_rate=config.MODEL.DROP_RATE,
                                 drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                 ape=config.MODEL.SWIN_G.APE,
                                 norm_layer=layernorm,
                                 patch_norm=config.MODEL.SWIN_G.PATCH_NORM,
                                 use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                 token_drop_max=config.MODEL.SWIN_G.TOKEN_DROP_MAX,
                                 token_drop_min=config.MODEL.SWIN_G.TOKEN_DROP_MIN,
                                 token_drop_pattern=config.MODEL.SWIN_G.TOKEN_DROP_PATTERN,
                                 fused_window_process=config.FUSED_WINDOW_PROCESS)
    if model_type == 'swin_s':
        model = SwinTransformerS(img_size=config.DATA.IMG_SIZE,
                                 patch_size=config.MODEL.SWIN_S.PATCH_SIZE,
                                 in_chans=config.MODEL.SWIN_S.IN_CHANS,
                                 num_classes=config.MODEL.NUM_CLASSES,
                                 embed_dim=config.MODEL.SWIN_S.EMBED_DIM,
                                 depths=config.MODEL.SWIN_S.DEPTHS,
                                 num_heads=config.MODEL.SWIN_S.NUM_HEADS,
                                 window_size=config.MODEL.SWIN_S.WINDOW_SIZE,
                                 std=config.MODEL.SWIN_S.STD,
                                 mlp_ratio=config.MODEL.SWIN_S.MLP_RATIO,
                                 qkv_bias=config.MODEL.SWIN_S.QKV_BIAS,
                                 qk_scale=config.MODEL.SWIN_S.QK_SCALE,
                                 drop_rate=config.MODEL.DROP_RATE,
                                 drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                 ape=config.MODEL.SWIN_S.APE,
                                 norm_layer=layernorm,
                                 patch_norm=config.MODEL.SWIN_S.PATCH_NORM,
                                 use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                 split_kv=config.MODEL.SWIN_S.SPLIT_KV,
                                 fused_window_process=config.FUSED_WINDOW_PROCESS)
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
