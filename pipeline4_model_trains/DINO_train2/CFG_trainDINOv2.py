import torch


class CFG_DINOV2:
    # ---------- general ----------
    seed = 3407
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 8
    K_samples = 16
    batch_size = 64
    epoch = 12

    # ---------- mode ----------
    # mode: 'base', 'auged1', 'auged1_withoutpadding'
    mode = 'auged1_withoutpadding'

    weight_decay = None
    grad_clip = None
    freeze_blocks = None

    # lr
    lr_backbone = None
    lr_head = None
    layer_decay = None

    # augment
    use_augment = False
    use_pad_aug = False
    mixup_alpha = 0.2
    mixup_prob = 1.0
    label_smoothing = 0.1
    pad_prob = 0.6
    pad_max_area = 0.2
    pad_min_bar = 0.03
    pad_shift_frac = 0.02

    # MoE
    use_moe = False
    moe_num_experts = 4
    moe_depth = 3
    moe_mlp_ratio = 2.0
    moe_dropout = 0.5
    moe_drop_path = 0.1
    moe_modal_ln = False
    moe_modal_scale = False
    moe_feat_drop = 0.0

    @classmethod
    def setup(cls):
        if cls.mode == 'base':
            cls.lr_backbone = 1e-5
            cls.lr_head = 1e-5
            cls.weight_decay = 1e-2
            cls.grad_clip = None
            cls.freeze_blocks = 20
            cls.use_augment = False
            cls.use_pad_aug = False
            cls.use_moe = False
            cls.layer_decay = None

        elif cls.mode == 'auged1':
            cls.weight_decay = 5e-2
            cls.grad_clip = 1.0
            cls.freeze_blocks = 8

            cls.lr_backbone = 1e-5
            cls.lr_head = 3e-4
            cls.layer_decay = 0.85

            cls.use_augment = True
            cls.use_pad_aug = True

            cls.use_moe = True
            cls.moe_num_experts = 4
            cls.moe_depth = 3
            cls.moe_mlp_ratio = 2.0
            cls.moe_dropout = 0.6
            cls.moe_drop_path = 0.10

            cls.moe_modal_ln = False
            cls.moe_modal_scale = False
            cls.moe_feat_drop = 0.0

        elif cls.mode == 'auged1_withoutpadding':
            cls.weight_decay = 5e-2
            cls.grad_clip = 1.0
            cls.freeze_blocks = 8

            cls.lr_backbone = 1e-5
            cls.lr_head = 3e-4
            cls.layer_decay = 0.85

            cls.use_augment = True
            cls.use_pad_aug = False   #  padding augmentation

            cls.use_moe = True
            cls.moe_num_experts = 4
            cls.moe_depth = 3
            cls.moe_mlp_ratio = 2.0
            cls.moe_dropout = 0.6
            cls.moe_drop_path = 0.10

            cls.moe_modal_ln = False
            cls.moe_modal_scale = False
            cls.moe_feat_drop = 0.0

        else:
            raise ValueError(
                f"Unsupported mode: {cls.mode}. Expected 'base', 'auged1', or 'auged1_withoutpadding'."
            )


CFG_DINOV2.setup()