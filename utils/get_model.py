import torchvision

from utils.ood_detection.ood_detector import CNN_IBP, MiniNet
from vit.src.model import VisionTransformer as ViT


def get_model_from_args(args, model_name, num_classes):
    if model_name.lower() == "resnet":
        model = torchvision.models.resnet18(pretrained=False, num_classes=num_classes).to(device=args.device) # cuda()
    elif model_name.lower() == "vit":
        #"""
        model = ViT(image_size=(args.image_size, args.image_size),
                    patch_size=(args.patch_size, args.patch_size) if args.patch_size else (16, 16), # 224,224
                    emb_dim=args.emb_dim if args.emb_dim else 768,
                    mlp_dim=args.mlp_dim if args.mlp_dim else 3072,
                    num_heads=args.num_heads, #12 #also a very small amount of heads to speed up training
                    num_layers=args.num_layers,  # 12 # 5 is a very small vit model
                    num_classes=num_classes,  # 2 for OOD detection, 10 or more for classification
                    attn_dropout_rate=args.attn_dropout_rate if args.attn_dropout_rate else 0.0,
                    dropout_rate=args.dropout_rate if args.dropout_rate else 0.1,
                    contrastive=False,
                    timm=True,
                    head=args.head if args.head else None,
                    ).to(device=args.device) # cuda()

        """
        feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
        model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
        inputs = feature_extractor(image, return_tensors="pt")
        """
    elif model_name.lower() == "cnn_ibp":
        #TODO currently not working, throwing error
        #"RuntimeError: mat1 dim 1 must match mat2 dim 0"
        model = CNN_IBP().to(device=args.device)
    elif model_name.lower() == "mininet":
        model = MiniNet().to(device=args.device)
    else:
        raise ValueError("Error - wrong model specified in 'args'")
    return model
