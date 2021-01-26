from modeling.backbone import resnet

def build_backbone(backbone, output_stride, BatchNorm, model_path):
    if backbone == 'resnet':
        print(model_path)
        return resnet.ResNet101(output_stride, BatchNorm, model_path)
    else:
        raise NotImplementedError
