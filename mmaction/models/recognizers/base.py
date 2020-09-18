import logging
from abc import ABCMeta, abstractmethod
import torch.nn as nn

from slowfast.utils.parser import  load_config,parse_args
from slowfast.models import head_helper

class BaseRecognizer(nn.Module):
    """Base class for recognizers"""

    __metaclass__ = ABCMeta

    def __init__(self):
        super(BaseRecognizer, self).__init__()
        self.args = parse_args()
        self.cfg = load_config(self.args)
        self.enable_detection = self.cfg.DETECTION.ENABLE

    @property
    def with_tenon_list(self):
        return hasattr(self, 'tenon_list') and self.tenon_list is not None

    @property
    def with_cls(self):
        return hasattr(self, 'cls_head') and self.cls_head is not None

    @abstractmethod
    def forward_train(self, num_modalities, **kwargs):
        pass

    @abstractmethod
    def forward_test(self, num_modalities, **kwargs):
        pass

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            logger = logging.getLogger()
            logger.info("load model from: {}".format(pretrained))

    # def forward(self, num_modalities, img_meta, return_loss=True, **kwargs):
    #     num_modalities = int(num_modalities[0])
    #     if return_loss:
    #         return self.forward_train(num_modalities, img_meta, **kwargs)
    #     else:
    #         return self.forward_test(num_modalities, img_meta, **kwargs)

    def forward(self, input, bboxes=None):
        # self.backbone = builder.build_backbone(backbone)
        input1= input[0]
        # print("input[0]: ",input[0].shape)
        try:
            input1 = input1.reshape(
            (-1, 3) + input1.shape[3:])                                      # (224,224)
        except RuntimeError:
            input1 = input1.squeeze(0)
            input1 = input1.reshape(
            (-1, 3) + input1.shape[2:])
        # print("input1: ", input1.shape)

        bs = self.cfg.TRAIN.BATCH_SIZE
        nf = self.cfg.DATA.NUM_FRAMES
        out = self.extract_feat(input1)
        # print("out: ", out.shape)
        m = nn.UpsamplingNearest2d(scale_factor=2)
        out = m(out)
        # print("out: ", out.shape)

        try:
            out = out.unsqueeze(0).view(bs, nf, 2048, 14, 14 )             # for train
        except RuntimeError:
            out = out.unsqueeze(0).view(-1, nf, 2048, 16, 16 )             # for val
        x = out.permute(0,2,1,3,4)                      ## # x[4,2048,8,7,7]

        if self.enable_detection:
            self.head = head_helper.TSMRoIHead(
                dim_in=[self.cfg.RESNET.WIDTH_PER_GROUP * 32],
                num_classes = self.cfg.MODEL.NUM_CLASSES,
                pool_size = [[self.cfg.DATA.NUM_FRAMES //1, 1, 1]],             # pool_size[0][0],
                resolution = [[self.cfg.DETECTION.ROI_XFORM_RESOLUTION] * 2],
                scale_factor = [self.cfg.DETECTION.SPATIAL_SCALE_FACTOR],
                dropout_rate = self.cfg.MODEL.DROPOUT_RATE,
                act_func = self.cfg.MODEL.HEAD_ACT,
                aligned = self.cfg.DETECTION.ALIGNED,
            )
            input[0] = x                                                    # .cuda()
            x = input
            x = self.head(x, bboxes)


        return x                  ## input:[32,3,224,224] ->[32,2048,7,7]         [9, 80]?

        # else:
        #     return self.forward_test(num_modalities, img_meta, **kwargs)
