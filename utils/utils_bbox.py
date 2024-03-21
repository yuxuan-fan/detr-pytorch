import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms


class DecodeBox(nn.Module):
    """预测结果的解码 This module converts the model's output into the format expected by the coco api"""
    def box_cxcywh_to_xyxy(self, x):# 用于将边界框的中心坐标宽高表示方式转换为左上角和右下角坐标表示方式
        x_c, y_c, w, h = x.unbind(-1) # 代码首先对输入张量 x 进行 unbind 操作，将最后一个维度拆分为四个独立的张量 (x_c, y_c, w, h)。
        # 这里的 unbind(-1) 操作将沿着最后一个维度拆分张量。
        #左上角右下角 函数将左上角 (x_min, y_min) 和右下角 (x_max, y_max) 的坐标堆叠在一起，形成一个新的张量 b。参数 dim=-1 表示在最后一个维度上进行堆叠
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
            (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=-1)
    
    @torch.no_grad()
    def forward(self, outputs, target_sizes, confidence):
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2
# 首先，代码对 out_logits 使用 softmax 函数进行类别概率的计算，并使用 max 函数找到最大概率对应的类别生成预测的得分 scores 和标签 labels。
        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)
        
        # convert to [x0, y0, x1, y1] format
        boxes = self.box_cxcywh_to_xyxy(out_bbox)

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w    = target_sizes.unbind(1)
        img_h           = img_h.float()
        img_w           = img_w.float()
        scale_fct       = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes           = boxes * scale_fct[:, None, :]
        # 接下来，代码将处理后的边界框、得分和标签堆叠在一起，形成一个新的张量 outputs。
        # 通过使用 torch.cat 函数进行堆叠，并在最后一个维度上添加维度为 1 的张量，以便与边界框、得分和标签的维度保持一致。
        outputs = torch.cat([
                torch.unsqueeze(boxes[:, :, 1], -1),
                torch.unsqueeze(boxes[:, :, 0], -1),
                torch.unsqueeze(boxes[:, :, 3], -1),
                torch.unsqueeze(boxes[:, :, 2], -1),
                torch.unsqueeze(scores, -1),
                torch.unsqueeze(labels.float(), -1),    
            ], -1)
        
        results = []#最后，代码通过遍历 outputs 中的每个元素，将得分大于 confidence 的结果添加到 results 列表中。
        for output in outputs:
            results.append(output[output[:, 4] > confidence])
        # results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
        return results