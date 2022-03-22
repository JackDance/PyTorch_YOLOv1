import torch
import torch.nn as nn
from utils import Conv, SPP
from backbone import resnet18
import numpy as np
import tools

class myYOLO(nn.Module):
    def __init__(self, device, input_size=None, num_classes=20, trainable=False, conf_thresh=0.01, nms_thresh=0.5):
        super(myYOLO, self).__init__()
        self.device = device                           # cuda或者是cpu
        self.num_classes = num_classes                 # 类别的数量
        self.trainable = trainable                     # 训练的标记, 用来区分训练阶段和测试阶段
        self.conf_thresh = conf_thresh                 # 得分阈值
        self.nms_thresh = nms_thresh                   # NMS阈值
        self.stride = 32                               # 网络的最大步长, 因为网络输入为416, 最后规定的输出为13, 故最终步长为416/13=32
        self.grid_cell = self.create_grid(input_size)  # 网格坐标矩阵
        self.input_size = input_size                   # 输入图像大小

        # >>>>>>>>>>>>>>>> backbone网络 <<<<<<<<<<<<<<<<<
        # backbone: resnet18
        self.backbone = resnet18(pretrained=True)
        c5 = 512

        # >>>>>>>>>>>>>>>> Neck网络 <<<<<<<<<<<<<<<<<<<<<
        # neck: SPP
        self.neck = nn.Sequential(
            SPP(),
            Conv(c5*4, c5, k=1), # 在SPP之后接一个1x1卷积层，来恢复通道数
        )

        # >>>>>>>>>>>>>>>> detection head网络 <<<<<<<<<<<
        # detection head
        self.convsets = nn.Sequential(
            Conv(c5, 256, k=1),
            Conv(256, 512, k=3, p=1),
            Conv(512, 256, k=1),
            Conv(256, 512, k=3, p=1)
        )

        # >>>>>>>>>>>>>>>> prediction层 <<<<<<<<<<<<<<<<
        # pred
        self.pred = nn.Conv2d(512, 1 + self.num_classes + 4, 1)
    

    def create_grid(self, input_size):
        """ 
            用于生成G矩阵，该矩阵的每个元素都是特征图上的像素坐标(gridx, gridy)。
        """
        # 获得图像的宽和高
        w, h = input_size, input_size
        # generate grid cells
        # 特征图的宽和高
        ws, hs = w // self.stride, h // self.stride
        # 使用torch.meshgrid函数来获得矩阵G的x坐标和y坐标
        grid_y, grid_x = torch.meshgrid([torch.arange(hs), torch.arange(ws)])
        # 将xy两部分坐标拼接到一起，得到矩阵G
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).float()
        # 最终G矩阵的维度是(1,H*W,2)
        grid_xy = grid_xy.view(1, hs*ws, 2).to(self.device)
        
        return grid_xy


    def set_grid(self, input_size):
        """
            用于重置G矩阵。
        """
        self.input_size = input_size
        self.grid_cell = self.create_grid(input_size)


    def decode_boxes(self, pred):
        """
            将网络输出的tx,ty,tw,th四个量转换成bbox的(x1, y1), (x2, y2)。
            pred: [B, HxW, 4] 的txtytwth预测
            output box: [xmin, ymin, xmax, ymax]
        """
        output = torch.zeros_like(pred)
        # 得到所有bbox 的中心点坐标和宽高
        pred[:, :, :2] = torch.sigmoid(pred[:, :, :2]) + self.grid_cell
        pred[:, :, 2:] = torch.exp(pred[:, :, 2:])

        # 将所有bbox的中心点坐标和宽高换算成x1y1x2y2形式
        output[:, :, 0] = pred[:, :, 0] * self.stride - pred[:, :, 2] / 2
        output[:, :, 1] = pred[:, :, 1] * self.stride - pred[:, :, 3] / 2
        output[:, :, 2] = pred[:, :, 0] * self.stride + pred[:, :, 2] / 2
        output[:, :, 3] = pred[:, :, 1] * self.stride + pred[:, :, 3] / 2
        
        return output


    def nms(self, dets, scores):
        """"
        Pure Python NMS baseline.
        该代码源自Faster RCNN项目。
        """
        x1 = dets[:, 0]  #xmin
        y1 = dets[:, 1]  #ymin
        x2 = dets[:, 2]  #xmax
        y2 = dets[:, 3]  #ymax

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        

        keep = []                                             
        while order.size > 0:
            i = order[0]
            keep.append(i)
            # 计算交集的左上角点和右下角点的坐标
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            # 计算交集的宽高
            w = np.maximum(1e-28, xx2 - xx1)
            h = np.maximum(1e-28, yy2 - yy1)
            # 计算交集的面积
            inter = w * h

            # 计算交并比
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            # 滤除超过nms阈值的检测框
            inds = np.where(ovr <= self.nms_thresh)[0]
            order = order[inds + 1]

        return keep


    def postprocess(self, bboxes, scores):
        """
        在对输出的txtytwth预测处理成bbox的x1y1x2y2后，还需要对预测的结果进行一次处理，后处理的作用是：
        1）过滤掉得分很低的框;
        2）过滤掉针对同一目标的冗余检测，即nms处理。
        bboxes: (HxW, 4), bsize = 1
        scores: (HxW, num_classes), bsize = 1
        """

        cls_inds = np.argmax(scores, axis=1) # 按行比较，返回最大值对应的索引
        scores = scores[(np.arange(scores.shape[0]), cls_inds)]
        
        # threshold
        keep = np.where(scores >= self.conf_thresh) # 返回索引keep
        bboxes = bboxes[keep]
        scores = scores[keep]
        cls_inds = cls_inds[keep]

        # NMS
        keep = np.zeros(len(bboxes), dtype=np.int)
        for i in range(self.num_classes):
            inds = np.where(cls_inds == i)[0]
            if len(inds) == 0:
                continue
            c_bboxes = bboxes[inds]
            c_scores = scores[inds]
            c_keep = self.nms(c_bboxes, c_scores)
            keep[inds[c_keep]] = 1
        # 获得最终的检测结果
        keep = np.where(keep > 0)
        bboxes = bboxes[keep]
        scores = scores[keep]
        cls_inds = cls_inds[keep]

        # bboxes：包含每一个检测框的x1,y1,x2,y2坐标; scores：包含每一个检测框的得分; cls_inds: 包含每一个检测框的预测类别序号
        return bboxes, scores, cls_inds


    def forward(self, x, target=None):
        """
        前向推理的代码，主要分为两部分：
        训练部分：网络得到obj，cls和txtytwth三个分支的预测结果，然后计算loss;
        推理部分：输出经过后处理得到的bbox，cls和每个bbox的预测得分。
        Args:
            x: 输入
            target: 标签

        Returns:

        """
        # backbone主干网络
        c5 = self.backbone(x)

        # neck网络
        p5 = self.neck(c5)

        # detection head网络
        p5 = self.convsets(p5)

        # 预测层
        pred = self.pred(p5)

        # 对pred 的size做一些view调整，便于后续的处理
        # [B, C, H, W] -> [B, C, H*W] -> [B, H*W, C]
        pred = pred.view(p5.size(0), 1 + self.num_classes + 4, -1).permute(0, 2, 1) # permute:将tensor的维度换位

        # 从pred中分离出objectness预测、类别class预测、bbox的txtytwth预测  
        # [B, H*W, 1]
        conf_pred = pred[:, :, :1] # 分离出objectness预测
        # [B, H*W, num_cls]
        cls_pred = pred[:, :, 1 : 1 + self.num_classes] # 分离出class预测
        # [B, H*W, 4]
        txtytwth_pred = pred[:, :, 1 + self.num_classes:] # 分离出bbox的txtytwth预测

        # train，训练时，网络返回三部分的loss
        if self.trainable:
            conf_loss, cls_loss, bbox_loss, total_loss = tools.loss(pred_conf=conf_pred, 
                                                                    pred_cls=cls_pred,
                                                                    pred_txtytwth=txtytwth_pred,
                                                                    label=target
                                                                    )

            return conf_loss, cls_loss, bbox_loss, total_loss            
        # test，执行推理
        else:
            with torch.no_grad(): # torch.no_grad() 是一个上下文管理器，被该语句wrap起来的部分将不会track梯度。
                # batch size = 1
                # 测试时，笔者默认batch是1，因此，我们不需要用batch这个维度，用[0]将其取走。
                # [B, H*W, 1] -> [H*W, 1]
                conf_pred = torch.sigmoid(conf_pred)[0]
                # [B, H*W, 4] -> [H*W, 4], 并做归一化处理，同时使用clamp函数保证归一化的结果都是01之间的数
                bboxes = torch.clamp((self.decode_boxes(txtytwth_pred) / self.input_size)[0], 0., 1.)
                # [B, H*W, 1] -> [H*W, num_class]，得分=<类别置信度>乘以<objectness置信度>
                scores = (torch.softmax(cls_pred[0, :, :], dim=1) * conf_pred)
                
                # 将预测放在cpu处理上，以便进行后处理
                scores = scores.to('cpu').numpy()
                bboxes = bboxes.to('cpu').numpy()
                
                # 后处理
                bboxes, scores, cls_inds = self.postprocess(bboxes, scores)
                # bboxes:包含每一个检测框的x1y1x2y2坐标;
                # scores：包含每一个检测框的得分；
                # cls_inds：包含每一个检测框的预测类别序号。
                return bboxes, scores, cls_inds
