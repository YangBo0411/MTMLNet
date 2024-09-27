import numpy as np

from ..metrics import ap_per_class
import torch
from skimage import measure
import torch.nn.functional as F



class ROCMetric():
    """Computes pixAcc and mIoU metric scores
    """
    def __init__(self, nclass, bins):  #bin的意义实际上是确定ROC曲线上的threshold取多少个离散值
        super(ROCMetric, self).__init__()
        self.nclass = nclass
        self.bins = bins
        self.tp_arr = np.zeros(self.bins+1)
        self.pos_arr = np.zeros(self.bins+1)
        self.fp_arr = np.zeros(self.bins+1)
        self.neg_arr = np.zeros(self.bins+1)
        self.class_pos=np.zeros(self.bins+1)
        # self.reset()
    def update(self, preds, labels):
        for iBin in range(self.bins+1):
            score_thresh = (iBin + 0.0) / self.bins
            # _, preds = torch.max(preds, 1)
            # print(iBin, "-th, score_thresh: ", score_thresh)
            i_tp, i_pos, i_fp, i_neg,i_class_pos = cal_tp_pos_fp_neg(preds, labels, self.nclass,score_thresh)
            self.tp_arr[iBin]   += i_tp
            self.pos_arr[iBin]  += i_pos
            self.fp_arr[iBin]   += i_fp
            self.neg_arr[iBin]  += i_neg
            self.class_pos[iBin]+=i_class_pos
    def get(self):
        tp_rates    = self.tp_arr / (self.pos_arr + 0.001)
        fp_rates    = self.fp_arr / (self.neg_arr + 0.001)
        recall      = self.tp_arr / (self.pos_arr   + 0.001)
        precision   = self.tp_arr / (self.class_pos + 0.001)
        return tp_rates, fp_rates, recall, precision

    def reset(self):

        self.tp_arr   = np.zeros([11])
        self.pos_arr  = np.zeros([11])
        self.fp_arr   = np.zeros([11])
        self.neg_arr  = np.zeros([11])
        self.class_pos= np.zeros([11])

def cal_tp_pos_fp_neg(output, target, nclass, score_thresh):

    # _, predict = torch.max(output, 1)
    # predict = (torch.sigmoid(output) > score_thresh).float()
    predict = (torch.sigmoid(output[:, 1, :, :]) > score_thresh).float()
    # if len(target.shape) == 3:
    #     target = np.expand_dims(target.float(), axis=1)
    # if len(target.shape) == 3:
    #     target = target.unsqueeze(1).float()
    # elif len(target.shape) == 4:
    #     target = target.float()
    # else:
    #     raise ValueError("Unknown target dimension")

    target = target.to('cuda')
    intersection = predict * ((predict == target).float())

    tp = intersection.sum()
    fp = (predict * ((predict != target).float())).sum()
    tn = ((1 - predict) * ((predict == target).float())).sum()
    fn = (((predict != target).float()) * (1 - predict)).sum()
    pos = tp + fn
    neg = fp + tn
    class_pos= tp+fp

    return tp, pos, fp, neg, class_pos

#-----------------------------------------------yb 2024.03.14-----------------------------------------------------
def batch_pix_accuracy(output, target):
    """Batch Pixel Accuracy
    Args:
        predict: input 4D tensor
        target: label 3D tensor
    """
    _, predict = torch.max(output, 1)  # 沿类别维度[1]进行输出，[4,640,640] 表示在640*640的4张图片中的预测值，数据表示类别预测值
    #+1的原因是因为在加载数据集时将白色设置成为了-1
    # predict = predict.cpu().numpy().astype('int64') + 1 #源码
    # target = target.cpu().numpy().astype('int64') + 1   #源码
    predict = predict.cpu().numpy().astype('int64')
    target = target.cpu().numpy().astype('int64')

    pixel_labeled = np.sum(target > 0)
    pixel_correct = np.sum((predict == target)*(target > 0))
    assert pixel_correct <= pixel_labeled, \
        "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled

def batch_intersection_union(output, target, nclass):
    """Batch Intersection of Union
    Args:
        predict: input 4D tensor
        target: label 3D tensor
        nclass: number of categories (int)
    """
    _, predict = torch.max(output, 1)
    mini = 1
    maxi = nclass
    nbins = nclass
    # predict = predict.cpu().numpy().astype('int64') + 1   #源码
    # target = target.cpu().numpy().astype('int64') + 1     #源码
    predict = predict.cpu().numpy().astype('int64')
    target = target.cpu().numpy().astype('int64')

    # predict = predict * (target > 0).astype(predict.dtype)
    intersection = predict * (predict == target)
    # areas of intersection and union
    area_inter, _ = np.histogram(intersection, bins=nbins, range=(mini, maxi))
    area_pred, _ = np.histogram(predict, bins=nbins, range=(mini, maxi))
    area_lab, _ = np.histogram(target, bins=nbins, range=(mini, maxi))
    area_union = area_pred + area_lab - area_inter
    assert (area_inter <= area_union).all(), \
        "Intersection area should be smaller than Union area"
    return area_inter, area_union
def batch_intersection_union_miou(output, target):
    """mIoU"""
    # inputs are tensor
    # the category 0 is ignored class, typically for background / boundary
    mini = 1
    maxi = 1  # nclass
    nbins = 1  # nclass
    score_thresh = 0.5
    _, predict = torch.max(output, 1)
    # predict = (F.sigmoid(output).detach().cpu().numpy() > score_thresh).astype('int64') # P
    predict = predict.cpu().numpy().astype('int64')
    target = target.cpu().numpy().astype('int64')
    intersection = predict * (predict == target) # TP

    num_sample = intersection.shape[0]
    area_inter_arr = np.zeros(num_sample)
    area_pred_arr = np.zeros(num_sample)
    area_lab_arr = np.zeros(num_sample)
    area_union_arr = np.zeros(num_sample)

    for b in range(num_sample):
        # areas of intersection and union
        area_inter, _ = np.histogram(intersection[b], bins=nbins, range=(mini, maxi))
        area_inter_arr[b] = area_inter

        area_pred, _ = np.histogram(predict[b], bins=nbins, range=(mini, maxi))
        area_pred_arr[b] = area_pred

        area_lab, _ = np.histogram(target[b], bins=nbins, range=(mini, maxi))
        area_lab_arr[b] = area_lab

        area_union = area_pred + area_lab - area_inter
        area_union_arr[b] = area_union

        assert (area_inter <= area_union).all()

    return area_inter_arr, area_union_arr
 #-----------------------------------------------yb 2024.03.14-----------------------------------------------------
class PD_FA():          # DNANet
    def __init__(self, nclass, bins):
        super(PD_FA, self).__init__()
        self.nclass = nclass
        self.bins = bins
        self.image_area_total = []
        self.image_area_match = []
        self.FA = 0
        self.PD = 0
        self.target= 0
    def update(self, preds, labels):

        _, preds = torch.max(preds, 1)
        predits  = np.array((preds > 0).cpu()).astype('int64')
        # predits  = np.reshape (predits,  (640,640))
        labelss = np.array((labels).cpu()).astype('int64') # P
        # labelss = np.reshape (labelss , (640,640))

        image = measure.label(predits, connectivity=2)
        coord_image = measure.regionprops(image)
        label = measure.label(labelss , connectivity=2)
        coord_label = measure.regionprops(label)

        self.target   += len(coord_label)
        self.image_area_total = []
        self.image_area_match = []
        self.distance_match   = []
        self.dismatch         = []

        for K in range(len(coord_image)):
            area_image = np.array(coord_image[K].area)
            self.image_area_total.append(area_image)

        for i in range(len(coord_label)):
            centroid_label = np.array(list(coord_label[i].centroid))
            for m in range(len(coord_image)):
                centroid_image = np.array(list(coord_image[m].centroid))
                distance = np.linalg.norm(centroid_image - centroid_label)
                area_image = np.array(coord_image[m].area)
                if distance < 3:
                    self.distance_match.append(distance)
                    self.image_area_match.append(area_image)

                    del coord_image[m]
                    break

        self.dismatch = [x for x in self.image_area_total if x not in self.image_area_match]
        self.FA +=np.sum(self.dismatch)
        self.PD +=len(self.distance_match)

    def get(self,img_num):

        Final_FA =  self.FA / ((640 * 640) * img_num)
        Final_PD =  self.PD /self.target

        return Final_FA,Final_PD


    def reset(self):
        self.FA  = np.zeros([self.bins+1])
        self.PD  = np.zeros([self.bins+1])

# class PD_FA():            #  只取阈值为0.5的值
#     def __init__(self, nclass, bins):
#         super(PD_FA, self).__init__()
#         self.nclass = nclass
#         self.bins = bins
#         self.FA = 0  # 只需要一个值，所以改为单一变量
#         self.PD = 0  # 只需要一个值，所以改为单一变量
#         self.target = 0  # 只需要一个值，所以改为单一变量

#     def update(self, preds, labels):
#         # 更新图像大小为 640x640
#         _, preds = torch.max(preds, 1)
#         preds = np.array(preds.cpu()).astype('int64').reshape(640, 640)
#         labels = np.array(labels.cpu()).astype('int64').reshape(640, 640)

#         # 设定 score_thresh = 127
#         # score_thresh = 127
#         predits = (preds > 0).astype('int64')
#         labelss = labels

#         image = measure.label(predits, connectivity=2)
#         coord_image = measure.regionprops(image)
#         label = measure.label(labelss, connectivity=2)
#         coord_label = measure.regionprops(label)

#         target += len(coord_label)
#         image_area_total = []
#         image_area_match = []
#         distance_match = []
#         dismatch = []

#         for K in range(len(coord_image)):
#             area_image = np.array(coord_image[K].area)
#             image_area_total.append(area_image)

#         for i in range(len(coord_label)):
#             centroid_label = np.array(list(coord_label[i].centroid))
#             for m in range(len(coord_image)):
#                 centroid_image = np.array(list(coord_image[m].centroid))
#                 distance = np.linalg.norm(centroid_image - centroid_label)
#                 area_image = np.array(coord_image[m].area)
#                 if distance < 3:
#                     distance_match.append(distance)
#                     image_area_match.append(area_image)

#                     del coord_image[m]
#                     break

#         dismatch = [x for x in image_area_total if x not in image_area_match]
#         FA +=np.sum(dismatch)
#         PD +=len(distance_match)

#     def get(self, img_num):
#         # 使用 640x640 计算标准化的假阳性率
#         Final_FA = FA / ((640 * 640) * img_num)
#         Final_PD = PD / target
#         return Final_FA, Final_PD

#     def reset(self):
#         self.FA = 0  # 重置为0
#         self.PD = 0  # 重置为0
#         self.target = 0  # 重置为0


# class PD_FA():
#     def __init__(self, nclass, bins):
#         super(PD_FA, self).__init__()
#         self.nclass = nclass
#         self.bins = bins
#         self.image_area_total = []
#         self.image_area_match = []
#         self.FA = np.zeros(self.bins+1)
#         self.PD = np.zeros(self.bins + 1)
#         self.target= np.zeros(self.bins + 1)
#     def update(self, preds, labels):
#         for iBin in range(self.bins+1):
#             for i in range(preds.shape[0]):      # 遍历每张图像   
#                 for c in range(preds.shape[1]):  # 遍历每个通道
#                     # print(preds.shape, i, c)
#                     predits = preds[i, c, :, :]  # 提取单张图像和单通道
                    
#                     score_thresh = iBin * (255/self.bins)
#                     predits  = np.array((predits > score_thresh).cpu()).astype('int64')

#                     predits  = np.reshape (predits,  (640,640))
#                     labelss = np.array((labels[i]).cpu()).astype('int64') # P
#                     labelss = np.reshape (labelss , (640,640))

#                     image = measure.label(predits, connectivity=2)
#                     coord_image = measure.regionprops(image)
#                     label = measure.label(labelss , connectivity=2)
#                     coord_label = measure.regionprops(label)

#                     self.target[iBin]    += len(coord_label)
#                     self.image_area_total = []
#                     self.image_area_match = []
#                     self.distance_match   = []
#                     self.dismatch         = []

#                     for K in range(len(coord_image)):
#                         area_image = np.array(coord_image[K].area)
#                         self.image_area_total.append(area_image)

#                     for j in range(len(coord_label)):
#                         centroid_label = np.array(list(coord_label[j].centroid))
#                         for m in range(len(coord_image)):
#                             centroid_image = np.array(list(coord_image[m].centroid))
#                             distance = np.linalg.norm(centroid_image - centroid_label)
#                             area_image = np.array(coord_image[m].area)
#                             if distance < 3:
#                                 self.distance_match.append(distance)
#                                 self.image_area_match.append(area_image)
#                                 del coord_image[m]
#                                 break
#             self.dismatch = [x for x in self.image_area_total if x not in self.image_area_match]
#             self.FA[iBin]+=np.sum(self.dismatch)
#             self.PD[iBin]+=len(self.distance_match)
#     def get(self,img_num):

#         Final_FA =  self.FA / ((640 * 640) * img_num)
#         Final_PD =  self.PD /self.target

#         return Final_FA,Final_PD

#     def reset(self):
#         self.FA  = np.zeros([self.bins+1])
#         self.PD  = np.zeros([self.bins+1])



# class PD_FA():
#     def __init__(self, nclass, bins):  #bin的意义实际上是确定ROC曲线上的threshold取多少个离散值
#         super(PD_FA, self).__init__()
#         self.nclass = nclass
#         self.bins = bins
#         self.image_area_total = []
#         self.image_area_match = []
#         self.match_index = []
#         self.FA = np.zeros(self.bins+1)
#         self.PD = np.zeros(self.bins + 1)
#         self.nIoU = np.zeros(self.bins + 1)
#         self.target= np.zeros(self.bins + 1)
#     def update(self, preds, labels):
#         labels = np.array((labels).cpu()).astype('int64')  # P

#         b,h,w = labels.shape
#         predit = torch.mean(preds, dim=1)  # [4,640,640]

#         labelss = labels
#         # labelss = np.reshape(labelss, (b*h, w))
#         label = measure.label(labelss, connectivity=2)
#         coord_label = measure.regionprops(label)
#         for iBin in range(self.bins+1):
#             score_thresh = (iBin ) / self.bins
#             predits = np.array((predit > score_thresh).cpu()).astype('int64')
#             # predits = np.reshape(predit, (b*h, w))


#             image = measure.label(predits, connectivity=2)
#             coord_image = measure.regionprops(image)


#             self.target[iBin]    += len(coord_label)
#             self.image_area_total = []
#             self.image_area_match = []
#             self.distance_match   = []
#             self.dismatch         = []
#             self.IoU = 0
#             for K in range(len(coord_image)):
#                 area_image = np.array(coord_image[K].area)
#                 self.image_area_total.append(area_image)

#             self.sum_match=0
#             self.match_index=[]
#             for m in range(len(coord_image)):
#                 centroid_image = np.array(list(coord_image[m].centroid))

#                 area_image = np.array(coord_image[m].area)

#                 for i in range(len(coord_label)):
#                     centroid_label = np.array(list(coord_label[i].centroid))
#                     distance = np.linalg.norm(centroid_image - centroid_label)
#                     if distance < 0.5*coord_label[i].equivalent_diameter:
#                         self.distance_match.append(distance)
#                         self.image_area_match.append(area_image)
#                         self.match_index.append(i)
#                         self.sum_match=self.sum_match+1

#                         intersection = np.sum(np.array(image==m+1)*np.array(label==i+1))
#                         label_sum = np.sum(np.array(coord_label[i].area))
#                         pred_sum = np.sum(area_image)
#                         self.IoU += intersection/(label_sum+pred_sum-intersection)
#                         # del coord_image[m]
#                         break

#             self.match_index= list(set(self.match_index))

#             self.dismatch = [x for x in self.image_area_total if x not in self.image_area_match]
#             self.FA[iBin]+=np.sum(self.dismatch)
#             self.PD[iBin]+=len(self.match_index)
#             self.nIoU[iBin]+=self.IoU


#     def get(self,img_num):

#         Final_FA =  self.FA / ((1024 * 1024) * img_num)
#         Final_PD =  self.PD /self.target
#         Final_nIoU = self.nIoU / self.target

#         return Final_FA,Final_PD,Final_nIoU


#     def reset(self):
#         self.FA  = np.zeros([self.bins+1])
#         self.PD  = np.zeros([self.bins+1])
#         self.nIoU= np.zeros([self.bins+1])
#         self.target = np.zeros(self.bins + 1)

#-----------------------------------------------yb 2024.08.21-----------------------------------------------------

def fitness(x):
    # Model fitness as a weighted combination of metrics
    w = [0.0, 0.0, 0.1, 0.9, 0.0, 0.0, 0.1, 0.9]
    return (x[:, :8] * w).sum(1)


def ap_per_class_box_and_mask(
        tp_m,
        tp_b,
        conf,
        pred_cls,
        target_cls,
        plot=False,
        save_dir=".",
        names=(),
):
    """
    Args:
        tp_b: tp of boxes.
        tp_m: tp of masks.
        other arguments see `func: ap_per_class`.
    """
    results_boxes = ap_per_class(tp_b,
                                 conf,
                                 pred_cls,
                                 target_cls,
                                 plot=plot,
                                 save_dir=save_dir,
                                 names=names,
                                 prefix="Box")[2:]
    results_masks = ap_per_class(tp_m,
                                 conf,
                                 pred_cls,
                                 target_cls,
                                 plot=plot,
                                 save_dir=save_dir,
                                 names=names,
                                 prefix="Mask")[2:]

    results = {
        "boxes": {
            "p": results_boxes[0],
            "r": results_boxes[1],
            "ap": results_boxes[3],
            "f1": results_boxes[2],
            "ap_class": results_boxes[4]},
        "masks": {
            "p": results_masks[0],
            "r": results_masks[1],
            "ap": results_masks[3],
            "f1": results_masks[2],
            "ap_class": results_masks[4]}}
    return results


class Metric:

    def __init__(self) -> None:
        self.p = []  # (nc, )
        self.r = []  # (nc, )
        self.f1 = []  # (nc, )
        self.all_ap = []  # (nc, 10)
        self.ap_class_index = []  # (nc, )

    @property
    def ap50(self):
        """AP@0.5 of all classes.
        Return:
            (nc, ) or [].
        """
        return self.all_ap[:, 0] if len(self.all_ap) else []

    @property
    def ap(self):
        """AP@0.5:0.95
        Return:
            (nc, ) or [].
        """
        return self.all_ap.mean(1) if len(self.all_ap) else []

    @property
    def mp(self):
        """mean precision of all classes.
        Return:
            float.
        """
        return self.p.mean() if len(self.p) else 0.0

    @property
    def mr(self):
        """mean recall of all classes.
        Return:
            float.
        """
        return self.r.mean() if len(self.r) else 0.0

    @property
    def map50(self):
        """Mean AP@0.5 of all classes.
        Return:
            float.
        """
        return self.all_ap[:, 0].mean() if len(self.all_ap) else 0.0

    @property
    def map(self):
        """Mean AP@0.5:0.95 of all classes.
        Return:
            float.
        """
        return self.all_ap.mean() if len(self.all_ap) else 0.0

    def mean_results(self):
        """Mean of results, return mp, mr, map50, map"""
        return (self.mp, self.mr, self.map50, self.map)

    def class_result(self, i):
        """class-aware result, return p[i], r[i], ap50[i], ap[i]"""
        return (self.p[i], self.r[i], self.ap50[i], self.ap[i])

    def get_maps(self, nc):
        maps = np.zeros(nc) + self.map
        for i, c in enumerate(self.ap_class_index):
            maps[c] = self.ap[i]
        return maps

    def update(self, results):
        """
        Args:
            results: tuple(p, r, ap, f1, ap_class)
        """
        p, r, all_ap, f1, ap_class_index = results
        self.p = p
        self.r = r
        self.all_ap = all_ap
        self.f1 = f1
        self.ap_class_index = ap_class_index


class Metrics:
    """Metric for boxes and masks."""

    def __init__(self) -> None:
        self.metric_box = Metric()
        self.metric_mask = Metric()

    def update(self, results):
        """
        Args:
            results: Dict{'boxes': Dict{}, 'masks': Dict{}}
        """
        self.metric_box.update(list(results["boxes"].values()))
        self.metric_mask.update(list(results["masks"].values()))

    def mean_results(self):
        return self.metric_box.mean_results() + self.metric_mask.mean_results()

    def class_result(self, i):
        return self.metric_box.class_result(i) + self.metric_mask.class_result(i)

    def get_maps(self, nc):
        return self.metric_box.get_maps(nc) + self.metric_mask.get_maps(nc)

    @property
    def ap_class_index(self):
        # boxes and masks have the same ap_class_index
        return self.metric_box.ap_class_index


KEYS = [
    "train/box_loss",
    "train/seg_loss",  # train loss
    "train/obj_loss",
    "train/cls_loss",
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP_0.5(B)",
    "metrics/mAP_0.5:0.95(B)",  # metrics
    "metrics/precision(M)",
    "metrics/recall(M)",
    "metrics/mAP_0.5(M)",
    "metrics/mAP_0.5:0.95(M)",  # metrics
    "val/box_loss",
    "val/seg_loss",  # val loss
    "val/obj_loss",
    "val/cls_loss",
    "x/lr0",
    "x/lr1",
    "x/lr2",]

BEST_KEYS = [
    "best/epoch",
    "best/precision(B)",
    "best/recall(B)",
    "best/mAP_0.5(B)",
    "best/mAP_0.5:0.95(B)",
    "best/precision(M)",
    "best/recall(M)",
    "best/mAP_0.5(M)",
    "best/mAP_0.5:0.95(M)",]
