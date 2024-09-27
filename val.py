import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  

from models.experimental import attempt_load        
from models.common import DetectMultiBackend
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.general import (LOGGER, TQDM_BAR_FORMAT, Profile, check_dataset, check_img_size, check_requirements,
                           check_yaml, coco80_to_coco91_class, colorstr, increment_path, non_max_suppression,
                           print_args, scale_boxes, xywh2xyxy, xyxy2xywh)
from utils.metrics import ConfusionMatrix, ap_per_class, box_iou
from utils.segment.metrics import batch_pix_accuracy, batch_intersection_union,batch_intersection_union_miou, PD_FA,ROCMetric, cal_tp_pos_fp_neg        
from utils.plots import output_to_target, plot_images, plot_val_study
from utils.torch_utils import select_device, smart_inference_mode

import SegmentationDataset                    
import torch.nn.functional as F               

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'    


def seg_validation(model, n_segcls, valloader, device, half_precision=True):
    
    
    pd_fa_metric = PD_FA(nclass=1, bins=10)
    ROC_metric   = ROCMetric(1, 10)

    
    
    def eval_batch(model, image, target, half):
       

        outputs = model(image)
        
        pred = outputs[1]  
        target = target.to(device, non_blocking=True)
        pred = F.interpolate(pred, (target.shape[1], target.shape[2]), mode='bilinear', align_corners=True)  

        correct, labeled = batch_pix_accuracy(pred.data, target)
        inter, union = batch_intersection_union(pred.data, target, n_segcls)

        return correct, labeled, inter, union

    
    half = device.type != 'cpu' and half_precision  
    if half:
        model.half()
    model.eval()
    
    total_inter, total_union, total_correct, total_label = 0, 0, 0, 0

    tbar = tqdm(valloader, desc='\r')
    for i, (image, target) in enumerate(tbar):
        image = image.to(device, non_blocking=True)
        image = image.half() if half else image.float()
        with torch.no_grad():
            correct, labeled, inter, union= eval_batch(model, image, target, half)


        total_correct += correct
        total_label += labeled
        total_inter += inter
        total_union += union
        pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
        mIoU = (1.0 * total_inter / (np.spacing(1) + total_union)).mean() 
        
        tbar.set_description('mIoU: %.3f' % (mIoU))

        outputs = model(image)
        
        pred = outputs[1]  
        target = target.to(device, non_blocking=True)
        pred = F.interpolate(pred, (target.shape[1], target.shape[2]), mode='bilinear', align_corners=True)  

        pd_fa_metric.update(pred.data, target)
        ROC_metric.update(pred.data, target)
        t, f, r, p = ROC_metric.get()             
    
    img_num = len(valloader)
    

    np.savetxt('metric_results/recall.csv', r,delimiter=',')  
    np.savetxt('metric_results/prec.csv', p,delimiter=',')  
    np.savetxt('metric_results/tp.csv', t,delimiter=',')  
    np.savetxt('metric_results/fp.csv', f,delimiter=',')  

    return mIoU
def segtest(weights, root="data/citys", batch_size=1, half_precision=True, n_segcls=19, base_size=640):  
    device = select_device(opt.device, batch_size=batch_size)
    model = attempt_load(weights, device=device)  
    testvalloader = SegmentationDataset.get_custom_loader(root, batch_size=batch_size, split="test", mode="val", workers=4, base_size=base_size)
    
    seg_validation(model, n_segcls, testvalloader, device, half_precision)




def save_one_txt(predn, save_conf, shape, file):
    
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  
        with open(file, 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')


def save_one_json(predn, jdict, path, class_map):
    
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])  
    box[:, :2] -= box[:, 2:] / 2  
    for p, b in zip(predn.tolist(), box.tolist()):
        jdict.append({
            'image_id': image_id,
            'category_id': class_map[int(p[5])],
            'bbox': [round(x, 3) for x in b],
            'score': round(p[4], 5)})


def process_batch(detections, labels, iouv):
    """
    Return correct prediction matrix
    Arguments:
        detections (array[N, 6]), x1, y1, x2, y2, conf, class
        labels (array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (array[N, 10]), for 10 IoU levels
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)  
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)


@smart_inference_mode()
def run(
        data,
        weights=None,  
        segdata=None,  
        batch_size=32,  
        base_size=320,  
        imgsz=640,  
        conf_thres=0.001,  
        iou_thres=0.7,  
        max_det=300,  
        task='val',  
        device='',  
        workers=8,  
        single_cls=False,  
        augment=False,  
        verbose=False,  
        save_txt=False,  
        save_hybrid=False,  
        save_conf=False,  
        save_json=False,  
        project=ROOT / 'runs/val',  
        name='exp',  
        exist_ok=False,  
        half=True,  
        dnn=False,  
        min_items=0,  
        model=None,
        dataloader=None,
        save_dir=Path(''),
        plots=True,
        callbacks=Callbacks(),
        compute_loss=None,
):
    
    training = model is not None
    if training:  
        device, pt, jit, engine = next(model.parameters()).device, True, False, False  
        half &= device.type != 'cpu'  
        model.half() if half else model.float()
    else:  
        device = select_device(device, batch_size=batch_size)

        
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  

        
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
        imgsz = check_img_size(imgsz, s=stride)  
        half = model.fp16  
        if engine:
            batch_size = model.batch_size
        else:
            device = model.device
            if not (pt or jit):
                batch_size = 1  
                LOGGER.info(f'Forcing --batch-size 1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models')

        
        data = check_dataset(data)  

    
    model.eval()
    cuda = device.type != 'cpu'
    
    is_coco = isinstance(data.get('val'), str) and data['val'].endswith(f'val2017.txt')  
    nc = 1 if single_cls else int(data['nc'])  
    iouv = torch.linspace(0.5, 0.95, 10, device=device)  
    niou = iouv.numel()

    
    if not training:
        if pt and not single_cls:  
            ncm = model.model.nc
            assert ncm == nc, f'{weights} ({ncm} classes) trained on different --data than what you passed ({nc} ' \
                              f'classes). Pass correct combination of --weights and --data that are trained together.'
        model.warmup(imgsz=(1 if pt else batch_size, 3, imgsz, imgsz))  
        pad, rect = (0.0, False) if task == 'speed' else (0.5, pt)  
        task = task if task in ('train', 'val', 'test') else 'val'  
        dataloader = create_dataloader(data[task],
                                       imgsz,
                                       batch_size,
                                       stride,
                                       single_cls,
                                       pad=pad,
                                       rect=rect,
                                       workers=workers,
                                       min_items=opt.min_items,
                                       prefix=colorstr(f'{task}: '))[0]

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = model.names if hasattr(model, 'names') else model.module.names  
    if isinstance(names, (list, tuple)):  
        names = dict(enumerate(names))
    class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
    s = ('%22s' + '%11s' * 6) % ('Class', 'Images', 'Instances', 'P', 'R', 'mAP50', 'mAP50-95')
    tp, fp, p, r, f1, mp, mr, map50, ap50, map = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    dt = Profile(), Profile(), Profile()  
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    callbacks.run('on_val_start')
    pbar = tqdm(dataloader, desc=s, bar_format=TQDM_BAR_FORMAT)  
    for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
        callbacks.run('on_val_batch_start')
        with dt[0]:
            if cuda:
                im = im.to(device, non_blocking=True)
                targets = targets.to(device)
            im = im.half() if half else im.float()  
            im /= 255  
            nb, _, height, width = im.shape  

        
        with dt[1]:
            
            preds, train_out = model(im)[0] if compute_loss else (model(im, augment=augment)[0], None)  
        
        if compute_loss:
            loss += compute_loss(train_out, targets)[1]  

        
        targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)  
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  
        with dt[2]:
            preds = non_max_suppression(preds,
                                        conf_thres,
                                        iou_thres,
                                        labels=lb,
                                        multi_label=True,
                                        agnostic=single_cls,
                                        max_det=max_det)

        
        for si, pred in enumerate(preds):
            labels = targets[targets[:, 0] == si, 1:]
            nl, npr = labels.shape[0], pred.shape[0]  
            path, shape = Path(paths[si]), shapes[si][0]
            correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  
            seen += 1

            if npr == 0:
                if nl:
                    stats.append((correct, *torch.zeros((2, 0), device=device), labels[:, 0]))
                    if plots:
                        confusion_matrix.process_batch(detections=None, labels=labels[:, 0])
                continue

            
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            scale_boxes(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  

            
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  
                scale_boxes(im[si].shape[1:], tbox, shape, shapes[si][1])  
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  
                correct = process_batch(predn, labelsn, iouv)
                if plots:
                    confusion_matrix.process_batch(predn, labelsn)
            stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  

            
            if save_txt:
                save_one_txt(predn, save_conf, shape, file=save_dir / 'labels' / f'{path.stem}.txt')
            if save_json:
                save_one_json(predn, jdict, path, class_map)  
            callbacks.run('on_val_image_end', pred, predn, path, names, im[si])

        
        if plots and batch_i < 3:
            plot_images(im, targets, paths, save_dir / f'val_batch{batch_i}_labels.jpg', names)  
            plot_images(im, output_to_target(preds), paths, save_dir / f'val_batch{batch_i}_pred.jpg', names)  

        callbacks.run('on_val_batch_end', batch_i, im, targets, paths, shapes, preds)

    
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    nt = np.bincount(stats[3].astype(int), minlength=nc)  

    
    pf = '%22s' + '%11i' * 2 + '%11.3g' * 4  
    LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map))
    if nt.sum() == 0:
        LOGGER.warning(f'WARNING ⚠️ no labels found in {task} set, can not compute metrics without labels')

    
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    
    t = tuple(x.t / seen * 1E3 for x in dt)  
    if not training:
        shape = (batch_size, 3, imgsz, imgsz)
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)

    
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        callbacks.run('on_val_end', nt, tp, fp, p, r, f1, ap, ap50, ap_class, confusion_matrix)

    
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  
        anno_json = str(Path(data.get('path', '../coco')) / 'annotations/instances_val2017.json')  
        pred_json = str(save_dir / f"{w}_predictions.json")  
        LOGGER.info(f'\nEvaluating pycocotools mAP... saving {pred_json}...')
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:  
            check_requirements('pycocotools')
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  
            pred = anno.loadRes(pred_json)  
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.im_files]  
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  
        except Exception as e:
            LOGGER.info(f'pycocotools unable to run: {e}')

    
    model.float()  
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'data/custom.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'runs/train/15/weights/best.pt', help='model path(s)')
    parser.add_argument('--segdata', type=str, default='/home/wuren123/yb/multi task/multi task/data/customdata', help='root path of segmentation data')  
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--base-size', type=int, default=640, help='long side of segtest image you want to input network')  
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=300, help='maximum detections per image')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='1', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a COCO-JSON results file')
    parser.add_argument('--project', default=ROOT / 'runs/val', help='save to project/name')
    parser.add_argument('--name', default='1', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--min-items', type=int, default=0, help='Experimental')
    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.save_txt |= opt.save_hybrid
    print_args(vars(opt))
    return opt


def main(opt):
    

    if opt.task in ('train', 'val', 'test'):  
        if opt.conf_thres > 0.001:  
            LOGGER.info(f'WARNING ⚠️ confidence threshold {opt.conf_thres} > 0.001 produces invalid results')
        if opt.save_hybrid:
            LOGGER.info('WARNING ⚠️ --save-hybrid will return high mAP from hybrid labels, not from predictions alone')
        run(**vars(opt))

    else:
        weights = opt.weights if isinstance(opt.weights, list) else [opt.weights]
        opt.half = torch.cuda.is_available() and opt.device != 'cpu'  
        if opt.task == 'speed':  
            
            opt.conf_thres, opt.iou_thres, opt.save_json = 0.25, 0.45, False
            for opt.weights in weights:
                run(**vars(opt), plots=False)

        elif opt.task == 'study':  
            
            for opt.weights in weights:
                f = f'study_{Path(opt.data).stem}_{Path(opt.weights).stem}.txt'  
                x, y = list(range(256, 1536 + 128, 128)), []  
                for opt.imgsz in x:  
                    LOGGER.info(f'\nRunning {f} --imgsz {opt.imgsz}...')
                    r, _, t = run(**vars(opt), plots=False)
                    y.append(r + t)  
                np.savetxt(f, y, fmt='%10.4g')  
            os.system('zip -r study.zip study_*.txt')
            plot_val_study(x=x)  


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
    segtest(root=opt.segdata, weights=opt.weights, batch_size=int(opt.batch_size / 8), n_segcls=1, base_size=opt.base_size)  