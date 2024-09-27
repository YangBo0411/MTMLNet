import argparse
import math
import os
import random
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.optim import lr_scheduler
from tqdm import tqdm


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  

import val as validate  

from models.experimental import attempt_load
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.autobatch import check_train_batch_size
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.downloads import attempt_download, is_url
from utils.general import (LOGGER, TQDM_BAR_FORMAT, check_amp, check_dataset, check_file, check_img_size,
                           check_suffix, check_yaml, colorstr, get_latest_run, increment_path, init_seeds,
                           intersect_dicts, labels_to_class_weights, labels_to_image_weights, methods,
                           one_cycle, one_flat_cycle, print_args, print_mutation, strip_optimizer, yaml_save)
from utils.loggers import Loggers
from utils.loggers.comet.comet_utils import check_comet_resume
from utils.loss_tal import ComputeLoss, SegmentationLosses
from utils.metrics import fitness, fitness2
from utils.plots import plot_evolve
from utils.torch_utils import (EarlyStopping, ModelEMA, de_parallel, select_device, smart_DDP,
                               smart_optimizer, smart_resume, torch_distributed_zero_first)
import SegmentationDataset  
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
GIT_INFO = None


def train(hyp, opt, device, callbacks):  
    save_dir, epochs, batch_size, weights, single_cls, evolve, data, cfg, resume, noval, nosave, workers, freeze = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights, opt.single_cls, opt.evolve, opt.data, opt.cfg, \
        opt.resume, opt.noval, opt.nosave, opt.workers, opt.freeze
    callbacks.run('on_pretrain_routine_start')

    
    w = save_dir / 'weights'  
    (w.parent if evolve else w).mkdir(parents=True, exist_ok=True)  
    last, best = w / 'last.pt', w / 'best.pt'
    

    
    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  
    LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    hyp['anchor_t'] = 5.0
    opt.hyp = hyp.copy()  

    
    if not evolve:
        yaml_save(save_dir / 'hyp.yaml', hyp)
        yaml_save(save_dir / 'opt.yaml', vars(opt))

    
    data_dict = None
    if RANK in {-1, 0}:
        loggers = Loggers(save_dir, weights, opt, hyp, LOGGER)  

        
        for k in methods(loggers):
            callbacks.register_action(k, callback=getattr(loggers, k))

        
        data_dict = loggers.remote_dataset
        if resume:  
            weights, epochs, hyp, batch_size = opt.weights, opt.epochs, opt.hyp, opt.batch_size

    
    plots = not evolve and not opt.noplots  
    cuda = device.type != 'cpu'
    init_seeds(opt.seed + 1 + RANK, deterministic=True)
    with torch_distributed_zero_first(LOCAL_RANK):
        data_dict = data_dict or check_dataset(data)  
    train_path, val_path = data_dict['train'], data_dict['val']
    segtrain_path, segval_path = data_dict['segtrain'], data_dict['segval']  
    nc = 1 if single_cls else int(data_dict['nc'])  
    names = {0: 'item'} if single_cls and len(data_dict['names']) != 1 else data_dict['names']  
    
    is_coco = isinstance(val_path, str) and val_path.endswith('val2017.txt')  

    
    check_suffix(weights, '.pt')  
    pretrained = weights.endswith('.pt')
    if pretrained:
        with torch_distributed_zero_first(LOCAL_RANK):
            weights = attempt_download(weights)  
        ckpt = torch.load(weights, map_location='cpu')  
        model = Model(cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  
        exclude = ['anchor'] if (cfg or hyp.get('anchors')) and not resume else []  
        csd = ckpt['model'].float().state_dict()  
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  
        model.load_state_dict(csd, strict=False)  
        LOGGER.info(f'Transferred {len(csd)}/{len(model.state_dict())} items from {weights}')  
    else:
        model = Model(cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  
    
    amp = False
    
    freeze = [f'model.{x}.' for x in (freeze if len(freeze) > 1 else range(freeze[0]))]  
    for k, v in model.named_parameters():
        
        
        if any(x in k for x in freeze):
            LOGGER.info(f'freezing {k}')
            v.requires_grad = False

    
    gs = max(int(model.stride.max()), 32)  
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)  

    
    if RANK == -1 and batch_size == -1:  
        batch_size = check_train_batch_size(model, imgsz, amp)
        loggers.on_params_update({"batch_size": batch_size})

    
    nbs = 64  
    accumulate = max(round(nbs / batch_size), 1)  
    hyp['weight_decay'] *= batch_size * accumulate / nbs  
    optimizer = smart_optimizer(model, opt.optimizer, hyp['lr0'], hyp['momentum'], hyp['weight_decay'])

    
    if opt.cos_lr:
        lf = one_cycle(1, hyp['lrf'], epochs)  
    elif opt.flat_cos_lr:
        lf = one_flat_cycle(1, hyp['lrf'], epochs)  
    elif opt.fixed_lr:
        lf = lambda x: 1.0
    else:
        lf = lambda x: (1 - x / epochs) * (1.0 - hyp['lrf']) + hyp['lrf']  

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    

    
    ema = ModelEMA(model) if RANK in {-1, 0} else None

    
    best_fitness, start_epoch = 0.0, 0
    if pretrained:
        if resume:
            best_fitness, start_epoch, epochs = smart_resume(ckpt, optimizer, ema, weights, epochs, resume)
        del ckpt, csd

    
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        LOGGER.warning('WARNING ⚠️ DP not recommended, use torch.distributed.run for best DDP Multi-GPU results.')
        model = torch.nn.DataParallel(model)

    
    if opt.sync_bn and cuda and RANK != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        LOGGER.info('Using SyncBatchNorm()')

    
    train_loader, dataset = create_dataloader(train_path,
                                              imgsz,
                                              batch_size // WORLD_SIZE,
                                              gs,
                                              single_cls,
                                              hyp=hyp,
                                              augment=True,
                                              cache=None if opt.cache == 'val' else opt.cache,
                                              rect=opt.rect,
                                              rank=LOCAL_RANK,
                                              workers=workers,
                                              image_weights=opt.image_weights,
                                              close_mosaic=opt.close_mosaic != 0,
                                              quad=opt.quad,
                                              prefix=colorstr('train: '),
                                              shuffle=True,
                                              min_items=opt.min_items)
    labels = np.concatenate(dataset.labels, 0)
    mlc = int(labels[:, 0].max())  
    assert mlc < nc, f'Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}'

    
    if RANK in {-1, 0}:
        val_loader = create_dataloader(val_path,
                                       imgsz,
                                       batch_size // WORLD_SIZE * 2,
                                       gs,
                                       single_cls,
                                       hyp=hyp,
                                       cache=None if noval else opt.cache,
                                       rect=True,
                                       rank=-1,
                                       workers=workers * 2,
                                       pad=0.5,
                                       prefix=colorstr('val: '))[0]

        if not resume:
            
            
            model.half().float()  

        callbacks.run('on_pretrain_routine_end', labels, names)
        
        
        seg_valloader = SegmentationDataset.get_custom_loader(root=segval_path, batch_size=4,  
                                                              split="val", mode="val",  
                                                              base_size=imgsz,  
                                                              
                                                              workers=2,
                                                              pin=True)  
    seg_trainloader = SegmentationDataset.get_custom_loader(root=segtrain_path,
                                                            split="train", mode="train",
                                                            base_size=imgsz,
                                                            
                                                            batch_size=batch_size,
                                                            workers=opt.workers, pin=True)

    segnb = len(seg_trainloader)
    



    
    if cuda and RANK != -1:
        model = smart_DDP(model)

    
    nl = de_parallel(model).model[-1].nl  
    
    
    
    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = nc  
    model.hyp = hyp  
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  
    model.names = names

    
    t0 = time.time()
    nb = len(train_loader)  
    nw = max(round(hyp['warmup_epochs'] * nb), 100)  
    
    last_opt_step = -1
    maps = np.zeros(nc)  
    results = (0, 0, 0, 0, 0, 0, 0)  
    scheduler.last_epoch = start_epoch - 1  
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    stopper, stop = EarlyStopping(patience=opt.patience), False
    compute_loss = ComputeLoss(model)  
    
    compute_seg_loss = SegmentationLosses().cuda()
    detgain, seggain = 0.6, 0.35  
    callbacks.run('on_train_start')
    LOGGER.info(f'Image sizes {imgsz} train, {imgsz} val\n'
                f'Using {train_loader.num_workers * WORLD_SIZE} dataloader workers\n'
                f"Logging results to {colorstr('bold', save_dir)}\n"
                f'Starting training for {epochs} epochs...')
    for epoch in range(start_epoch, epochs):  
        callbacks.run('on_train_epoch_start')
        mIoU = 0                                    
        print(f'accumulate: {accumulate}')          
        model.train()

        
        if opt.image_weights:
            cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  
            iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  
            dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  
        if epoch == (epochs - opt.close_mosaic):
            LOGGER.info("Closing dataloader mosaic")
            dataset.mosaic = False

        
        
        

        mloss = torch.zeros(3, device=device)  
        msegloss = torch.zeros(1, device=device)  
        if RANK != -1:
            train_loader.sampler.set_epoch(epoch)
        pbar = enumerate(train_loader)
        segpbar = enumerate(seg_trainloader)  
        
        LOGGER.info(('\n' + '%11s' * 9) % ('Epoch', 'GPU_mem', 'box_loss', 'cls_loss', 'dfl_loss', 'total', 'seg', 'Instances', 'Size'))  
        if RANK in {-1, 0}:
            pbar = tqdm(pbar, total=nb, bar_format=TQDM_BAR_FORMAT)  
            segpbar = tqdm(segpbar, total=min(nb, segnb), bar_format=TQDM_BAR_FORMAT)  
        optimizer.zero_grad()
        
        for det_batch, seg_batch in zip(pbar, segpbar):  
            callbacks.run('on_train_batch_start')
            i, (imgs, targets, paths, _) = det_batch  
            _, (segimgs, segtargets) = seg_batch  

            
            
            

            ni = i + nb * epoch  
            imgs = imgs.to(device, non_blocking=True).float() / 255  
            segimgs = segimgs.to(device, non_blocking=True)  

            
            if ni <= nw:
                xi = [0, nw]  
                
                
                accumulate = max(1, np.interp(ni, xi, [1, math.floor(nbs / batch_size)]).round())  
                for j, x in enumerate(optimizer.param_groups):
                    
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 0 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            
            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  
                sf = sz / max(imgs.shape[2:])  
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  
                    imgs = nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            
            with torch.cuda.amp.autocast(amp):
                pred = model(imgs)  
                
                loss, loss_items = compute_loss(pred[0], targets.to(device))  
                if RANK != -1:
                    loss *= WORLD_SIZE  
                if opt.quad:
                    loss *= 4.
                loss *= detgain  
            
            torch.use_deterministic_algorithms(False)  
            scaler.scale(loss).backward()

            
            with torch.cuda.amp.autocast(amp):
                seg_pred = model(segimgs)
                
                
                
                
                segloss = compute_seg_loss(seg_pred[1], segtargets.to(device),3, epoch) * batch_size     
                if RANK != -1:
                    segloss *= WORLD_SIZE  
                if opt.quad:
                    segloss *= 4.
                segloss *= seggain
            torch.use_deterministic_algorithms(False)  
            scaler.scale(segloss).backward()
            
            

            
            if ni - last_opt_step >= accumulate:
                scaler.unscale_(optimizer)  
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  
                scaler.step(optimizer)  
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)
                last_opt_step = ni

            
            if RANK in {-1, 0}:
                mloss = (mloss * i + loss_items) / (i + 1)  
                msegloss = (msegloss * i + segloss.detach() / batch_size) / (i + 1)  
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  
                
                pbar.set_description(('%11s' * 2 + '%11.4g' * 6) % (f'{epoch}/{epochs - 1}', mem, *msegloss, *mloss, targets.shape[0], imgs.shape[-1])) 
                callbacks.run('on_train_batch_end', model, ni, imgs, targets, paths, list(mloss))
                if callbacks.stop_training:
                    return
            

        
        lr = [x['lr'] for x in optimizer.param_groups]  
        scheduler.step()

        if RANK in {-1, 0}:
            
            callbacks.run('on_train_epoch_end', epoch=epoch)
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])

            
            
            mIoU = validate.seg_validation(model=ema.ema, valloader=seg_valloader, device=device, n_segcls=1,
                                               half_precision=True)  
            
            

            final_epoch = (epoch + 1 == epochs) or stopper.possible_stop
            if not noval or final_epoch:  
                results, maps, _ = validate.run(data_dict,
                                                batch_size=batch_size // WORLD_SIZE * 2,
                                                imgsz=imgsz,
                                                half=amp,
                                                model=ema.ema,
                                                single_cls=single_cls,
                                                dataloader=val_loader,
                                                save_dir=save_dir,
                                                plots=False,
                                                callbacks=callbacks,
                                                compute_loss=compute_loss)

            
            
            fi = fitness2(np.array(results).reshape(1, -1), mIoU)  
            stop = stopper(epoch=epoch, fitness=fi)  
            if fi > best_fitness:
                best_fitness = fi

            
            log_vals = list(mloss) + list(results) + lr + list(msegloss) + [mIoU]  
            callbacks.run('on_fit_epoch_end', log_vals, epoch, best_fitness, fi)

            
            if (not nosave) or (final_epoch and not evolve):  
                ckpt = {
                    'epoch': epoch,
                    'best_fitness': best_fitness,
                    'model': deepcopy(de_parallel(model)).half(),
                    'ema': deepcopy(ema.ema).half(),
                    'updates': ema.updates,
                    'optimizer': optimizer.state_dict(),
                    'opt': vars(opt),
                    'git': GIT_INFO,  
                    'date': datetime.now().isoformat()}

                
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                if opt.save_period > 0 and epoch % opt.save_period == 0:
                    torch.save(ckpt, w / f'epoch{epoch}.pt')
                del ckpt
                callbacks.run('on_model_save', last, epoch, final_epoch, best_fitness, fi)

        
        if RANK != -1:  
            broadcast_list = [stop if RANK == 0 else None]
            dist.broadcast_object_list(broadcast_list, 0)  
            if RANK != 0:
                stop = broadcast_list[0]
        if stop:
            break  

        
    
    if RANK in {-1, 0}:
        LOGGER.info(f'\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')
        for f in last, best:
            if f.exists():
                
                
                
                
                if f is best:
                    LOGGER.info(f'\nValidating {f}...')
                    results, _, _ = validate.run(
                        data_dict,
                        batch_size=batch_size // WORLD_SIZE * 2,
                        imgsz=imgsz,
                        model=attempt_load(f, device).half(),
                        single_cls=single_cls,
                        dataloader=val_loader,
                        save_dir=save_dir,
                        save_json=is_coco,
                        verbose=True,
                        plots=plots,
                        callbacks=callbacks,
                        compute_loss=compute_loss)  
                    if is_coco:
                        callbacks.run('on_fit_epoch_end', list(mloss) + list(results) + lr, epoch, best_fitness, fi)

        callbacks.run('on_train_end', last, best, epoch, results)

    torch.cuda.empty_cache()

    
    
    return results

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    
    
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='models/detect/P234-MEA-HPDown-HFF.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default=ROOT / 'data/custom.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch-high.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=300, help='total training epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=512, help='train, val image size (pixels)')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable AutoAnchor')
    parser.add_argument('--noplots', action='store_true', help='save no plot files')
    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='image --cache ram/disk')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='1', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW', 'LION'], default='SGD', help='optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', default=ROOT / 'runs/train', help='save to project/name')
    parser.add_argument('--name', default='1', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--cos-lr', action='store_true', help='cosine LR scheduler')
    parser.add_argument('--flat-cos-lr', action='store_true', help='flat cosine LR scheduler')
    parser.add_argument('--fixed-lr', action='store_true', help='fixed LR scheduler')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--patience', type=int, default=0, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone=10, first3=0 1 2')
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--seed', type=int, default=0, help='Global training seed')
    parser.add_argument('--local_rank', type=int, default=-1, help='Automatic DDP Multi-GPU argument, do not modify')
    parser.add_argument('--min-items', type=int, default=0, help='Experimental')
    parser.add_argument('--close-mosaic', type=int, default=0, help='Experimental')

    
    parser.add_argument('--entity', default=None, help='Entity')
    parser.add_argument('--upload_dataset', nargs='?', const=True, default=False, help='Upload data, "val" option')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval')
    parser.add_argument('--artifact_alias', type=str, default='latest', help='Version of dataset artifact to use')

    return parser.parse_known_args()[0] if known else parser.parse_args()


def main(opt, callbacks=Callbacks()):
    
    if RANK in {-1, 0}:
        print_args(vars(opt))

    
    if opt.resume and not check_comet_resume(opt) and not opt.evolve:
        last = Path(check_file(opt.resume) if isinstance(opt.resume, str) else get_latest_run())
        opt_yaml = last.parent.parent / 'opt.yaml'  
        opt_data = opt.data  
        if opt_yaml.is_file():
            with open(opt_yaml, errors='ignore') as f:
                d = yaml.safe_load(f)
        else:
            d = torch.load(last, map_location='cpu')['opt']
        opt = argparse.Namespace(**d)  
        opt.cfg, opt.weights, opt.resume = '', str(last), True  
        if is_url(opt_data):
            opt.data = check_file(opt_data)  
    else:
        opt.data, opt.cfg, opt.hyp, opt.weights, opt.project = \
            check_file(opt.data), check_yaml(opt.cfg), check_yaml(opt.hyp), str(opt.weights), str(opt.project)  
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        if opt.evolve:
            if opt.project == str(ROOT / 'runs/train'):  
                opt.project = str(ROOT / 'runs/evolve')
            opt.exist_ok, opt.resume = opt.resume, False  
        if opt.name == 'cfg':
            opt.name = Path(opt.cfg).stem  
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

    
    device = select_device(opt.device, batch_size=opt.batch_size)
    if LOCAL_RANK != -1:
        msg = 'is not compatible with YOLO Multi-GPU DDP training'
        assert not opt.image_weights, f'--image-weights {msg}'
        assert not opt.evolve, f'--evolve {msg}'
        assert opt.batch_size != -1, f'AutoBatch with --batch-size -1 {msg}, please pass a valid --batch-size'
        assert opt.batch_size % WORLD_SIZE == 0, f'--batch-size {opt.batch_size} must be multiple of WORLD_SIZE'
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")

    
    if not opt.evolve:
        train(opt.hyp, opt, device, callbacks)

    
    else:
        
        meta = {
            'lr0': (1, 1e-5, 1e-1),  
            'lrf': (1, 0.01, 1.0),  
            'momentum': (0.3, 0.6, 0.98),  
            'weight_decay': (1, 0.0, 0.001),  
            'warmup_epochs': (1, 0.0, 5.0),  
            'warmup_momentum': (1, 0.0, 0.95),  
            'warmup_bias_lr': (1, 0.0, 0.2),  
            'box': (1, 0.02, 0.2),  
            'cls': (1, 0.2, 4.0),  
            'cls_pw': (1, 0.5, 2.0),  
            'obj': (1, 0.2, 4.0),  
            'obj_pw': (1, 0.5, 2.0),  
            'iou_t': (0, 0.1, 0.7),  
            'anchor_t': (1, 2.0, 8.0),  
            'anchors': (2, 2.0, 10.0),  
            'fl_gamma': (0, 0.0, 2.0),  
            'hsv_h': (1, 0.0, 0.1),  
            'hsv_s': (1, 0.0, 0.9),  
            'hsv_v': (1, 0.0, 0.9),  
            'degrees': (1, 0.0, 45.0),  
            'translate': (1, 0.0, 0.9),  
            'scale': (1, 0.0, 0.9),  
            'shear': (1, 0.0, 10.0),  
            'perspective': (0, 0.0, 0.001),  
            'flipud': (1, 0.0, 1.0),  
            'fliplr': (0, 0.0, 1.0),  
            'mosaic': (1, 0.0, 1.0),  
            'mixup': (1, 0.0, 1.0),  
            'copy_paste': (1, 0.0, 1.0)}  

        with open(opt.hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  
            if 'anchors' not in hyp:  
                hyp['anchors'] = 3
        if opt.noautoanchor:
            del hyp['anchors'], meta['anchors']
        opt.noval, opt.nosave, save_dir = True, True, Path(opt.save_dir)  
        
        evolve_yaml, evolve_csv = save_dir / 'hyp_evolve.yaml', save_dir / 'evolve.csv'
        if opt.bucket:
            os.system(f'gsutil cp gs://{opt.bucket}/evolve.csv {evolve_csv}')  

        for _ in range(opt.evolve):  
            if evolve_csv.exists():  
                
                parent = 'single'  
                x = np.loadtxt(evolve_csv, ndmin=2, delimiter=',', skiprows=1)
                n = min(5, len(x))  
                x = x[np.argsort(-fitness(x))][:n]  
                w = fitness(x) - fitness(x).min() + 1E-6  
                if parent == 'single' or len(x) == 1:
                    
                    x = x[random.choices(range(n), weights=w)[0]]  
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  

                
                mp, s = 0.8, 0.2  
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([meta[k][0] for k in hyp.keys()])  
                ng = len(meta)
                v = np.ones(ng)
                while all(v == 1):  
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(hyp.keys()):  
                    hyp[k] = float(x[i + 7] * v[i])  

            
            for k, v in meta.items():
                hyp[k] = max(hyp[k], v[1])  
                hyp[k] = min(hyp[k], v[2])  
                hyp[k] = round(hyp[k], 5)  

            
            
            results = train(hyp.copy(), opt, device, callbacks)     
            callbacks = Callbacks()
            
            
            
            keys = ('metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95', 'val/box_loss',
                    'val/obj_loss', 'val/cls_loss')             
            
            print_mutation(keys, results, hyp.copy(), save_dir, opt.bucket)  

        
        plot_evolve(evolve_csv)
        LOGGER.info(f'Hyperparameter evolution finished {opt.evolve} generations\n'
                    f"Results saved to {colorstr('bold', save_dir)}\n"
                    f'Usage example: $ python train.py --hyp {evolve_yaml}')


def run(**kwargs):
    
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)
    return opt


if __name__ == "__main__":

    opt = parse_opt()
    main(opt)
