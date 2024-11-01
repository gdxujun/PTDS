import argparse
import os

import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import models
import utils

from torchvision import transforms
from mmcv.runner import load_checkpoint
import cv2
import numpy as np
from PIL import Image
from medpy.metric import binary
import time


def batched_predict(model, inp, coord, bsize):
    with torch.no_grad():
        model.gen_feat(inp)
        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = model.query_rgb(coord[:, ql: qr, :])
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=1)
    return pred, preds


def tensor2PIL(tensor):
    toPIL = transforms.ToPILImage()
    return toPIL(tensor)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_dice(seg, gt):
    return binary.dc(seg, gt)

def compute_iou(s, g):
    intersection = np.logical_and(s, g).sum()
    union = np.logical_or(s, g).sum()
    return np.nanmean(intersection / (union+ 1e-10)) 

def eval_dice(label_base,pred_base,thres=0.5):
    # Note=open('/data/models/qxf/mic/snr/in_snr_decoder/eval_fig/preds_everyone.txt',mode='w')
    dices,ious,res = [],[],[]
    label_base = label_base
    pred_base = pred_base
    # if 'highlight' in args.test_seg_dir:
    #     seg_type = '_highlight.png'
    # elif 'polyp' in args.test_seg_dir:
    seg_type = '_polyp.png'
    for path in tqdm(sorted(os.listdir(label_base)), desc="eval:"):
        label = os.path.join(label_base, path)
        pred = os.path.join(pred_base, path.split('.')[0]+seg_type)

        image2 = Image.open(pred)
        seg = np.array(image2)
        image1 = Image.open(label).convert('L')
        image1 = image1.resize((seg.shape[1],seg.shape[0]), Image.NEAREST)
        gt = np.array(image1)
        
        seg[seg>=255*thres] = 255
        seg[seg<255*thres] = 0

        Dice = compute_dice(seg, gt)
        iou = compute_iou(seg, gt)
        dices.append(Dice)
        ious.append(iou)
        # Note.write("\n"+path[4:]+", Dice: " + str("%.4f" %(Dice)) + ', ' + "IoU: " + str("%.4f" %(iou)))
    # Note.close()

    return ["%.4f" %(sum(dices)/len(dices)),"%.4f" %(sum(ious)/len(ious))]


def eval_psnr(loader, model, data_norm=None, eval_type=None, eval_bsize=None,
              verbose=False):
    model.eval()
    '''
    if data_norm is None:
        data_norm = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }
    '''
    save_dir = '/root/autodl-tmp/method/add_2bolck/pred'
    
    t0 = time.time()
    fps_save, fps_list, frame_num = [], [], 0 # mean fps
    frame_num=0
    for batch in tqdm(loader, leave=False, desc='val'):
        # frame_num=frame_num+1
        '''
        for k, v in batch.items():
            batch[k] = v.cuda()
        '''
        inp = batch['inp'].cuda()
        gt = batch['gt'].cuda()
        filename = batch['filename']
      
        with torch.no_grad():
            t1 = time.time()
            pred_tmp = model.infer(inp)
            fps_list.append(inp.size(0) / (time.time() - t1)) # mean fps
            pred = torch.sigmoid(pred_tmp).resize(1024,1024).cpu().numpy()
            pred = (np.maximum(pred, 0) / pred.max()) * 255.0
            
            mask_image= Image.fromarray(pred).convert('L')
                # print(mask_image)
            polyp_file = os.path.join(save_dir, filename[0][:-4]+'_polyp.png')
            mask_image.save(polyp_file)
        frame_num += inp.size(0)
        # print(inp.shape)
    # print(frame_num)
    # print(fps_list)
    fps_save.append(frame_num / (time.time() - t0)) # mean save fps
    # print(fps_save)
    return [sum(fps_list)/len(fps_list),sum(fps_save)/len(fps_save)]

def test_all(models_path):
    Note=open('/root/autodl-tmp/method/add_2bolck/save/eval/eval.txt',mode='w')
    Note.truncate(0)
    
    for file_name in sorted(os.listdir(models_path)):
        if(file_name[-4:]=='.pth' and file_name[-8:]!='last.pth'):
            
            model_path=os.path.join(models_path,file_name)
            print(model_path)
            model = models.make(config['model']).cuda()
            sam_checkpoint = torch.load(model_path, map_location='cuda:0')
            model.load_state_dict(sam_checkpoint, strict=True)

            _,_=eval_psnr(loader, model,data_norm=config.get('data_norm'),eval_type=config.get('eval_type'),eval_bsize=config.get('eval_bsize'),verbose=True)

            pred_base='/root/autodl-tmp/method/add_2bolck/pred'
            label_base=spec['dataset']['args']['root_path_2']
            [m_dice,m_iou]=eval_dice(label_base,pred_base,thres=0.5)
            print(model_path +", mean Dice is " + str(m_dice) + ', ' + "IoU is " + str(m_iou))
            Note.write("\n"+model_path[-7:]+", Dice: " + m_dice + ', ' + "IoU: " + m_iou)
    
    Note.close()

def test_one(model_path):
    model = models.make(config['model']).cuda()
    sam_checkpoint = torch.load(model_path, map_location='cuda')
    model.load_state_dict(sam_checkpoint, strict=True)

    [FPS_, FPS]=eval_psnr(loader, model,data_norm=config.get('data_norm'),eval_type=config.get('eval_type'),eval_bsize=config.get('eval_bsize'),verbose=True)

    pred_base='/root/autodl-tmp/method/add_2bolck/pred'
    label_base=spec['dataset']['args']['root_path_2']
    [m_dice,m_iou]=eval_dice(label_base,pred_base,thres=0.5)
    print(model_path +", mean Dice is " + str(m_dice) + ', ' + "IoU is " + str(m_iou)+',' +  'FPS_:%.2f'%FPS_+' FPS:%.2f'%FPS)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="/root/autodl-tmp/method/add_2bolck/configs/demo.yaml")
    parser.add_argument('--model',default="/root/autodl-tmp/method/add_2bolck/save/_demo/21.pth")
    parser.add_argument('--prompt', default='none')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')
    spec = config['test_dataset']
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
                        num_workers=8, shuffle=False)

    # model_path='/data/models/qxf/SAM/save/_demo'
    # models_path='/root/autodl-tmp/method/add_2bolck/save_lr=1e-4/_demo'
    # test_all(models_path)
    model_path='/root/autodl-tmp/method/add_2bolck/save/_demo/21.pth'
    test_one(model_path)
    
    '''
    model = models.make(config['model']).cuda()
    sam_checkpoint = torch.load(args.model, map_location='cuda:0')
    model.load_state_dict(sam_checkpoint, strict=True)

    eval_psnr(loader, model,data_norm=config.get('data_norm'),eval_type=config.get('eval_type'),eval_bsize=config.get('eval_bsize'),verbose=True)

    pred_base='/data/models/qxf/SAM/msak'
    label_base=spec['dataset']['args']['root_path_2']
    [m_dice,m_iou]=eval_dice(label_base,pred_base,thres=0.5)
    print(" mean Dice is " + str(m_dice) + ', ' + "IoU is " + str(m_iou))
    '''