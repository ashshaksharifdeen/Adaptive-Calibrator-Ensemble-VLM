from utils_a import *
from vlm_utils import *
import metrics
import metrics2
import fire
import numpy as np
import torch
import torchvision as tv
import torchvision
import os
import ipdb
import tqdm
import time
from calibration_methods.temperature_scaling import tune_temp
import calibration_methods.splines as splines
from calibration_methods.vector_scaling import VectorScaling, VectorScaling_NN
import open_clip
from torchvision import datasets, transforms
import torch
from PIL import Image
import open_clip
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
from sklearn import calibration
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch.nn as nn
import torch.optim as optim
from sklearn.tree import DecisionTreeRegressor as DT
from sklearn.isotonic import IsotonicRegression as IR
from sklearn.linear_model import LogisticRegression as LR
from scipy.stats import norm
from tqdm import tqdm
import pandas as pd
import json

resnet50_files = [
                    # 'probs_resnet152_imgnet_logits.p',
                   'cal_resnet50_imgnet-v2-a_logits.p',
                   'cal_resnet50_imgnet-v2-b_logits.p',
                   'cal_resnet50_imgnet-v2-c_logits.p',
                   'cal_resnet50_imgnet-s_logits.p',
                   # 'cal_resnet152_imgnet-gauss_logits.p',
                   # 'cal_resnet152_imgnet-a_logits.p',
                   # 'cal_resnet152_imgnet-r_logits.p',
                   ]

resnet152_files = [
                    # 'probs_resnet152_imgnet_logits.p',
                   'cal_resnet152_imgnet-v2-a_logits.p',
                   'cal_resnet152_imgnet-v2-b_logits.p',
                   'cal_resnet152_imgnet-v2-c_logits.p',
                   'cal_resnet152_imgnet-s_logits.p',
                   'cal_resnet152_imgnet-a_logits.p',
                   'cal_resnet152_imgnet-r_logits.p',
                   ]

vit_small_patch32_224_files = [
                    'probs_vit_small_patch32_224_imgnet_logits.p',
                   'cal_vit_small_patch32_224_imgnet-v2-a_logits.p',
                   'cal_vit_small_patch32_224_imgnet-v2-b_logits.p',
                   'cal_vit_small_patch32_224_imgnet-v2-c_logits.p',
                   'cal_vit_small_patch32_224_imgnet-s_logits.p',
                   # 'cal_vit_small_patch32_224_imgnet-gauss_logits.p',
                   'cal_vit_small_patch32_224_imgnet-a_logits.p',
                   'cal_vit_small_patch32_224_imgnet-r_logits.p',
                   ]

deit_small_patch16_224_files = [
                    'probs_deit_small_patch16_224_imgnet_logits.p',
                   'cal_deit_small_patch16_224_imgnet-v2-a_logits.p',
                   'cal_deit_small_patch16_224_imgnet-v2-b_logits.p',
                   'cal_deit_small_patch16_224_imgnet-v2-c_logits.p',
                   'cal_deit_small_patch16_224_imgnet-s_logits.p',
                   # 'cal_deit_small_patch16_224_imgnet-gauss_logits.p',
                   'cal_deit_small_patch16_224_imgnet-a_logits.p',
                   'cal_deit_small_patch16_224_imgnet-r_logits.p',
                   ]

deit_base_patch16_224_files = [
                    'probs_deit_base_patch16_224_imgnet_logits.p',
                   'cal_deit_base_patch16_224_imgnet-v2-a_logits.p',
                   'cal_deit_base_patch16_224_imgnet-s_logits.p',
                   'cal_deit_base_patch16_224_imgnet-gauss_logits.p',
                   'cal_deit_base_patch16_224_imgnet-a_logits.p',
                   ]
dct = {}
dct['resnet50_files'] = resnet50_files
dct['resnet152_files'] = resnet152_files
dct['vit_small_patch32_224_files'] = vit_small_patch32_224_files
dct['deit_small_patch16_224_files'] = deit_small_patch16_224_files
dct['deit_base_patch16_224_files'] = deit_base_patch16_224_files



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
batch_size = 128

model_name = 'ViT-B-16'#'ViT-B-32-quickgelu'
dataset_name = 'CIFAR10'

model, _, preprocess = open_clip.create_model_and_transforms(model_name,
    pretrained='laion400m_e31',
    device=device)

tokenizer = open_clip.get_tokenizer(model_name)

test_dset,num_classes = get_test_set(dataset_name, preprocess)
classes, templates = get_openai_prompts(dataset_name)
val_dset = get_val_set(dataset_name, classes, preprocess)

t = templates[0]

def demo(model, tokenizer, test_dset, val_dset, text_template=t, device=device):
    #geting the logit values
        """
        for f in dct[files]:
        print('###############################################################################')
        print('Evaluating : {}'.format(f))
        dataset = f.split('_')[-2]
        if dataset == 'imgnet-a':
            valid_indices = imagenet_a_valid_labels()
        elif dataset == 'imgnet-r':
            valid_indices = imagenet_r_valid_labels()
        else:
            valid_indices = None
        # set your path of logits here
        path = os.path.join('./logits',f)
        #logits values of validation and test with respective label of those
        (y_probs_val, y_val), (y_probs_test, y_test) = unpickle_probs(path)
        y_val = np.squeeze(y_val)
        y_test = np.squeeze(y_test)"""
        

        #validation data
        predictions_val, actual_val, img_embedings_val, probs_val = get_preds(model, tokenizer, val_dset, text_template=t, device=device)
        avg_conf_val = probs_val.cpu().numpy().mean()

        #test data
        predictions_test, actual_test, img_embedings_test, probs_test = get_preds(model, tokenizer, test_dset, text_template=t, device=device)
        avg_conf_test = probs_test.cpu().numpy().mean()

        #This is important to create alpha value
        #you can calculate average confident and accuracy from the returning get_preds 
        """avg_conf_val, _ = metrics.AvgConf(torch.from_numpy(y_probs_val).cuda(), torch.from_numpy(y_val).cuda())
        if valid_indices:
            avg_conf_test, _ = metrics.AvgConf(torch.from_numpy(y_probs_test[:, valid_indices]).cuda(), torch.from_numpy(y_test).cuda())
        else:
            avg_conf_test, _ = metrics.AvgConf(torch.from_numpy(y_probs_test).cuda(), torch.from_numpy(y_test).cuda())
        alpha = avg_conf_test/avg_conf_val"""
        alpha = avg_conf_test/avg_conf_val 

        # r = 0
        #creating easy temperature value
        r1 = 0
        print('n_T/n_P : {}'.format(r1))
        sampled_logits, sampled_labels = sample_calibration_set(predictions_val, actual_val, img_embedings_val, probs_val, r1)

        temp = tune_temp(torch.from_numpy(sampled_logits), torch.from_numpy(np.squeeze(sampled_labels)))
        #selecting the range of scaled logit value.
        
        
        logits_temp_0 = img_embedings_test / temp
        
        """
        #spiline method
        ece_criterion = splines._ECELoss(n_bins=25)
        softmax_val = torch.nn.functional.softmax(torch.from_numpy(sampled_logits), dim=1).cpu().numpy()
        if valid_indices:
            softmax_test = torch.nn.functional.softmax(torch.from_numpy(y_probs_test[:, valid_indices]), dim=1).cpu().numpy()
        else:
            softmax_test = torch.nn.functional.softmax(torch.from_numpy(y_probs_test), dim=1).cpu().numpy()
        confs_spline_0, labels_spline_0 = splines.cal_splines(softmax_val, sampled_labels, softmax_test, y_test, ece_criterion)

        model = VectorScaling_NN(classes=y_probs_val.shape[1]).cuda()
        torch_sample_logits = torch.from_numpy(sampled_logits).cuda()
        torch_sample_labels = torch.from_numpy(sampled_labels).cuda()
        model.fit(torch_sample_logits, torch_sample_labels)
        torch_test_logits = torch.from_numpy(y_probs_test).cuda()
        # torch_test_labels = torch.from_numpy(np.squeeze(y_test)).cuda()
        preds_test_0 = model(torch_test_logits).detach()
        if valid_indices:
            preds_test_0 = preds_test_0[:, valid_indices]"""

        # r = 2
        #creating hard/difficult/OOD distribution value
        #here Probability is okay but there problem with label value think how to transfer lable in CLIP. 
        r2 = 0.1
        print('n_T/n_P : {}'.format(r2))
        #sampeled logit from OOD
        sampled_logits, sampled_labels = sample_calibration_set(predictions_val, actual_val, img_embedings_val, probs_val, r2)
        #try different method for conf of calibration set
        #avg_conf_cal, _ = metrics.AvgConf(torch.from_numpy(sampled_logits).cuda(), torch.from_numpy(sampled_labels).cuda())
        # alpha = 1-abs((avg_conf_test-avg_conf_val)/(avg_conf_cal-avg_conf_val))

        temp = tune_temp(torch.from_numpy(sampled_logits), torch.from_numpy(np.squeeze(sampled_labels)))
        
        logits_temp_1 = img_embedings_test / temp

        """
        #Spiline method
        ece_criterion = splines._ECELoss(n_bins=25)
        softmax_val = torch.nn.functional.softmax(torch.from_numpy(sampled_logits), dim=1).cpu().numpy()
        if valid_indices:
            softmax_test = torch.nn.functional.softmax(torch.from_numpy(y_probs_test[:, valid_indices]),
                                                       dim=1).cpu().numpy()
        else:
            softmax_test = torch.nn.functional.softmax(torch.from_numpy(y_probs_test), dim=1).cpu().numpy()
        confs_spline_1, labels_spline_1 = splines.cal_splines(softmax_val, sampled_labels, softmax_test, y_test, ece_criterion)

        model = VectorScaling_NN(classes=y_probs_val.shape[1]).cuda()
        torch_sample_logits = torch.from_numpy(sampled_logits).cuda()
        torch_sample_labels = torch.from_numpy(sampled_labels).cuda()
        model.fit(torch_sample_logits, torch_sample_labels)
        torch_test_logits = torch.from_numpy(y_probs_test).cuda()
        # torch_test_labels = torch.from_numpy(np.squeeze(y_test)).cuda()
        preds_test_1 = model(torch_test_logits).detach()
        if valid_indices:
            preds_test_1 = preds_test_1[:, valid_indices]"""

        # ensemble
        #calibrated logit value
        logits_temp = alpha*logits_temp_0 + (1-alpha)*logits_temp_1
        #here use different function to predict similiraties with text and calibrated image embedings
        
        predictions, actual, probs_scaled_cifar10 = get_final_preds(model, tokenizer, test_dset, logits_temp, actual_test, text_template=t, device=device)
        ECE_scaled_cifar10, _, acc_scaled = get_metrics(predictions, actual, probs_scaled_cifar10)
        bins_scaled_cifar10, _, bin_accs_scaled_cifar10, _, bin_sizes_scaled_cifar10 = calc_bins(predictions, actual, probs_scaled_cifar10)
        
        """torch_test_logits = torch.from_numpy(logits_temp).cuda()
        torch_test_labels = torch.from_numpy(y_test).cuda()
        avg_conf, acc = metrics.AvgConf(torch_test_logits, torch_test_labels)
        acc, ece = metrics.ECE(torch_test_logits, torch_test_labels)
        # acc, mce = metrics.MCE(torch_test_logits, torch_test_labels, isLogits=0)
        # nll = metrics.NLL(torch_test_logits, torch_test_labels, 0)
        # bs = metrics.BS(torch_test_logits, torch_test_labels, 0)
        # softmax_test = torch.nn.functional.softmax(torch.from_numpy(logits_temp), dim=1).cpu().numpy()
        # cw_ece = metrics2.classwise_ECE(softmax_test, y_test)
        print("Avg Conf Val : %4f, Cal : %4f, Test : %4f" %(avg_conf_val, avg_conf_cal, avg_conf_test))
        print('temp logits ensemble -------> ')
        print('ACC: %4f' % (acc))
        # print('NLL: %4f' % (nll))
        # print('BS: %4f' % (bs))
        # print('MCE: %4f' % (mce))
        print('AVG Conf: %4f' % (avg_conf))
        print('conf-ECE: %4f' % (ece))
        # print('cw-ECE: %4f' % (cw_ece))

        softmax_0 = torch.nn.functional.softmax(torch.from_numpy(logits_temp_0), dim=1).cuda()
        softmax_1 = torch.nn.functional.softmax(torch.from_numpy(logits_temp_1), dim=1).cuda()
        #calibrated softmax value.
        softmax_temp = alpha*softmax_0+softmax_1*(1-alpha)
        avg_conf, acc = metrics.AvgConf(softmax_temp, torch_test_labels, isLogits=1)
        acc, ece = metrics.ECE(softmax_temp, torch_test_labels, isLogits=1)
        print('temp softmax ensemble -------> ')
        print('AVG Conf: %4f' % (avg_conf))
        print('conf-ECE: %4f' % (ece))

        confs_spline = alpha*confs_spline_0 + confs_spline_1*(1-alpha)
        avg_conf, acc = metrics.AvgConf(torch.from_numpy(confs_spline), torch.from_numpy(np.squeeze(labels_spline_1)), isLogits=2)
        acc, ece = metrics.ECE(torch.from_numpy(confs_spline), torch.from_numpy(np.squeeze(labels_spline_1)), isLogits=2)
        print('splines (confidence) ensemble -------> ')
        print('ACC: %4f' % (acc))
        print('AVG Conf: %4f' % (avg_conf))
        print('conf-ECE: %4f' % (ece))

        softmax_0 = torch.nn.functional.softmax(preds_test_0, dim=1)
        softmax_1 = torch.nn.functional.softmax(preds_test_1, dim=1)
        softmax_temp = alpha * softmax_0 + softmax_1 * (1 - alpha)
        avg_conf, acc = metrics.AvgConf(softmax_temp, torch_test_labels, isLogits=1)
        acc, ece = metrics.ECE(softmax_temp, torch_test_labels, isLogits=1)
        print('vector scaling softmax ensemble -------> ')
        print('ACC: %4f' % (acc))
        print('AVG Conf: %4f' % (avg_conf))
        print('conf-ECE: %4f' % (ece))"""

if __name__ == '__main__':
    fire.Fire(demo)
