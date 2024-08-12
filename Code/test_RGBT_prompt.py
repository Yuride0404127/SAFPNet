import torch as t
from torch import nn
# from RGBT_dataprocessing_CNet import testData1,testData2,testData3
from train_test1.RGBT_dataprocessing_CNet import testData1
from torch.utils.data import DataLoader
import os
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch

import torch.nn.functional as F
import cv2
import torchvision
# from KD_model_3.student_model_single import KD_model_3_student
# from KD_model_3.Ablation_student_model_single import KD_model_3_student
from Model_4.Ablation_Prompt_model_4_v5 import ablation_prompt_model_4_v5
from backbone.Shunted_Transformer.SSA import shunted_b
from backbone.Dformer.Dformer import DFormer_Base
# from PLSCN.model_plscn import PLSCN
# from KD_model_2.teacher_model_V2 import teacher_model
# from KD_model_2.teacher_model_V2_single import teacher_single
# from KD_model_3.student_model_single import KD_model_3_student

import numpy as np
from tqdm import tqdm
from datetime import datetime

test_dataloader1 = DataLoader(testData1, batch_size=1, shuffle=False, num_workers=4)
# test_dataloader2 = DataLoader(testData2, batch_size=1, shuffle=False, num_workers=4)
# test_dataloader3 = DataLoader(testData3, batch_size=1, shuffle=False, num_workers=4)
net = ablation_prompt_model_4_v5()
# prompt_net = DFormer_Base(pretrained=True)
prompt_net = shunted_b(pretrained=True)
prompt_net.load_state_dict(torch.load('/media/yuride/date/model/train_test1/Pth4/Model_4_train_rgb_shunted_b_2024_06_18_16_32_best.pth'), strict=False)   ######gaiyixia

net.load_state_dict(t.load('/media/yuride/date/model/train_test1/Pth4/Ablation_prompt_model_4_v5_wo_CD_2024_07_26_08_45_last.pth'))   ######gaiyixia

a = '/media/yuride/date/RGBT-EvaluationTools/SalMap/'
b = 'Ablation_prompt_model_4_v5_wo_CD'
c = '/rail_362/'
d = '/VT1000/'
e = '/VT5000/'

aa = []

vt800 = a + b + c
vt1000 = a + b + d
vt5000 = a + b + e


path1 = vt800
isExist = os.path.exists(vt800)
if not isExist:
	os.makedirs(vt800)
else:
	print('path1 exist')

with torch.no_grad():
	net.eval()
	net.cuda()
	prompt_net.cuda()
	test_mae = 0

	for sample in tqdm(test_dataloader1, desc="Converting:"):
		image = sample['RGB']
		depth = sample['depth']
		label = sample['label']
		name = sample['name']
		name = "".join(name)

		image = Variable(image).cuda()
		depth = Variable(depth).cuda()
		label = Variable(label).cuda()

		prompt_list = prompt_net(image)
		out1 = net(image, depth, prompt_list)
		# out1 = net(image)
		# s1, s2, NIF_r1, NIF_d1 = net(image, depth)
		out1 = torch.sigmoid(out1[0])
		# out1 = torch.sigmoid(out1)
		out = out1

		out_img = out.cpu().detach().numpy()
		out_img = out_img.squeeze()

		plt.imsave(path1 + name + '.png', arr=out_img, cmap='gray')
		# print(path1 + name + '.png')

	# 	am_out1 = net(image, depth)
	# 	# stage_out4 = net(image)
	# 	out = am_out1[8]
	# 	# print(out.shape)
	# 	out = F.interpolate(out, size=(320, 320), mode='bilinear', align_corners=False)
	# 	out_img = out.cpu().detach().numpy()
	# 	out_img = np.max(out_img, axis=1).reshape(320, 320)
	# 	out_img = (((out_img - np.min(out_img))/(np.max(out_img) - np.min(out_img)))*255).astype(np.uint8)
	# 	out_img = cv2.applyColorMap(out_img, cv2.COLORMAP_JET)
	# 	cv2.imwrite(path1 + name + '.png', out_img)
	# print("Have done!")




##########################################################################################


# path2 = vt1000
# isExist = os.path.exists(vt1000)
# if not isExist:
# 	os.makedirs(vt1000)
# else:
# 	print('path2 exist')
#
# with torch.no_grad():
# 	net.eval()
# 	net.cuda()
# 	test_mae = 0
# 	prec_time = datetime.now()
# 	for i, sample in enumerate(test_dataloader2):
# 		image = sample['RGB']
# 		depth = sample['depth']
# 		label = sample['label']
# 		name = sample['name']
# 		name = "".join(name)
#
# 		image = Variable(image).cuda()
# 		depth = Variable(depth).cuda()
# 		label = Variable(label).cuda()
#
#
# 		# out1,out2,out3,out4,out5 = net(image, depth)
# 		# out1, out2 = net(image, depth)
# 		out1, out2, out3, out4, out5, out6, out7, out8 = net(image, depth)
# 		out = torch.sigmoid(out1)
#
#
# 		out_img = out.cpu().detach().numpy()
# 		out_img = out_img.squeeze()
#
# 		plt.imsave(path2 + name + '.png', arr=out_img, cmap='gray')
# 		print(path2 + name + '.png')
# 	cur_time = datetime.now()




#######################################################################################################
#
# path3 = vt5000
# isExist = os.path.exists(vt5000)
# if not isExist:
# 	os.makedirs(vt5000)
# else:
# 	print('path3 exist')
#
# with torch.no_grad():
# 	net.eval()
# 	net.cuda()
# 	test_mae = 0
# 	prec_time = datetime.now()
# 	for i, sample in enumerate(test_dataloader3):
# 		image = sample['RGB']
# 		depth = sample['depth']
# 		label = sample['label']
# 		name = sample['name']
# 		name = "".join(name)
#
# 		image = Variable(image).cuda()
# 		depth = Variable(depth).cuda()
# 		label = Variable(label).cuda()
#
#
# 		# out1,out2,out3,out4,out5= net(image, depth)
# 		# out1, out2 = net(image, depth)
# 		out1, out2, out3, out4, out5, out6, out7, out8 = net(image, depth)
# 		out = torch.sigmoid(out1)
#
#
#
#
#
# 		out_img = out.cpu().detach().numpy()
# 		out_img = out_img.squeeze()
#
#
# 		plt.imsave(path3 + name + '.png', arr=out_img, cmap='gray')
# 		print(path3 + name + '.png')
#
# 	cur_time = datetime.now()
#   TIANCAIDAOCIYIYOU








