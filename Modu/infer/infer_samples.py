import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import pickle
import time
import sys
sys.path.append("/home/gyw/workspace/VC_GAN/UnetVC/")
sys.path.append("../")

import torch
from torch.backends import cudnn
import numpy as np
import yaml
from torch.utils.data import DataLoader
from Modu.modules.model.base_CAIN_16 import MagicModel
from Modu  import util
from Modu.infer.meldataset_infer import Test_MelDataset, mel_denormalize
from hifivoice.inference_e2e import  hifi_infer

class Solver():
	def __init__(self, config):
		super(Solver, self).__init__()
		self.config = config
		self.local_rank = self.config['local_rank']
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.make_records()
		# train
		self.Generator = MagicModel().to(self.device)

		self.init_epoch = 0
		if self.config['resume']:
			self.resume_model(self.config['resume_path'])
		print('config = %s', self.config)
		print('param Generator size = %fM ' % (util.count_parameters_in_M(self.Generator)))

	def make_records(self):
		self.voice_dir = os.path.join(self.config['out_dir'],"voice")
		os.makedirs(self.voice_dir, exist_ok=True)

	def get_test_data_loaders(self):
		test_filelist = self.get_test_dataset_filelist()
		testset = Test_MelDataset(test_filelist,self.config["n_fft"],self.config["num_mels"],
							 self.config["hop_size"], self.config["win_size"], self.config["sampling_rate"],self.config["fmin"],
							 self.config["fmax"], device=self.device)
		test_data_loader = DataLoader(testset, num_workers=1, shuffle=False, sampler=None,
									  batch_size=1, pin_memory=False, drop_last=True)
		return test_data_loader


	def resume_model(self, resume_model_path):
		print("*********  [load]   ***********")
		checkpoint_file = resume_model_path
		checkpoint = torch.load(checkpoint_file, map_location='cpu')
		self.Generator.load_state_dict(checkpoint['Generator'])

	def infer(self):
		# infer  prepare
		test_data_loader = self.get_test_data_loaders()
		# self.criterion = torch.nn.L1Loss()
		self.Generator.eval()
		mel_npy_file_list=[]
		with torch.no_grad():
			for idx, (src_mel, tar_mel, convert_label) in enumerate(test_data_loader):
				src_mel = src_mel.cuda()
				tar_mel = tar_mel.cuda()
				if tar_mel.shape[1] < 40 or src_mel.shape[1] < 40:
					continue
				fake_mel,_= self.Generator(src_mel,tar_mel)
				# fake_mel = src_mel
				fake_mel = torch.clamp(fake_mel, min=0, max=1)
				fake_mel = mel_denormalize(fake_mel)
				fake_mel = fake_mel.transpose(1,2)
				fake_mel = fake_mel.detach().cpu().numpy()
				file_name = convert_label[0]
				mel_npy_file_list.append([file_name,fake_mel])
		# self.logging.info('【infer_%d】 len: %d', epoch, len(mel_npy_list))
		hifi_infer(mel_npy_file_list,self.voice_dir)
		self.Generator.train()

	def get_test_dataset_filelist(self):
		choice_list = []
		## vctk
		f2f_vctk_1_src = r"./SAMPLE_DATA/RLVC_data22/p264_045.wav"
		f2f_vctk_1_tar = r"./SAMPLE_DATA/RLVC_data22/p225_039.wav"
		label = os.path.basename(f2f_vctk_1_src).split(".")[0]+"TO"+os.path.basename(f2f_vctk_1_tar).split(".")[0]
		choice_list.append([f2f_vctk_1_src,f2f_vctk_1_tar,label])
		return choice_list

def setup_seed(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	torch.backends.cudnn.deterministic = True
	
if __name__ == '__main__':
	print("【Solver UnetVC ")
	cudnn.benchmark = True
	config_path = r"./Modu/infer/infer_config.yaml"
	with open(config_path) as f:
		config = yaml.load(f, Loader=yaml.Loader)
	solver = Solver(config)
	solver.infer()


