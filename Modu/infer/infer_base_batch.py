import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from torch.backends import cudnn
import numpy as np
import yaml
from torch.utils.data import DataLoader

from Modu.modules.model.base_CAIN_16 import MagicModel
from Modu  import util
from Modu.infer.meldataset_infer import Test_MelDataset, get_test_dataset_filelist,mel_denormalize
from hifivoice.inference_e2e import  hifi_infer

class Solver():
	def __init__(self, config):
		super(Solver, self).__init__()
		self.config = config
		self.local_rank = self.config['local_rank']
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		# self.make_records()
		# train
		self.Generator = MagicModel().to(self.device)

		self.init_epoch = 0
		if self.config['resume']:
			self.resume_model(self.config['resume_path'])
		print('config = %s', self.config)
		print('param Generator size = %fM ' % (util.count_parameters_in_M(self.Generator)))

	def get_test_data_loaders(self,wav_dir,out_dir):
		test_filelist = get_test_dataset_filelist(wav_dir)
		testset = Test_MelDataset(test_filelist,self.config["n_fft"],self.config["num_mels"],
							 self.config["hop_size"], self.config["win_size"], self.config["sampling_rate"],self.config["fmin"],
							 self.config["fmax"], device=self.device)
		test_data_loader = DataLoader(testset, num_workers=1, shuffle=False, sampler=None,
									  batch_size=1, pin_memory=False, drop_last=True)
		self.voice_dir = os.path.join(out_dir, "voice")
		os.makedirs(self.voice_dir, exist_ok=True)
		return test_data_loader


	def resume_model(self, resume_model_path):
		print("*********  [load]   ***********")
		checkpoint_file = resume_model_path
		checkpoint = torch.load(checkpoint_file, map_location='cpu')
		self.Generator.load_state_dict(checkpoint['Generator'])

	def infer(self):
		test_data_loader = self.get_test_data_loaders(self.config["test_wav_dir"],self.config["out_dir"])
		self.Generator.eval()
		mel_npy_file_list=[]
		with torch.no_grad():
			for idx, (src_mel, tar_mel, convert_label) in enumerate(test_data_loader):
				src_mel = src_mel.cuda()
				tar_mel = tar_mel.cuda()
				fake_mel,_= self.Generator(src_mel,tar_mel)
				fake_mel = torch.clamp(fake_mel, min=0, max=1)
				fake_mel = mel_denormalize(fake_mel)
				fake_mel = fake_mel.transpose(1,2)
				fake_mel = fake_mel.detach().cpu().numpy()
				file_name = convert_label[0]
				mel_npy_file_list.append([file_name,fake_mel])
		hifi_infer(mel_npy_file_list,self.voice_dir)
		self.Generator.train()


def setup_seed(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	torch.backends.cudnn.deterministic = True
	
if __name__ == '__main__':
	print("【Solver RLVC ")
	cudnn.benchmark = True
	config_path = r"./Modu/infer/infer_config.yaml"
	with open(config_path) as f:
		config = yaml.load(f, Loader=yaml.Loader)
	solver = Solver(config)
	solver.infer()