import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch.nn.functional as F
import time
import torch
from torch.backends import cudnn
import numpy as np
import yaml
from torch.utils.data import DataLoader
from Modu.data.ge2e_dataset import get_data_loaders
from Modu.modules.ge2e import GE2ELoss
from Modu  import util
from Modu.meldataset_e2e import Test_MelDataset, get_test_dataset_filelist,mel_denormalize
from hifivoice.inference_e2e import  hifi_infer
from Modu.modules.model.base_CAIN_16 import MagicModel
from Modu.modules.model.Discriminator import SpeakerDiscrimer,FakeDiscrimer
class Solver():
	def __init__(self, config):
		super(Solver, self).__init__()
		self.config = config
		self.local_rank = self.config['local_rank']
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.batch_size = self.config['batch_size']
		self.utters_num = self.config['utters_num']
		self.make_records()
		# train
		self.total_steps = self.config['total_steps']
		self.save_steps = self.config['save_steps']
		self.eval_steps = self.config['eval_steps']
		self.log_steps = self.config['log_steps']
		self.adv_steps = self.config['adv_steps']
		self.schedule_steps = self.config["schedule_steps"]
		self.warmup_steps = self.config["warmup_steps"]
		self.learning_rate = self.config["learning_rate"]
		self.learning_rate_min = self.config["learning_rate_min"]
		self.d_learning_rate = self.config["d_learning_rate"]
		self.d_learning_rate_min = self.config["d_learning_rate_min"]
		self.Generator = MagicModel().to(self.device)
		self.Discrimer = SpeakerDiscrimer().to(self.device)
		self.fake_Discrimer = FakeDiscrimer().to(self.device)
		self.train_data_loader = get_data_loaders(self.config['label_clip_mel_pkl'],self.batch_size,self.utters_num)
		self.optimizer = torch.optim.AdamW(
			[{'params': self.Generator.parameters(), 'initial_lr': self.config["learning_rate"]}],
			self.config["learning_rate"],betas=[self.config["adam_b1"], self.config["adam_b2"]])
		self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.schedule_steps,
																	eta_min=self.learning_rate_min)
		self.d_optimizer = torch.optim.AdamW(
			[{'params': list(self.Discrimer.parameters())+list(self.fake_Discrimer.parameters()), 'initial_lr': self.config["d_learning_rate"]}],
			self.config["d_learning_rate"], betas=[self.config["adam_b1"], self.config["adam_b2"]])
		self.d_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.d_optimizer, self.schedule_steps,
																	eta_min=self.d_learning_rate_min)
		self.criterion = torch.nn.L1Loss()
		self.GE2ELoss = GE2ELoss().to(self.device)
		self.init_epoch = 0
		if self.config['resume']:
			self.resume_model(self.config['resume_num'])
		self.logging.info('config = %s', self.config)
		print('param Generator size = %fM ' % (util.count_parameters_in_M(self.Generator)))

	def make_records(self):
		time_record = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(time.time()))
		self.log_dir = os.path.join(self.config['out_dir'],time_record,"log")
		self.model_dir = os.path.join(self.config['out_dir'],time_record,"model")
		self.write_dir = os.path.join(self.config['out_dir'],time_record,"write")
		self.convt_mel_dir = os.path.join(self.config['out_dir'],time_record,"infer","mel")
		self.convt_voice_dir = os.path.join(self.config['out_dir'],time_record,"infer","voice")
		os.makedirs(self.log_dir, exist_ok=True)
		os.makedirs(self.model_dir, exist_ok=True)
		os.makedirs(self.write_dir, exist_ok=True)
		os.makedirs(self.convt_mel_dir, exist_ok=True)
		os.makedirs(self.convt_voice_dir, exist_ok=True)
		self.logging = util.Logger(self.log_dir, "log.txt")
		# self.writer = util.Writer(self.write_dir)
		self.logging.info('balance  cuda')

	def get_test_data_loaders(self):
		test_filelist = get_test_dataset_filelist(self.config["test_wav_dir"])
		testset = Test_MelDataset(test_filelist,self.config["n_fft"],self.config["num_mels"],
							 self.config["hop_size"], self.config["win_size"], self.config["sampling_rate"],self.config["fmin"],
							 self.config["fmax"], device=self.device)
		test_data_loader = DataLoader(testset, num_workers=1, shuffle=False, sampler=None,
									  batch_size=1, pin_memory=False, drop_last=True)
		return test_data_loader

	def warmup_lr(self, total_step):
		lr = self.learning_rate_min + (total_step/self.warmup_steps)*self.learning_rate
		for param_group in self.optimizer.param_groups:
			param_group['lr'] = lr
		d_lr = self.d_learning_rate_min + (total_step / self.warmup_steps) * self.d_learning_rate
		for param_group in self.d_optimizer.param_groups:
			param_group['lr'] = d_lr

	def train(self):
		self.Generator.train()
		for step in range(self.total_steps):
			batch = next(self.train_data_loader)
			feats = batch[0].to(self.device)
			spk_feats = batch[1].to(self.device)
			fake_mel, spk_emb = self.Generator(feats, spk_feats)
			spk_emb = spk_emb.squeeze(1) # B 256
			spk_emb_sim = spk_emb.view(self.batch_size, self.utters_num, -1)
			# loss
			sim_loss = self.GE2ELoss(spk_emb_sim)
			rec_loss = self.criterion(fake_mel,feats)
			## consist disc
			consist_loss = self.Discrimer(fake_mel, spk_emb)
			adv_consist_loss = torch.mean(torch.abs(1 - consist_loss)**2)
			## fake disc
			fake_loss = self.fake_Discrimer(fake_mel)
			adv_fake_loss = torch.mean(torch.abs(1 - fake_loss) ** 2)
			loss = 20*rec_loss + sim_loss + 2*adv_consist_loss + 2*adv_fake_loss
			# self.reset_grad()
			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()

			if step % self.adv_steps == 0:
				fake_mel, spk_emb = self.Generator(feats, spk_feats)
				spk_emb = spk_emb.squeeze(1)  # B 256
				d_real_consist = self.Discrimer(feats, spk_emb)
				d_fake_consist = self.Discrimer(fake_mel, spk_emb)
				d_loss_consist_real = torch.mean((1 - d_real_consist) ** 2)
				d_loss_consist_fake = torch.mean((0 - d_fake_consist) ** 2)
				d_consist_loss = (d_loss_consist_real + d_loss_consist_fake) / 2.0
				d_real_cls = self.fake_Discrimer(feats)
				d_fake_cls = self.fake_Discrimer(fake_mel)
				d_loss_cls_real = torch.mean((1 - d_real_cls) ** 2)
				d_loss_cls_fake = torch.mean((0 - d_fake_cls) ** 2)
				d_cls_loss = (d_loss_cls_real + d_loss_cls_fake) / 2.0
				d_loss = d_consist_loss + d_cls_loss
				self.d_optimizer.zero_grad()
				d_loss.backward()
				self.d_optimizer.step()
			if step % self.log_steps == 0:
				lr = self.optimizer.state_dict()['param_groups'][0]['lr']
				self.logging.info('【train %d】lr:  %.10f', step, lr)
				self.logging.info('【step %d】loss:%5f rec_loss:%4f sim_loss:%4f adv_consist_loss:%4f adv_fake_loss:%4f d_loss:%4f d_consist_loss:%4f d_loss_cls:%4f', step, loss,rec_loss,sim_loss,adv_consist_loss,adv_fake_loss,d_loss,d_consist_loss,d_cls_loss)
			if step < self.warmup_steps:
				self.warmup_lr(step)
			else:
				self.scheduler.step()
				self.d_scheduler.step()
			if step%self.save_steps == 0 or step == self.total_steps-1:
				save_model_path = os.path.join(self.model_dir, 'checkpoint-%d.pt' % (step))
				self.logging.info('saving the model to the path:%s', save_model_path)
				torch.save({'step': step+1,
							'config': self.config,
							'Generator': self.Generator.state_dict(),
							'Discrimer': self.Discrimer.state_dict(),
							'fake_Discrimer': self.fake_Discrimer.state_dict(),
							'optimizer': self.optimizer.state_dict(),
							'scheduler': self.scheduler.state_dict()},
						   save_model_path, _use_new_zipfile_serialization=False)
				self.infer(step)

	def infer(self,epoch):
		# infer  prepare
		test_data_loader = self.get_test_data_loaders()
		self.Generator.eval()
		mel_npy_file_list=[]
		with torch.no_grad():
			for idx, (src_mel, tar_mel, convert_label) in enumerate(test_data_loader):
				src_mel = src_mel.transpose(1,2) # B dim len
				ori_len = src_mel.size(-1)
				seg = 16
				if ori_len % seg != 0:
					src_mel = F.pad(src_mel, (0, seg - ori_len % seg), mode='reflect')
				src_mel = src_mel.transpose(1, 2)  # B len dim
				src_mel = src_mel.cuda()
				tar_mel = tar_mel.cuda()
				fake_mel,_= self.Generator(src_mel,tar_mel)
				fake_mel = torch.clamp(fake_mel, min=0, max=1)
				fake_mel = mel_denormalize(fake_mel)
				fake_mel = fake_mel.transpose(1,2)
				fake_mel = fake_mel.detach().cpu().numpy()
				file_name = "epoch"+str(epoch)+"_"+convert_label[0]
				mel_npy_file = os.path.join(self.convt_mel_dir, file_name+ '.npy')
				np.save(mel_npy_file, fake_mel, allow_pickle=False)
				mel_npy_file_list.append([file_name,fake_mel])
		hifi_infer(mel_npy_file_list,self.convt_voice_dir)
		self.Generator.train()

def setup_seed(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	torch.backends.cudnn.deterministic = True
	
if __name__ == '__main__':
	print("【Solver RLVC】")
	cudnn.benchmark = True
	config_path = r"./Modu/config.yaml"
	with open(config_path) as f:
		config = yaml.load(f, Loader=yaml.Loader)
	solver = Solver(config)
	solver.train()


