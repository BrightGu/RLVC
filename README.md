# RLVC
In this study, we depart from the reliance on extensive pre-trained models for feature representation or mutual information minimization for diverse feature decoupling. Instead, we revisit decoupling methods based on instance normalization. To achieve this, we introduce a novel feature coupling module named cross-adaptive instance normalization (CAIN), which extends the concept of adaptive instance normalization (AdaIN). Beyond offering style injection capabilities similar to AdaIN, CAIN is explicitly designed to maintain content consistency by reconstructing frame-level statistics in mel-spectrograms. The results indicate that CAIN, serving as a lightweight plugin, significantly improves conventional instance normalization-driven approaches. Building upon this, we introduce RLVC, which achieves robust performance with a mere 5.29M parameters.
For the audio samples, please refer to our [demo page](https://brightgu.github.io/RLVC/).


### Envs
python=3.7+

You can install the dependencies with
```bash
pip install -r requirements.txt
```

### Vocoder
The [HiFi-GAN](https://github.com/jik876/hifi-gan) vocoder is employed to convert log mel-spectrograms to waveforms. The model is trained on universal datasets with 13.93M parameters. Please edit the path of hifigan model in "./hifivoice/inference_e2e.py".

### Infer
You can download the [pretrained model](https://drive.google.com/file/d/1FDBpL0xOcrFCkO4MihYhSq2l-_H0RXHD/view?usp=drive_link), and then edit "./Modu/infer/infer_config.yaml".Test Samples could be organized  as "wav22050/*.wav". 
```bash
python ./Modu/infer/infer_base_batch.py
```
Or you can access "./Modu/infer_samples.py" for the source and target speeches specified by yourself.
### Train from scratch

####  Preprocessing
The corpus should be organized as "VCTK22050/$figure$/*.wav", and then edit the "train_wav_dir" and "out_dir" in file "./Modu/predata/robust_mels.py". The output "figure_label_mel_map.pkl" will be used for training.
```bash
python Modu/predata/robust_mels.py
```
#### Training
Please edit the path "test_wav_dir" and "label_clip_mel_pkl" for evaluation and train corpus in config file "./Modu/config.yaml".
```bash
python Modu/solver.py
```
