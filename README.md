## pre-saliency IQA

PyTorch 0.4 implementation of the following paper:[Z. Cheng, M. Takeuchi and J. Katto, A Pre-Saliency Map Based Blind Image Quality Assessment via Convolutional Neural Networks, 2017 IEEE International Symposium on Multimedia (ISM), Taichung, 2017, pp. 77-82.](https://ieeexplore.ieee.org/document/8241584)   

### Database

[LIVE](https://live.ece.utexas.edu/research/quality/subjective.htm)

*** Download the database, then gbvs code is used to generate the correcponding saliency images.  

### Requirements

* h5py==2.7.1
* numpy==1.14.2
* PyYAML==3.12
* scipy==1.0.1
* torch==0.4.1
* torchvision==0.2.1
* tensorflow-gpu==1.0.0
* tensorboardX==1.2
* pytorch-ignite==0.1.2

### Training

python main1.py --batch_size=128 --epochs=2000 --lr=0.0001 --exp_id=0 --database=LIVE

** Before training, the img_dir in config.yaml must to be specified. Train/Val/Test split ratio in intra-database experiments can be set in config.yaml (default is 0.8/0.1/0.1).