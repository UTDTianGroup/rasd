# Repository for our work "rASD: Robust Active Speaker Detection in Noisy Environments"

## Dataset

We use the AVA-Active Speaker Dataset (AVA-ASD) and noise samples from the RNA Dataset to train and test our proposed method. 

### Download AVA-ASD and RNA datasets

1. Please follow instructions from the [GitHub page of TalkNet](https://github.com/TaoRuijie/TalkNet-ASD#data-preparation) to setup the AVA-Active Speaker dataset.
2. Download RNA dataset from [here](https://drive.google.com/file/d/13QKFeUV0cWMfSCR3H0F6qFh37PEBnEij/view?usp=sharing).

## Installation and Execution

The following commands have been tested on Ubuntu 20.04 with python 3.7.9 and cuda 11.7

1. Create a new conda environment:
```
 conda create -n rasd python=3.7.9
 conda activate rasd
 ```
2. Install PyTorch:
```
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```
3. Install dependencies:
```
pip install -r requirement.txt
```
5. Train rASD:
```
python train_rASD.py --dataPathAVA /path/to/ava-asd-dataset/ --rnaDirPath /path/to/rna-dataset/ --savePath /path/to/output/ 
```
6. Evaluate rASD:
```
python train_rASD.py --dataPathAVA /path/to/ava-asd-dataset/ --rnaDirPath /path/to/rna-dataset/ --evaluation --evalModelPath /path/to/evaluation-checkpoint/
```
A trained checkpoint is available [here](https://drive.google.com/file/d/1TOdZnQhS6DgvulZhdOdo_UAt3ccrtN_g/view?usp=sharing).

### Demo

We provide a demonstration code that can be used to test our work on a video file. This code is modified from the [demonstration code provided by TalkNet](https://github.com/TaoRuijie/TalkNet-ASD/blob/main/demoTalkNet.py). Please use the following command to run the demonstration.
```
python demorASD.py --videoName 4L5Yv6gADuM --videoFolder demo --pretrainModel exps/exp1/model/model_0025.model
```
The output video file should be present at the path `demo/4L5Yv6gADuM/pyavi/video_out.avi`.