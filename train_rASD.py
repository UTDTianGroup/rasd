import time, os, torch, argparse, warnings, glob

from dataLoader import train_loader, val_loader
from utils.tools import *
from rASD import rASD_Model

from torch.utils.tensorboard import SummaryWriter

import numpy as np
import os
import csv

def main():
    # The structure of this code is learnt from https://github.com/TaoRuijie/TalkNet-ASD
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description = "Audio Feature Enhancement Baseline for Speaker Classificatin Training")
    # Training details
    parser.add_argument('--lr',           type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--lrDecay',      type=float, default=0.95,  help='Learning rate decay rate')
    parser.add_argument('--maxEpoch',     type=int,   default=40,    help='Maximum number of epochs')
    parser.add_argument('--testInterval', type=int,   default=1,     help='Test and save every [testInterval] epochs')
    parser.add_argument('--batchSize',    type=int,   default=1500,  help='Dynamic batch size, default is 1500 frames.')
    parser.add_argument('--nDataLoaderThread', type=int, default=4,  help='Number of loader threads')
    # Data path
    parser.add_argument('--dataPathAVA',  type=str, default="/mnt/data/datasets/AVAData/", help='Path of the root directory of AVA dataset')
    parser.add_argument('--noisePairsTrainFile',  type=str, default="noisePairsTrain.csv", help='File containing the noise pairs of each of the training samples')
    parser.add_argument('--noisePairsValFile',  type=str, default="noisePairsVal.csv", help='File containing the noise pairs of each of the training samples')
    parser.add_argument('--noiseLabelsTrainFile',  type=str, default="noiseLabels_train.csv", help='File containing the noise pairs of each of the training samples')
    parser.add_argument('--rnaDirPath',  type=str, default="/mnt/data/datasets/Noise_AudioSet/", help='File containing the noise pairs of each of the training samples')
    parser.add_argument('--savePath',     type=str, default="exps/exp1", help='Path of the folder to save the results of experiments.')
    # Data selection
    parser.add_argument('--evalDataType', type=str, default="val", help='Only for AVA, to choose the dataset for evaluation, val or test')
    # For download dataset only, for evaluation only
    parser.add_argument('--downloadAVA',     dest='downloadAVA', action='store_true', help='Only download AVA dataset and do related preprocess')
    parser.add_argument('--processNewDataset', dest='processNewDataset', action='store_true', help='Process a new dataset to format it similar to AVA-ActiveSpeaker for the ASD task.')
    parser.add_argument('--evaluation',      dest='evaluation', action='store_true', help='Only do evaluation using the model pointed to by the evalModelPath argument.')
    parser.add_argument('--alpha', type=float, default=0, help='The noise level (alpha) at which to evaluate the network')
    parser.add_argument('--evalModelPath', type=str, default='exps/exp1/model/model_0025.model', help='Path to the saved trained weights to evaluate the model.')
    parser.add_argument('--samplingRate',  type=int, default=16000, help='Sampling Rate of the audio files.')
    parser.add_argument('--stft_frame',  type=int, default=1022, help='Window length for stft in number of samples')
    parser.add_argument('--stft_hop_sec',  type=float, default=0.01, help='Hop length for stft in seconds')
    parser.add_argument('--computePESQ', dest='computePESQ', action='store_true', help='If selected, the evaluation code also computes PESQ.')

    args = parser.parse_args()
    args.audiosetDirPath = args.rnaDirPath

    # Data loader
    args = init_args(args)

    if (args.downloadAVA == True) or args.processNewDataset:
        preprocess_AVA(args)
        quit()

    if not args.evaluation:
        loader = train_loader(trialFileName = args.trainTrialAVA, \
                            audioPath      = os.path.join(args.audioPathAVA , 'train'), \
                            visualPath     = os.path.join(args.visualPathAVA, 'train'), \
                            **vars(args))
        trainLoader = torch.utils.data.DataLoader(loader, batch_size = 1, shuffle = True, num_workers = args.nDataLoaderThread)

    loader = val_loader(trialFileName = args.evalTrialAVA, \
                        audioPathOrig     = os.path.join(args.audioPathAVA , args.evalDataType), \
                        visualPath    = os.path.join(args.visualPathAVA, args.evalDataType), \
                        **vars(args))
    valLoader = torch.utils.data.DataLoader(loader, batch_size = 1, shuffle = False, num_workers = 16)

    writer = SummaryWriter(log_dir=os.path.join(args.savePath, 'runs'))

    if args.evaluation == True:
        s = rASD_Model(**vars(args))
        s.loadParameters(args.evalModelPath)
        print("Model %s loaded from the file: "%(args.evalModelPath))
        mAP = s.evaluate_network(loader = valLoader, **vars(args))
        print("mAP %2.2f%%"%(mAP))
        quit()

    modelfiles = glob.glob('%s/model_0*.model'%args.modelSavePath)
    modelfiles.sort()  
    if len(modelfiles) >= 1:
        print("Model %s loaded from previous state!"%modelfiles[-1])
        epoch = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][6:]) + 1
        s = rASD_Model(epoch = epoch, **vars(args))
        s.loadParameters(modelfiles[-1])
    else:
        epoch = 1
        s = rASD_Model(epoch = epoch, **vars(args))
    mAPs = []
    scoreFile = open(args.scoreSavePath, "a+")

    while(1):        
        loss, lr, acc = s.train_network(epoch = epoch, loader = trainLoader, **vars(args))

        writer.add_scalar("Loss/train", loss, epoch)
        writer.add_scalar("Acc/train", acc, epoch)
        
        
        if epoch % args.testInterval == 0:        
            s.saveParameters(args.modelSavePath + "/model_%04d.model"%epoch)
            mAPs.append(s.evaluate_network(epoch = epoch, loader = valLoader, **vars(args)))
            print(time.strftime("%Y-%m-%d %H:%M:%S"), "%d epoch, mAP %2.2f%%, bestmAP %2.2f%%"%(epoch, mAPs[-1], max(mAPs)))
            scoreFile.write("%d epoch, LR %f, LOSS %f, mAP %2.2f%%, bestmAP %2.2f%%\n"%(epoch, lr, loss, mAPs[-1], max(mAPs)))
            scoreFile.flush()
            writer.add_scalar("Acc/val", mAPs[-1], epoch)
            writer.flush()
            print(time.strftime("%Y-%m-%d %H:%M:%S"), "%d epoch"%(epoch))
            scoreFile.write("%d epoch, LR %f, LOSS %f\n"%(epoch, lr, loss))
            scoreFile.flush()

        if epoch >= args.maxEpoch:
            quit()

        epoch += 1

    writer.close()

if __name__ == '__main__':
    main()
