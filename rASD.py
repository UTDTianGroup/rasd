import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, time, numpy, os, subprocess, pandas, tqdm

from loss import lossAV, lossA, lossV
from model.talkNetModel import talkNetModel

from models_ausep.models import ModelBuilder
from models_ausep.networks import AuSepWeightGenerator, specClassifierNetwork
from utils_ausep.utils import warpgrid, istft_reconstruction, AverageMeter
from models_ausep import criterion

from torchaudio import transforms
from pesq import pesq, PesqError
import librosa
import soundfile as sf

from sklearn.metrics import average_precision_score
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, accuracy_score

class rASD_Model(nn.Module):
    def __init__(self, lr = 0.0001, lrDecay = 0.95, savePath='default',**kwargs):
        super(rASD_Model, self).__init__()

        self.model = talkNetModel().cuda()

        #build audio separator models
        builder = ModelBuilder()
        self.net_unet = builder.build_unet(unet_num_layers = 7, ngf=64, input_nc=1, output_nc=1)
        self.net_unet = self.net_unet.cuda()

        FELayers = [nn.Conv2d(640, 256, 3, padding=1, stride=(2,1)), nn.BatchNorm2d(256), nn.ReLU(), nn.Conv2d(256, 128, 3, padding=1, stride=(2,1)), nn.BatchNorm2d(128), nn.ReLU(), nn.Conv2d(128, 128, 3, padding=1, stride=(2,1)), nn.AvgPool2d(kernel_size=(8,1))]
        self.FEModel = nn.Sequential(*FELayers)
        self.FEModel = self.FEModel.cuda()  

        self.weightGenerator = AuSepWeightGenerator()
        self.weightGenerator = self.weightGenerator.cuda()

        self.lossAV = lossAV().cuda()
        self.lossA = lossA().cuda()
        self.lossV = lossV().cuda()
        self.L1LossWeighted = criterion.L1Loss().cuda()
        weightTensor = torch.FloatTensor([1,0.4,0.486]).cuda()
        self.weightLossCriterion = nn.CrossEntropyLoss(weight=weightTensor).cuda()

        self.optim = torch.optim.Adam(self.parameters(), lr = lr) #No regularization
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size = 1, gamma=lrDecay)

        self.savePath = savePath
        self.weightwriter = SummaryWriter(log_dir=os.path.join(savePath, 'weightsAvgs'))
        self.weightavgsteps = 0

        print(time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f"%(sum(param.numel() for param in self.model.parameters()) / 1024 / 1024))


    def train_network(self, loader, epoch, lr, **kwargs):
        self.train()
        
        self.scheduler.step(epoch - 1)
        index, top1, loss = 0, 0, 0
        lr = self.optim.param_groups[0]['lr']       
        for num, (mixAudioMagFeature, mixAudioPhaseFeature, origAudioMagFeature, origAudioPhaseFeature, visualFeature, labels, imageFrameNoiseLabelIndices) in enumerate(loader, start=1):
            self.zero_grad()
            mixAudioMagFeature = mixAudioMagFeature[0]
            mixAudioPhaseFeature = mixAudioPhaseFeature[0]
            origAudioMagFeature = origAudioMagFeature[0]
            origAudioPhaseFeature = origAudioPhaseFeature[0]
            visualFeature = visualFeature[0]
            imageFrameNoiseLabelIndices = imageFrameNoiseLabelIndices[0]
            imageFrameNoiseLabelIndices = imageFrameNoiseLabelIndices.cuda()
            
            
            #Add 1e-10 to remove the possibility of nan in the gt_masks
            mixAudioMagFeature = mixAudioMagFeature + 1e-10
            origAudioMagFeature = origAudioMagFeature + 1e-10
            # print('mixuaudio mag feature shape in train ', mixAudioMagFeature.shape)
            
            visualEmbed, visualEmbed512 = self.model.forward_visual_frontend(visualFeature.cuda())
            visualEmbedMean = torch.mean(visualEmbed512, dim=1, keepdim = True)
            visualEmbedMean = visualEmbedMean.view(visualEmbedMean.shape[0]*visualEmbedMean.shape[1], visualEmbedMean.shape[2], 1, 1)

            B = mixAudioMagFeature.size(0)
            T = mixAudioMagFeature.size(3)
            
            grid_warp = torch.from_numpy(warpgrid(B, 256, T, warp=True)).cuda()
            mixAudioMagFeatureSampled = F.grid_sample(mixAudioMagFeature.cuda(), grid_warp)
            origAudioMagFeatureSampled = F.grid_sample(origAudioMagFeature.cuda(), grid_warp)

            weightFeatures, finalPredWeights = self.weightGenerator(origAudioMagFeatureSampled)
            imageFrameNoiseLabelIndices = torch.repeat_interleave(imageFrameNoiseLabelIndices, repeats=4, dim=1)
            
            weightLoss = self.weightLossCriterion(weightFeatures, imageFrameNoiseLabelIndices)

            gt_masks = origAudioMagFeatureSampled/mixAudioMagFeatureSampled
            gt_masks.clamp_(0., 5.)
            
            audio_log_mags = torch.log(mixAudioMagFeatureSampled).detach()

            mask_prediction, FEInput = self.net_unet(audio_log_mags.cuda(), visualEmbedMean.cuda())

            FEOutput = self.FEModel(FEInput)
            FEOutput = torch.squeeze(FEOutput, 2)

                
    
            weight = torch.log1p(mixAudioMagFeatureSampled)
            weight = torch.clamp(weight, 1e-3, 10)

            
            finalPredWeights = torch.unsqueeze(finalPredWeights, dim=1)
            weightValueLoss = torch.mean(torch.ones_like(finalPredWeights) - finalPredWeights)
            weight = finalPredWeights*weight
            finalPredWeightsSqueezed = finalPredWeights.squeeze()
            bool0 = (imageFrameNoiseLabelIndices == torch.zeros_like(finalPredWeightsSqueezed))
            bool1 = (imageFrameNoiseLabelIndices == torch.ones_like(finalPredWeightsSqueezed))
            bool2 = (imageFrameNoiseLabelIndices == 2*torch.ones_like(finalPredWeightsSqueezed))

            finalPredWeightsSqueezed0 = torch.mul(finalPredWeightsSqueezed, bool0)
            finalPredWeightsSqueezed1 = torch.mul(finalPredWeightsSqueezed, bool1)
            finalPredWeightsSqueezed2 = torch.mul(finalPredWeightsSqueezed, bool2)

            weightmean = torch.mean(finalPredWeightsSqueezed)
            weightmean0 = torch.sum(finalPredWeightsSqueezed0)/torch.sum(bool0)
            weightmean1 = torch.sum(finalPredWeightsSqueezed1)/torch.sum(bool1)
            weightmean2 = torch.sum(finalPredWeightsSqueezed2)/torch.sum(bool2)

            self.weightwriter.add_scalar('train/avgweights', weightmean, self.weightavgsteps)
            self.weightwriter.add_scalar('train/cleanweights', weightmean0, self.weightavgsteps)
            self.weightwriter.add_scalar('train/noiseweights', weightmean1, self.weightavgsteps)
            self.weightwriter.add_scalar('train/nonspweights', weightmean2, self.weightavgsteps)
            self.weightavgsteps += 1
            self.weightwriter.flush()

            audioEmbed = FEOutput.transpose(1,2)

            audioEmbed, visualEmbed = self.model.forward_cross_attention(audioEmbed, visualEmbed)
            outsAV= self.model.forward_audio_visual_backend(audioEmbed, visualEmbed)  
            outsA = self.model.forward_audio_backend(audioEmbed)
            outsV = self.model.forward_visual_backend(visualEmbed)

            labels = labels[0].reshape((-1)).cuda() # Loss
            nlossAV, _, _, prec = self.lossAV.forward(outsAV, labels)
            nlossA = self.lossA.forward(outsA, labels)
            nlossV = self.lossV.forward(outsV, labels)
            nloss = nlossAV + 0.4 * nlossA + 0.4 * nlossV
        
            sepLoss = self.L1LossWeighted(mask_prediction, gt_masks, weight)
            
            combinedLoss = 0.1*nloss + sepLoss + 0.1*weightLoss + 0.25*weightValueLoss

            self.optim.zero_grad()
            
            combinedLoss.backward()
            
            self.optim.step()
            loss += combinedLoss.detach().cpu().numpy()
            if not (self.trainAudioSeparator or self.trainWeightGenerator):
                top1 += prec
            
            index += len(labels)
            
            sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
            " [%2d] Lr: %5f, Training: %.2f%%, "    %(epoch, lr, 100 * (num / loader.__len__())) + \
            " Loss: %.5f, ACC: %2.2f%% , numBatches: %d \r"        %(loss/(num), 100 * (top1/index), loader.__len__()))
            sys.stderr.flush()  
        sys.stdout.write("\n")      
        return loss/num, lr, 100*(top1/index)

    def evaluate_network(self, loader, evalCsvSave, evalOrig, savePath, stft_frame, stft_hop_sec, samplingRate, computePESQ, **kwargs):
        self.eval()
        predScores = []

        countLines = 0
        countErrorPesq = 0
        countValidPesq = 0
        totalPesq = 0

        for mixAudioMagFeature, mixAudioPhaseFeature, origAudioMagFeature, origAudioPhaseFeature, visualFeature, labels, fps, audioLength in tqdm.tqdm(loader):
            mixAudioMagFeature = mixAudioMagFeature[0]
            mixAudioPhaseFeature = mixAudioPhaseFeature[0]
            origAudioMagFeature = origAudioMagFeature[0]
            origAudioPhaseFeature = origAudioPhaseFeature[0]
            visualFeature = visualFeature[0]
            fps = fps[0][0]
            audioLength = audioLength[0][0]

            mixAudioMagFeature = mixAudioMagFeature + 1e-10
            origAudioMagFeature = origAudioMagFeature + 1e-10

            numSamples = mixAudioMagFeature.shape[0]
            
            with torch.no_grad():
                visualEmbed, visualEmbed512 = self.model.forward_visual_frontend(visualFeature.cuda())
    
                visualEmbedMean = torch.mean(visualEmbed512, dim=1, keepdim = True)
                visualEmbedMean = visualEmbedMean.view(visualEmbedMean.shape[0]*visualEmbedMean.shape[1], visualEmbedMean.shape[2], 1, 1)
            
                B = mixAudioMagFeature.size(0)
                T = mixAudioMagFeature.size(3)
                
                grid_warp = torch.from_numpy(warpgrid(B, 256, T, warp=True)).cuda()
                mixAudioMagFeatureSampled = F.grid_sample(mixAudioMagFeature.cuda(), grid_warp)

                
                audio_log_mags = torch.log(mixAudioMagFeatureSampled).detach()
                mask_prediction, FEInput = self.net_unet(audio_log_mags.cuda(), visualEmbedMean.cuda())

                B = mixAudioMagFeature.size(0)
                
                grid_unwarp = torch.from_numpy(warpgrid(B, 1022//2+1, mask_prediction.size(3), warp=False)).cuda()
                pred_masks_linear = F.grid_sample(mask_prediction, grid_unwarp)

                pred_mag = mixAudioMagFeature.cuda() * pred_masks_linear.cuda()
                
                if computePESQ:
                    spec = pred_mag.type(torch.cfloat) * torch.exp(1j*mixAudioPhaseFeature.cuda())
                    specOrig = origAudioMagFeature.type(torch.cfloat).cuda() * torch.exp(1j*origAudioPhaseFeature.cuda())
                    specNoisy = mixAudioMagFeature.type(torch.cfloat).cuda() * torch.exp(1j*mixAudioPhaseFeature.cuda())

                    
                    invSpecTransform = transforms.InverseSpectrogram(n_fft=stft_frame, hop_length=int(stft_hop_sec*samplingRate*25/fps))
                    invSpecTransform = invSpecTransform.cuda()
                    pred_wav = invSpecTransform(spec, length=audioLength)
                    orig_wav = invSpecTransform(specOrig, length=audioLength)
                    noisy_wav = invSpecTransform(specNoisy, length=audioLength)
                    

                    pred_wav = torch.reshape(pred_wav, (numSamples, -1))
                    orig_wav = torch.reshape(orig_wav, (numSamples, -1))
                    noisy_wav = torch.reshape(noisy_wav, (numSamples, -1))

                FEOutput = self.FEModel(FEInput)
                
                FEOutput = torch.squeeze(FEOutput, 2)

                audioEmbed = FEOutput.transpose(1,2)
                
                audioEmbed, visualEmbed = self.model.forward_cross_attention(audioEmbed, visualEmbed)
                outsAV= self.model.forward_audio_visual_backend(audioEmbed, visualEmbed)  
                labels = labels[0].reshape((-1)).cuda()             
                _, predScore, _, _ = self.lossAV.forward(outsAV, labels)
                predScore_neg = predScore[:,0].detach().cpu().numpy()    
                predScore = predScore[:,1].detach().cpu().numpy()
                predScores.extend(predScore)
                
                if computePESQ:
                    if torch.any(labels): #Compute pesq only if there is a speaker
                        #Ref to pesq : https://github.com/ludlows/PESQ
                        pesq_score = pesq(16000, torch.reshape(orig_wav, (-1,)).cpu().numpy(), torch.reshape(pred_wav, (-1,)).cpu().numpy(), mode='wb', on_error=PesqError.RETURN_VALUES)
                        if pesq_score == -1:
                            countErrorPesq = countErrorPesq + 1
                        else:
                            countValidPesq = countValidPesq + 1
                            totalPesq = totalPesq + pesq_score

                countLines += 1
        
        if computePESQ:
            print('pesq ', totalPesq/countValidPesq)
            print('countValidPesq ', countValidPesq)
            print('countErrorPesq ', countErrorPesq)
        if computePESQ:
            return totalPesq/countValidPesq
        else:
            evalLines = open(evalOrig).read().splitlines()[1:]
            labels = []
            labels = pandas.Series( ['SPEAKING_AUDIBLE' for line in evalLines])
            scores = pandas.Series(predScores)
            print("score count 1 ", scores.count())
            evalRes = pandas.read_csv(evalOrig)
            evalRes['score'] = scores
            print("score count 2 ", evalRes['score'].count())
            evalRes['label'] = labels
            evalRes.drop(['label_id'], axis=1,inplace=True)
            evalRes.drop(['instance_id'], axis=1,inplace=True)
            evalRes.to_csv(evalCsvSave, index=False)
            print('evalOrig', evalOrig)
            print('evalCsvSave', evalCsvSave)
            cmd = "python -O utils/get_ava_active_speaker_performance.py -g %s -p %s "%(evalOrig, evalCsvSave)
            mAP = float(str(subprocess.run(cmd, shell=True, capture_output =True).stdout).split(' ')[2][:5])
            return mAP

    def saveParameters(self, path):
        torch.save(self.state_dict(), path)

    def loadParameters(self, path):
        selfState = self.state_dict()
        loadedState = torch.load(path)
        for name, param in loadedState.items():
            origName = name;
            if name not in selfState:
                name = name.replace("module.", "")
                if name not in selfState:
                    print("%s is not in the model."%origName)
                    continue
            if selfState[name].size() != loadedState[origName].size():
                sys.stderr.write("Wrong parameter length: %s, model: %s, loaded: %s"%(origName, selfState[name].size(), loadedState[origName].size()))
                continue
            selfState[name].copy_(param)
