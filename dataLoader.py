import os, torch, numpy, cv2, random, glob, python_speech_features
from scipy.io import wavfile
from torchvision.transforms import RandomCrop
import csv
import numpy as np
import librosa

def overlapAlpha(sourceAudio, noiseAudio, alpha):
    if len(noiseAudio) <= len(sourceAudio):
        shortage = len(sourceAudio) - len(noiseAudio)
        noiseAudio = np.pad(noiseAudio, (0, shortage), 'wrap')
    else:
        # When the noiseAudio is longer, then split the noise audio into segments and find the highest energy segment to merge with noise
        segmentLength = len(sourceAudio)
        if segmentLength == 0:
            noiseAudio = sourceAudio
        else:
            numSegments = len(noiseAudio)//segmentLength
            if(len(noiseAudio)%segmentLength != 0):
                numSegments = numSegments+1
            segments = [noiseAudio[i*segmentLength:(i+1)*segmentLength] for i in range(numSegments-1)]
            segments.append(noiseAudio[len(noiseAudio)-segmentLength:len(noiseAudio)]) #Append the last segment separately as length of noiseAudio might not be an integer multiple of the length of source audio
            energy = [np.mean(seg**2) for seg in segments]
            maxEnergy = max(energy)
            maxIndex = energy.index(maxEnergy)
            noiseAudio = segments[maxIndex]
    
    mergedAudio = sourceAudio + alpha*noiseAudio

    return mergedAudio

def generate_audio_set_eval(dataPathOrig, batchList, noisePairsDict, audiosetDirPath, alpha):
    audioSetOrig = {}
    audioSetNoisy = {}
    for line in batchList:
        data = line.replace(":", "_").split('\t')
        videoName = data[0][:11]
        dataName = data[0]
        audioFilePath = os.path.join(dataPathOrig, videoName, dataName + '.wav')
        audioOrig, _ = librosa.load(audioFilePath, sr=16000)

        noiseFileList = noisePairsDict[audioFilePath]
        noiseFilePath = os.path.join(audiosetDirPath, 'audios', noiseFileList[1], noiseFileList[0])

        noiseAudio, _ = librosa.load(noiseFilePath, sr=16000)
        
        if(len(noiseAudio.shape) > 1):
            noiseAudio = np.mean(noiseAudio, axis=-1)

        audioSetOrig[dataName] = audioOrig
        audioSetNoisy[dataName] = overlapAlpha(audioOrig, noiseAudio, alpha)
    return audioSetOrig, audioSetNoisy

def overlapAlphaTrain(sourceAudio, noiseAudio, alpha):
    if len(noiseAudio) <= len(sourceAudio):
        shortage = len(sourceAudio) - len(noiseAudio)
        noiseAudio = np.pad(noiseAudio, (0, shortage), 'wrap')
    else:
        segmentLength = len(sourceAudio)
        numSegments = len(noiseAudio)//segmentLength
        if(len(noiseAudio)%segmentLength != 0):
            numSegments = numSegments+1
        segments = [noiseAudio[i*segmentLength:(i+1)*segmentLength] for i in range(numSegments-1)]
        segments.append(noiseAudio[len(noiseAudio)-segmentLength:len(noiseAudio)]) #Append the last segment separately as length of noiseAudio might not be an integer multiple of the length of source audio

        randSegmentIndex = np.random.choice(numSegments)
        noiseAudio = segments[randSegmentIndex]

    mergedAudio = sourceAudio + alpha*noiseAudio

    return mergedAudio, sourceAudio

def generate_audio_set_noisy(dataPath, batchList, noisePairsDict, audiosetDirPath, samplingRate):
    audioSet = {}
    audioSetOrig = {}
    audioSet_tA = {}
    audioSetOrig_tA = {}
    for line in batchList:
        data = line.replace(":", "_").split('\t')
        videoName = data[0][:11]
        dataName = data[0]
        audioFilePath = os.path.join(dataPath, videoName, dataName + '.wav')
        sourceAudio, _ = librosa.load(audioFilePath, sr=16000)

        noiseFileList = noisePairsDict[audioFilePath]
        numNoiseFiles = int(len(noiseFileList)/2)
        randNoiseFileIndex = np.random.choice(numNoiseFiles)
        noiseFilePath = os.path.join(audiosetDirPath, 'audios', noiseFileList[2*randNoiseFileIndex+1], noiseFileList[2*randNoiseFileIndex])
        noiseAudio, _ = librosa.load(noiseFilePath, sr=16000)
        
        if(len(noiseAudio.shape) > 1):
            noiseAudio = np.mean(noiseAudio, axis=-1)
        randAlpha = np.random.uniform(0,1)
        
        audioSet[dataName], audioSetOrig[dataName] = overlapAlphaTrain(sourceAudio, noiseAudio, randAlpha)

    return audioSet, audioSetOrig

def generate_spectrogram_magphase(audio, stft_frame, stft_hop, with_phase=True):
    spectro = librosa.core.stft(audio, hop_length=stft_hop, n_fft=stft_frame, center=True)
    spectro_mag, spectro_phase = librosa.core.magphase(spectro)
    spectro_mag = np.expand_dims(spectro_mag, axis=0)
    # print('spectro mag shape in the data loader ', spectro_mag.shape)
    if with_phase:
        spectro_phase = np.expand_dims(np.angle(spectro_phase), axis=0)
        return spectro_mag, spectro_phase
    else:
        return spectro_mag

def load_audio(data, dataPath, numFrames, stft_frame, stft_hop_sec, samplingRate, audioAug, audioSet = None, audioSetOrig = None):
    dataName = data[0]
    fps = float(data[2])    
    audio = audioSet[dataName]    
    audioOrig = audioSetOrig[dataName]

    audioLength = len(audio)

    mix_audio_mag, mix_audio_phase = generate_spectrogram_magphase(audio, stft_frame, int(stft_hop_sec*samplingRate*25/fps))
    orig_audio_mag, orig_audio_phase = generate_spectrogram_magphase(audioOrig, stft_frame, int(stft_hop_sec*samplingRate*25/fps))
    
    maxAudioFrames = int(round(numFrames * 4))
    if mix_audio_mag.shape[-1] < maxAudioFrames:
        shortage = maxAudioFrames - mix_audio_mag.shape[-1]
        mix_audio_mag = numpy.pad(mix_audio_mag, ((0,0), (0, 0), (0,shortage)), 'wrap')
        mix_audio_phase = numpy.pad(mix_audio_phase, ((0,0),(0, 0), (0,shortage)), 'wrap')
        orig_audio_mag = numpy.pad(orig_audio_mag, ((0,0),(0, 0), (0,shortage)), 'wrap')
        orig_audio_phase = numpy.pad(orig_audio_phase, ((0,0),(0, 0), (0,shortage)), 'wrap')
    mix_audio_mag = mix_audio_mag[:, :, :int(round(numFrames*4))]
    mix_audio_phase = mix_audio_phase[:, :, :int(round(numFrames*4))]
    orig_audio_mag = orig_audio_mag[:, :, :int(round(numFrames*4))]
    orig_audio_phase = orig_audio_phase[:, :, :int(round(numFrames*4))]
    return np.array(mix_audio_mag), np.array(mix_audio_phase), np.array(orig_audio_mag), np.array(orig_audio_phase), audioLength

def load_visual(data, dataPath, numFrames, visualAug, evalTalkies=False): 
    dataName = data[0]
    if evalTalkies:
        videoName = data[0][:25]
        #videoName = 'col'
    else:
        videoName = data[0][:11]
        #videoName = 'col'
    faceFolderPath = os.path.join(dataPath, videoName, dataName)
    faceFiles = glob.glob("%s/*.jpg"%faceFolderPath)
    sortedFaceFiles = sorted(faceFiles, key=lambda data: (float(data.split('/')[-1][:-4])), reverse=False) 
    faces = []
    H = 112
    if visualAug == True:
        new = int(H*random.uniform(0.7, 1))
        x, y = numpy.random.randint(0, H - new), numpy.random.randint(0, H - new)
        M = cv2.getRotationMatrix2D((H/2,H/2), random.uniform(-15, 15), 1)
        augType = random.choice(['orig', 'flip', 'crop', 'rotate']) 
    else:
        augType = 'orig'
    for faceFile in sortedFaceFiles[:numFrames]:
        face = cv2.imread(faceFile)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(face, (H,H))
        if augType == 'orig':
            faces.append(face)
        elif augType == 'flip':
            faces.append(cv2.flip(face, 1))
        elif augType == 'crop':
            faces.append(cv2.resize(face[y:y+new, x:x+new] , (H,H))) 
        elif augType == 'rotate':
            faces.append(cv2.warpAffine(face, M, (H,H)))
    faces = numpy.array(faces)
    return faces


def load_label(data, numFrames):
    res = []
    labels = data[3].replace('[', '').replace(']', '')
    labels = labels.split(',')
    for label in labels:
        res.append(int(label))
        if res[-1] == 2:
            res[-1] = 0
    res = numpy.array(res[:numFrames])
    return res

#Returns noise label indices. The noise label indices are CLEAN_SPEECH : 0, SPEECH_WITH_MUSIC : 1, SPEECH_WITH_NOISE : 2, NO_SPEECH : 3
def load_imageNoiseLabelIndices(data, numFrames, evalTalkies=False, numTalkies=0):
    if evalTalkies:
        res = numpy.ones((1,numTalkies))
    else:
        res = []
        labels = data[5].split(',')
        for label in labels:
            if label.__eq__('CLEAN_SPEECH'):
                res.append(0)
            elif label.__eq__('SPEECH_WITH_MUSIC') or label.__eq__('SPEECH_WITH_NOISE'):
                res.append(1)
            elif label.__eq__('NO_SPEECH'):
                res.append(2)
            else:
                print('invalid label during generation of noise label indices')

        res = numpy.array(res[:numFrames])
    return res

class train_loader(object):
    def __init__(self, trialFileName, audioPath, visualPath, batchSize, dataPathAVA, noisePairsTrainFile, audiosetDirPath, stft_frame, stft_hop_sec, samplingRate, noiseLabelsTrainFile, **kwargs):
        
        self.audioPath  = audioPath
        self.visualPath = visualPath
        self.audiosetDirPath = audiosetDirPath

        self.stft_frame = stft_frame
        self.stft_hop_sec = stft_hop_sec
        self.samplingRate = samplingRate
        
        self.miniBatch = []      
        mixLst = open(trialFileName).read().splitlines()
        
        self.imageNoiseLabelsFilePath = os.path.join(dataPathAVA, 'csv', 'train_noiseLabelLoader.csv')
        self.imageNoiseLabelsDict = {}
        with open(self.imageNoiseLabelsFilePath) as csvfile:
            csvreader = csv.reader(csvfile)
            for row in csvreader:
                entityID = row[0]
                numFrames = int(row[1])
                imageFrameNoiseLabels = row[2].split(',')
                imageFrameNoiseLabels = [elem.strip() for elem in imageFrameNoiseLabels]
                self.imageNoiseLabelsDict[entityID] = imageFrameNoiseLabels
        for i in range(len(mixLst)):
            entityID = mixLst[i].split('\t')[0]
            mixLst[i] = mixLst[i] + '\t' + ','.join(self.imageNoiseLabelsDict[entityID])
        
        # sort the training set by the length of the videos, shuffle them to make more videos in the same batch belong to different movies
        sortedMixLst = sorted(mixLst, key=lambda data: (int(data.split('\t')[1]), int(data.split('\t')[4])), reverse=True)         
        start = 0        
        while True:
          length = int(sortedMixLst[start].split('\t')[1])
          end = min(len(sortedMixLst), start + max(int(batchSize / length), 1))
          self.miniBatch.append(sortedMixLst[start:end])
          if end == len(sortedMixLst):
              break
          start = end    

        #Construct noise pairs dict
        f = open(noisePairsTrainFile)
        reader = csv.reader(f)
        self.noisePairsDict = {}
        for row in reader:
            sourceFileLocalPath = row[0]
            key = os.path.join(dataPathAVA, sourceFileLocalPath)
            self.noisePairsDict[key] = row[1:]

    def __getitem__(self, index):
        batchList    = self.miniBatch[index]
        numFrames   = int(batchList[-1].split('\t')[1])

        audioFeatures, visualFeatures, labels, imageFrameNoiseLabelIndices = [], [], [], []
        audioSet, audioSetOrig = generate_audio_set_noisy(self.audioPath, batchList, self.noisePairsDict, self.audiosetDirPath, self.samplingRate)
        mix_audio_magList, mix_audio_phaseList, orig_audio_magList, orig_audio_phaseList = [], [], [], []
        for line in batchList:
            data = line.replace(":", "_").split('\t')  
            mix_audio_mag, mix_audio_phase, orig_audio_mag, orig_audio_phase, _ = load_audio(data, self.audioPath, numFrames, self.stft_frame, self.stft_hop_sec, self.samplingRate, audioAug = False, audioSet = audioSet, audioSetOrig = audioSetOrig)
            mix_audio_magList.append(mix_audio_mag)
            mix_audio_phaseList.append(mix_audio_phase)
            orig_audio_magList.append(orig_audio_mag)
            orig_audio_phaseList.append(orig_audio_phase)
            visualFeatures.append(load_visual(data, self.visualPath,numFrames, visualAug = True))
            labels.append(load_label(data, numFrames))
            imageFrameNoiseLabelIndices.append(load_imageNoiseLabelIndices(data, numFrames))
        return torch.FloatTensor(numpy.array(mix_audio_magList)), torch.FloatTensor(numpy.array(mix_audio_phaseList)), torch.FloatTensor(numpy.array(orig_audio_magList)), torch.FloatTensor(numpy.array(orig_audio_phaseList)), torch.FloatTensor(numpy.array(visualFeatures)), torch.LongTensor(numpy.array(labels)), torch.LongTensor(np.array(imageFrameNoiseLabelIndices))

    def __len__(self):
        return len(self.miniBatch)


class val_loader(object):
    def __init__(self, trialFileName, audioPathOrig, visualPath, dataPathAVA, stft_frame, stft_hop_sec, samplingRate, noisePairsValFile, audiosetDirPath, alpha, **kwargs):
        self.audioPathOrig  = audioPathOrig
        self.visualPath = visualPath
        self.stft_frame = stft_frame
        self.stft_hop_sec = stft_hop_sec
        self.samplingRate = samplingRate
        self.audiosetDirPath = audiosetDirPath
        self.alpha = alpha

        self.miniBatch = open(trialFileName).read().splitlines()

        self.imageNoiseLabelsFilePath = os.path.join(dataPathAVA, 'csv', 'val_noiseLabelLoader.csv')
        self.imageNoiseLabelsDict = {}
        with open(self.imageNoiseLabelsFilePath) as csvfile:
            csvreader = csv.reader(csvfile)
            for row in csvreader:
                entityID = row[0]
                numFrames = int(row[1])
                imageFrameNoiseLabels = row[2].split(',')
                imageFrameNoiseLabels = [elem.strip() for elem in imageFrameNoiseLabels]
                self.imageNoiseLabelsDict[entityID] = imageFrameNoiseLabels
        for i in range(len(self.miniBatch)):
            entityID = self.miniBatch[i].split('\t')[0]
            self.miniBatch[i] = self.miniBatch[i] + '\t' + ','.join(self.imageNoiseLabelsDict[entityID])
        
        #Construct noise pairs dict
        f = open(noisePairsValFile)
        reader = csv.reader(f)
        self.noisePairsDict = {}
        for row in reader:
            sourceFileLocalPath = row[0]
            key = os.path.join(dataPathAVA, sourceFileLocalPath)
            self.noisePairsDict[key] = row[1:]

    def __getitem__(self, index):
        line       = [self.miniBatch[index]]
        numFrames  = int(line[0].split('\t')[1])

        fps = float(line[0].split('\t')[2])        
        audioSetOrig, audioSetNoisy   = generate_audio_set_eval(self.audioPathOrig, line, self.noisePairsDict, self.audiosetDirPath, self.alpha)        
        data = line[0].replace(":", "_").split('\t')
        mix_audio_mag, mix_audio_phase, orig_audio_mag, orig_audio_phase, audioLength = load_audio(data, self.audioPathOrig, numFrames, self.stft_frame, self.stft_hop_sec, self.samplingRate, audioAug = False,  audioSet = audioSetNoisy, audioSetOrig = audioSetOrig)
        mix_audio_magList = [mix_audio_mag]
        mix_audio_phaseList = [mix_audio_phase]
        orig_audio_magList = [orig_audio_mag]
        orig_audio_phaseList = [orig_audio_phase]

        visualFeatures = [load_visual(data, self.visualPath,numFrames, visualAug = False)]
        
        labels = [load_label(data, numFrames)]     
        return torch.FloatTensor(numpy.array(mix_audio_magList)), torch.FloatTensor(numpy.array(mix_audio_phaseList)), torch.FloatTensor(numpy.array(orig_audio_magList)), torch.FloatTensor(numpy.array(orig_audio_phaseList)), torch.FloatTensor(numpy.array(visualFeatures)), torch.LongTensor(numpy.array(labels)), torch.FloatTensor([fps]), torch.IntTensor([audioLength])

    def __len__(self):
        return len(self.miniBatch)
