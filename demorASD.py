import sys, time, os, tqdm, torch, argparse, glob, subprocess, warnings, cv2, pickle, numpy, pdb, math, python_speech_features
import librosa
from scipy import signal
from shutil import rmtree
from scipy.io import wavfile
from scipy.interpolate import interp1d
from sklearn.metrics import accuracy_score, f1_score

from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector

from model.faceDetector.s3fd import S3FD
from rASD import rASD_Model

import numpy as np
import torch.nn.functional as F
from operator import add

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description = "TalkNet Demo or Columnbia ASD Evaluation")

parser.add_argument('--videoName',             type=str, default="001",   help='Demo video name')
parser.add_argument('--videoFolder',           type=str, default="demo",  help='Path for inputs, tmps and outputs')
parser.add_argument('--pretrainModel',         type=str, default="exps/exp1/model/model",   help='Path for the pretrained TalkNet model')

parser.add_argument('--nDataLoaderThread',     type=int,   default=10,   help='Number of workers')
parser.add_argument('--facedetScale',          type=float, default=0.25, help='Scale factor for face detection, the frames will be scale to 0.25 orig')
parser.add_argument('--minTrack',              type=int,   default=10,   help='Number of min frames for each shot')
parser.add_argument('--numFailedDet',          type=int,   default=10,   help='Number of missed detections allowed before tracking is stopped')
parser.add_argument('--minFaceSize',           type=int,   default=1,    help='Minimum face size in pixels')
parser.add_argument('--cropScale',             type=float, default=0.40, help='Scale bounding box')

parser.add_argument('--start',                 type=int, default=0,   help='The start time of the video')
parser.add_argument('--duration',              type=int, default=0,  help='The duration of the video, when set as 0, will extract the whole video')

parser.add_argument('--evalCol',               dest='evalCol', action='store_true', help='Evaluate on Columnbia dataset')
parser.add_argument('--colSavePath',           type=str, default="saveVideo",  help='Path for inputs, tmps and outputs')

args = parser.parse_args()

# if os.path.isfile(args.pretrainModel) == False: # Download the pretrained model
#     Link = "1AbN9fCf9IexMxEKXLQY2KYBlb-IhSEea"
#     cmd = "gdown --id %s -O %s"%(Link, args.pretrainModel)
#     subprocess.call(cmd, shell=True, stdout=None)

if args.evalCol == True:
	# The process is: 1. download video and labels(I have modified the format of labels to make it easiler for using)
	# 	              2. extract audio, extract video frames
	#                 3. scend detection, face detection and face tracking
	#                 4. active speaker detection for the detected face clips
	#                 5. use iou to find the identity of each face clips, compute the F1 results
	# The step 1 to 3 will take some time (That is one-time process). It depends on your cpu and gpu speed. For reference, I used 1.5 hour
	# The step 4 and 5 need less than 10 minutes
	# Need about 20G space finally
	# ```
	args.videoName = 'col'
	args.videoFolder = args.colSavePath
	args.savePath = os.path.join(args.videoFolder, args.videoName)
	args.videoPath = os.path.join(args.videoFolder, args.videoName + '.mp4')
	args.duration = 0
	if os.path.isfile(args.videoPath) == False:  # Download video
		link = 'https://www.youtube.com/watch?v=6GzxbrO0DHM&t=2s'
		#cmd = "youtube-dl -f best -o %s '%s'"%(args.videoPath, link)
		cmd = "yt-dlp -f best -o %s '%s'"%(args.videoPath, link)
		output = subprocess.call(cmd, shell=True, stdout=None)
	if os.path.isdir(args.videoFolder + '/col_labels') == False: # Download label
		link = "1Tto5JBt6NsEOLFRWzyZEeV6kCCddc6wv"
		cmd = "gdown --id %s -O %s"%(link, args.videoFolder + '/col_labels.tar.gz')
		subprocess.call(cmd, shell=True, stdout=None)
		cmd = "tar -xzvf %s -C %s"%(args.videoFolder + '/col_labels.tar.gz', args.videoFolder)
		subprocess.call(cmd, shell=True, stdout=None)
		os.remove(args.videoFolder + '/col_labels.tar.gz')	
else:
	args.videoPath = glob.glob(os.path.join(args.videoFolder, args.videoName + '.*'))[0]
	args.savePath = os.path.join(args.videoFolder, args.videoName)

def scene_detect(args):
	# CPU: Scene detection, output is the list of each shot's time duration
	print('video file path ', args.videoFilePath)
	videoManager = VideoManager([args.videoFilePath])
	statsManager = StatsManager()
	sceneManager = SceneManager(statsManager)
	sceneManager.add_detector(ContentDetector())
	baseTimecode = videoManager.get_base_timecode()
	videoManager.set_downscale_factor()
	videoManager.start()
	sceneManager.detect_scenes(frame_source = videoManager)
	sceneList = sceneManager.get_scene_list(baseTimecode)
	savePath = os.path.join(args.pyworkPath, 'scene.pckl')
	# fil = open(savePath, 'rb')
	# sceneList = pickle.load(fil)
	if sceneList == []:
		sceneList = [(videoManager.get_base_timecode(),videoManager.get_current_timecode())]
	with open(savePath, 'wb') as fil:
		pickle.dump(sceneList, fil)
	sys.stderr.write('%s - scenes detected %d\n'%(args.videoFilePath, len(sceneList)))
	return sceneList

def inference_video(args):
	# GPU: Face detection, output is the list contains the face location and score in this frame
	DET = S3FD(device='cuda')
	flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg'))
	flist.sort()
	dets = []
	for fidx, fname in enumerate(flist):
		image = cv2.imread(fname)
		imageNumpy = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		bboxes = DET.detect_faces(imageNumpy, conf_th=0.9, scales=[args.facedetScale])
		print('bboxes: ', bboxes)
		dets.append([])
		for bbox in bboxes:
			dets[-1].append({'frame':fidx, 'bbox':(bbox[:-1]).tolist(), 'conf':bbox[-1]}) # dets has the frames info, bbox info, conf info
		sys.stderr.write('%s-%05d; %d dets\r' % (args.videoFilePath, fidx, len(dets[-1])))
	savePath = os.path.join(args.pyworkPath,'faces.pckl')
	# fil = open(savePath, 'rb')
	# dets = pickle.load(fil)
	
	with open(savePath, 'wb') as fil:
		pickle.dump(dets, fil)
	return dets

def bb_intersection_over_union(boxA, boxB, evalCol = False):
	# CPU: IOU Function to calculate overlap between two image
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	interArea = max(0, xB - xA) * max(0, yB - yA)
	boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
	boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
	if evalCol == True:
		iou = interArea / float(boxAArea)
	else:
		iou = interArea / float(boxAArea + boxBArea - interArea)
	return iou

def track_shot(args, sceneFaces):
	# CPU: Face tracking
	iouThres  = 0.5     # Minimum IOU between consecutive face detections
	tracks    = []
	while True:
		track     = []
		for frameFaces in sceneFaces:
			for face in frameFaces:
				if track == []:
					track.append(face)
					frameFaces.remove(face)
				elif face['frame'] - track[-1]['frame'] <= args.numFailedDet:
					iou = bb_intersection_over_union(face['bbox'], track[-1]['bbox'])
					if iou > iouThres:
						track.append(face)
						frameFaces.remove(face)
						continue
				else:
					break
		if track == []:
			break
		elif len(track) > args.minTrack:
			frameNum    = numpy.array([ f['frame'] for f in track ])
			bboxes      = numpy.array([numpy.array(f['bbox']) for f in track])
			frameI      = numpy.arange(frameNum[0],frameNum[-1]+1)
			bboxesI    = []
			for ij in range(0,4):
				interpfn  = interp1d(frameNum, bboxes[:,ij])
				bboxesI.append(interpfn(frameI))
			bboxesI  = numpy.stack(bboxesI, axis=1)
			if max(numpy.mean(bboxesI[:,2]-bboxesI[:,0]), numpy.mean(bboxesI[:,3]-bboxesI[:,1])) > args.minFaceSize:
				tracks.append({'frame':frameI,'bbox':bboxesI})
	return tracks

def crop_video(args, track, cropFile):
	# CPU: crop the face clips
	flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg')) # Read the frames
	flist.sort()
	vOut = cv2.VideoWriter(cropFile + 't.avi', cv2.VideoWriter_fourcc(*'XVID'), 25, (224,224))# Write video
	dets = {'x':[], 'y':[], 's':[]}
	for det in track['bbox']: # Read the tracks
		dets['s'].append(max((det[3]-det[1]), (det[2]-det[0]))/2) 
		dets['y'].append((det[1]+det[3])/2) # crop center x 
		dets['x'].append((det[0]+det[2])/2) # crop center y
	dets['s'] = signal.medfilt(dets['s'], kernel_size=13)  # Smooth detections 
	dets['x'] = signal.medfilt(dets['x'], kernel_size=13)
	dets['y'] = signal.medfilt(dets['y'], kernel_size=13)
	for fidx, frame in enumerate(track['frame']):
		cs  = args.cropScale
		bs  = dets['s'][fidx]   # Detection box size
		bsi = int(bs * (1 + 2 * cs))  # Pad videos by this amount 
		image = cv2.imread(flist[frame])
		frame = numpy.pad(image, ((bsi,bsi), (bsi,bsi), (0, 0)), 'constant', constant_values=(110, 110))
		my  = dets['y'][fidx] + bsi  # BBox center Y
		mx  = dets['x'][fidx] + bsi  # BBox center X
		face = frame[int(my-bs):int(my+bs*(1+2*cs)),int(mx-bs*(1+cs)):int(mx+bs*(1+cs))]
		vOut.write(cv2.resize(face, (224, 224)))
	audioTmp    = cropFile + '.wav'
	audioStart  = (track['frame'][0]) / 25
	audioEnd    = (track['frame'][-1]+1) / 25
	vOut.release()
	command = ("ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 -threads %d -ss %.3f -to %.3f %s -loglevel panic" % \
		      (args.audioFilePath, args.nDataLoaderThread, audioStart, audioEnd, audioTmp)) 
	output = subprocess.call(command, shell=True, stdout=None) # Crop audio file
	_, audio = wavfile.read(audioTmp)
	command = ("ffmpeg -y -i %st.avi -i %s -threads %d -c:v copy -c:a copy %s.avi -loglevel panic" % \
			  (cropFile, audioTmp, args.nDataLoaderThread, cropFile)) # Combine audio and video file
	output = subprocess.call(command, shell=True, stdout=None)
	os.remove(cropFile + 't.avi')
	return {'track':track, 'proc_track':dets}

def extract_MFCC(file, outPath):
	# CPU: extract mfcc
	sr, audio = wavfile.read(file)
	mfcc = python_speech_features.mfcc(audio,sr) # (N_frames, 13)   [1s = 100 frames]
	featuresPath = os.path.join(outPath, file.split('/')[-1].replace('.wav', '.npy'))
	numpy.save(featuresPath, mfcc)

def warpgrid(bs, HO, WO, warp=True):
    # meshgrid
    x = np.linspace(-1, 1, WO)
    y = np.linspace(-1, 1, HO)
    xv, yv = np.meshgrid(x, y)
    grid = np.zeros((bs, HO, WO, 2))
    grid_x = xv
    if warp:
        grid_y = (np.power(21, (yv+1)/2) - 11) / 10
    else:
        grid_y = np.log(yv * 10 + 11) / np.log(21) * 2 - 1
    grid[:, :, :, 0] = grid_x
    grid[:, :, :, 1] = grid_y
    grid = grid.astype(np.float32)
    return grid

def evaluate_network(files, args):
	# GPU: active speaker detection by pretrained TalkNet
	s = rASD_Model()
	s.loadParameters(args.pretrainModel)
	#sys.stderr.write("Model %s loaded from previous state! \r\n"%args.pretrainModel)
	s.eval()
	allLabels = []
	durationSet = {1,2,3,4,6} # To make the result more reliable
	# durationSet = {1,1,1,2,2,2,3,3,4,5,6} # Use this line can get more reliable result
	#durationSet = {1}
	for file in tqdm.tqdm(files, total = len(files)):
		fileName = os.path.splitext(file.split('/')[-1])[0] # Load audio and video

		video = cv2.VideoCapture(os.path.join(args.pycropPath, fileName + '.avi'))
		videoFeature = []
		imageFrameCount=0
		while video.isOpened():
			ret, frames = video.read()
			if ret == True:
				face = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
				face = cv2.resize(face, (224,224))
				face = face[int(112-(112/2)):int(112+(112/2)), int(112-(112/2)):int(112+(112/2))]
				videoFeature.append(face)
				imageFrameCount=imageFrameCount+1
			else:
				break
		video.release()
		videoFeature = numpy.array(videoFeature)

		#_, audio = wavfile.read(os.path.join(args.pycropPath, fileName + '.wav'))
		audio, samplingRate = librosa.load(os.path.join(args.pycropPath, fileName + '.wav'), sr=16000)
		#print('sampling rate ', samplingRate)
		#audioFeature = python_speech_features.mfcc(audio, 16000, numcep = 13, winlen = 0.025, winstep = 0.010)
		spectro = librosa.core.stft(audio, hop_length=int(samplingRate*0.01), n_fft=1022, center=True)
		audioFeature, _ = librosa.core.magphase(spectro)
		# audioFeature = np.expand_dims(np.expand_dims(audioFeature, axis=0), axis=0)
		audioFeature = np.expand_dims(audioFeature, axis=0)
		maxAudioFrames = int(round(imageFrameCount*4))
		if audioFeature.shape[-1] < maxAudioFrames:
			shortage = maxAudioFrames - audioFeature.shape[-1]
			audioFeature = numpy.pad(audioFeature, ((0,0), (0, 0), (0,shortage)), 'wrap')
		#audioFeature = audioFeature[:,:,:,0:-2]
		audioFeature = audioFeature[:, :, :maxAudioFrames]
		audioFeature = np.expand_dims(audioFeature, axis=0)
		# audioFeature = torch.FloatTensor(audioFeature)

		# print('size of the audio feature ', audioFeature.size)

		# B = audioFeature.size(0)
		# T = audioFeature.size(3)

		# audioFeature = audioFeature + 1e-10
		# grid_warp = torch.from_numpy(warpgrid(B, 256, T, warp=True)).cuda()
		# audioFeature = F.grid_sample(audioFeature.cuda(), grid_warp)

		#print('audio feature shape ', audioFeature.shape)

		# audioFeature = audioFeature.cpu().numpy()

		
		#print('video feature shape ', videoFeature.shape)
		# length = min((audioFeature.shape[-1] - audioFeature.shape[-1] % 4) / 100, videoFeature.shape[0])
		length = min(int(audioFeature.shape[-1]/100), int(videoFeature.shape[0]/25))

		#print('length ', length)
		# audioFeature = audioFeature[:, :, :, :int(round(length * 100))]
		# videoFeature = videoFeature[:int(round(length * 25)),:,:]
		#print('audio feature shape after length ', audioFeature.shape)
		#print('video feature shape after length ', videoFeature.shape)

		allLabel = [] # Evaluation use TalkNet
		for duration in durationSet:
			batchSize = int(math.ceil(length / duration))
			#print(duration, batchSize)
			labels = []
			with torch.no_grad():
				for i in range(batchSize):
					inputA = torch.FloatTensor(audioFeature[:, :, :, i * duration * 100:(i+1) * duration * 100])
					inputV = torch.FloatTensor(videoFeature[i * duration * 25: (i+1) * duration * 25,:,:]).unsqueeze(0).cuda()
					#print('duration '+str(duration)+' batch size '+str(batchSize)+' inputA shape ', inputA.shape)
					#print('duration '+str(duration)+' batch size '+str(batchSize)+' inputV shape ', inputV.shape)

					#embedA = s.model.forward_audio_frontend(inputA)
					#print('input v shape ', inputV.shape)
					B = inputA.size(0)
					T = inputA.size(3)

					inputA = inputA + 1e-10
					grid_warp = torch.from_numpy(warpgrid(B, 256, T, warp=True)).cuda()
					inputA = F.grid_sample(inputA.cuda(), grid_warp)
					shortage=0
					if inputV.shape[1] < 5:
						shortage = 5 - inputV.shape[1]
						inputV = torch.nn.functional.pad(inputV, (0, shortage, 0,0,0,0))
						inputA = torch.nn.functional.pad(inputA, (0, 4*shortage))
					embedV, embedV512 = s.model.forward_visual_frontend(inputV)
					if shortage > 0:
						embedV = embedV[:, 0:embedV.shape[1]-shortage, :]
						embedV512 = embedV512[:, 0:embedV512.shape[1]-shortage, :]
					#print('embed v shape ', embedV.shape)
					#print('embed v 512 shape ', embedV512.shape)
					embedVMean = torch.mean(embedV512, dim=1, keepdim=True)
					embedVMean = embedVMean.view(embedVMean.shape[0]*embedVMean.shape[1], embedVMean.shape[2], 1, 1)
					audioFeatureLog = torch.log(inputA)
					mask_prediction, FEInput = s.net_unet(audioFeatureLog.cuda(), embedVMean.cuda())
					FEOutput = s.FEModel(FEInput)
					FEOutput = torch.squeeze(FEOutput, 2)
					embedA = FEOutput.transpose(1,2)

					#print('embed A shape ', embedA.shape)
					#print('embed V mean shape ', embedVMean.shape)
					if embedA.shape[1] != embedV.shape[1]:
						embedA = embedA[:, 0:embedV.shape[1], :]
					embedA, embedV = s.model.forward_cross_attention(embedA, embedV)
					out = s.model.forward_audio_visual_backend(embedA, embedV)
					label = s.lossAV.forward(out, labels = None)
					
					labels.extend(label)
			allLabel.append(labels)
		
		for i in range(len(durationSet)):
			if i==0:
				tmpList = allLabel[i]
			else:
				tmpList = list(map(add, tmpList, allLabel[i]))
		
		# scoreAvg = [x/len(durationSet) for x in tmpList]
		finalLabel = [1 if x>=1 else 0 for x in tmpList]
		# allScore = numpy.round((numpy.mean(numpy.array(allScore), axis = 0)), 1).astype(float)
		# #print('all score list ', allScore)
		# allScore1 = numpy.array(allScore)
		# #print('all score array ', allScore1)
		# #print(allScore1)
		# allScore2 = numpy.mean(allScore1, axis=0)
		# allScore3 = numpy.round(allScore2, 1)
		# allScore = allScore3.astype(float)

		allLabels.append(finalLabel)
	return allLabels

def visualization(tracks, labels, args):
	# CPU: visulize the result for video format
	flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg'))
	flist.sort()
	faces = [[] for i in range(len(flist))]
	for tidx, track in enumerate(tracks):
		label = labels[tidx]
		# print('label ', label)
		# print('label length: ', len(label))
		# print('list length: ', len(track['track']['frame'].tolist()))
		for fidx, frame in enumerate(track['track']['frame'].tolist()):
			s = label[max(fidx - 2, 0): min(fidx + 3, len(label) - 1)] # smoothing
			# print('s ', s)
			if len(s) < 3: #quick fix for empty lists. Check the reason for differences in length and fix it properly.
				# print('before extension: ', s)
				s.extend([0,0,0])
				# print('after extension: ', s)
			# print('before max: ', s)
			s = numpy.max(s)
			# print('after max: ', s)
			faces[frame].append({'track':tidx, 'label':int(s),'s':track['proc_track']['s'][fidx], 'x':track['proc_track']['x'][fidx], 'y':track['proc_track']['y'][fidx]})
	firstImage = cv2.imread(flist[0])
	fw = firstImage.shape[1]
	fh = firstImage.shape[0]
	vOut = cv2.VideoWriter(os.path.join(args.pyaviPath, 'video_only.avi'), cv2.VideoWriter_fourcc(*'XVID'), 25, (fw,fh))
	colorDict = {0: 0, 1: 255}
	for fidx, fname in tqdm.tqdm(enumerate(flist), total = len(flist)):
		image = cv2.imread(fname)
		for face in faces[fidx]:
			clr = colorDict[int((face['label'] == 1))]
			# txt = round(face['score'], 1)
			# if face['score'] >= 0:
			cv2.rectangle(image, (int(face['x']-face['s']), int(face['y']-face['s'])), (int(face['x']+face['s']), int(face['y']+face['s'])),(0,clr,255-clr),10)
			# cv2.putText(image,'%s'%(txt), (int(face['x']-face['s']), int(face['y']+face['s'])), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,clr,255-clr),5)
		vOut.write(image)
	vOut.release()
	command = ("ffmpeg -y -i %s -i %s -threads %d -c:v copy -c:a copy %s -loglevel panic" % \
		(os.path.join(args.pyaviPath, 'video_only.avi'), os.path.join(args.pyaviPath, 'audio.wav'), \
		args.nDataLoaderThread, os.path.join(args.pyaviPath,'video_out.avi'))) 
	output = subprocess.call(command, shell=True, stdout=None)

def evaluate_col_ASD(tracks, scores, args):
	txtPath = args.videoFolder + '/col_labels/fusion/*.txt' # Load labels
	predictionSet = {}
	for name in {'long', 'bell', 'boll', 'lieb', 'sick', 'abbas'}:
		predictionSet[name] = [[],[]]
	dictGT = {}
	txtFiles = glob.glob("%s"%txtPath)
	for file in txtFiles:
		lines = open(file).read().splitlines()
		idName = file.split('/')[-1][:-4]
		for line in lines:
			data = line.split('\t')
			frame = int(int(data[0]) / 29.97 * 25)
			x1 = int(data[1])
			y1 = int(data[2])
			x2 = int(data[1]) + int(data[3])
			y2 = int(data[2]) + int(data[3])
			gt = int(data[4])
			if frame in dictGT:
				dictGT[frame].append([x1,y1,x2,y2,gt,idName])
			else:
				dictGT[frame] = [[x1,y1,x2,y2,gt,idName]]	
	flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg')) # Load files
	flist.sort()
	faces = [[] for i in range(len(flist))]
	for tidx, track in enumerate(tracks):
		score = scores[tidx]				
		for fidx, frame in enumerate(track['track']['frame'].tolist()):
			s = numpy.mean(score[max(fidx - 2, 0): min(fidx + 3, len(score) - 1)]) # average smoothing
			faces[frame].append({'track':tidx, 'score':float(s),'s':track['proc_track']['s'][fidx], 'x':track['proc_track']['x'][fidx], 'y':track['proc_track']['y'][fidx]})
	for fidx, fname in tqdm.tqdm(enumerate(flist), total = len(flist)):
		if fidx in dictGT: # This frame has label
			for gtThisFrame in dictGT[fidx]: # What this label is ?
				faceGT = gtThisFrame[0:4]
				labelGT = gtThisFrame[4]
				idGT = gtThisFrame[5]
				ious = []
				for face in faces[fidx]: # Find the right face in my result
					faceLocation = [int(face['x']-face['s']), int(face['y']-face['s']), int(face['x']+face['s']), int(face['y']+face['s'])]
					faceLocation_new = [int(face['x']-face['s']) // 2, int(face['y']-face['s']) // 2, int(face['x']+face['s']) // 2, int(face['y']+face['s']) // 2]
					iou = bb_intersection_over_union(faceLocation_new, faceGT, evalCol = True)
					if iou > 0.5:
						ious.append([iou, round(face['score'],2)])
				if len(ious) > 0: # Find my result
					ious.sort()
					labelPredict = ious[-1][1]
				else:					
					labelPredict = 0
				x1 = faceGT[0]
				y1 = faceGT[1]
				width = faceGT[2] - faceGT[0]
				predictionSet[idGT][0].append(labelPredict)
				predictionSet[idGT][1].append(labelGT)
	names = ['long', 'bell', 'boll', 'lieb', 'sick', 'abbas'] # Evaluate
	names.sort()
	F1s = 0
	for i in names:
		scores = numpy.array(predictionSet[i][0])
		labels = numpy.array(predictionSet[i][1])
		scores = numpy.int64(scores > 0)
		F1 = f1_score(labels, scores)
		ACC = accuracy_score(labels, scores)
		if i != 'abbas':
			F1s += F1
			print("%s, ACC:%.2f, F1:%.2f"%(i, 100 * ACC, 100 * F1))
	print("Average F1:%.2f"%(100 * (F1s / 5)))	  

# Main function
def main():
	# This preprocesstion is modified based on this [repository](https://github.com/joonson/syncnet_python).
	# ```
	# .
	# ├── pyavi
	# │   ├── audio.wav (Audio from input video)
	# │   ├── video.avi (Copy of the input video)
	# │   ├── video_only.avi (Output video without audio)
	# │   └── video_out.avi  (Output video with audio)
	# ├── pycrop (The detected face videos and audios)
	# │   ├── 000000.avi
	# │   ├── 000000.wav
	# │   ├── 000001.avi
	# │   ├── 000001.wav
	# │   └── ...
	# ├── pyframes (All the video frames in this video)
	# │   ├── 000001.jpg
	# │   ├── 000002.jpg
	# │   └── ...	
	# └── pywork
	#     ├── faces.pckl (face detection result)
	#     ├── scene.pckl (scene detection result)
	#     ├── scores.pckl (ASD result)
	#     └── tracks.pckl (face tracking result)
	# ```

	# Initialization 
	args.pyaviPath = os.path.join(args.savePath, 'pyavi')
	args.pyframesPath = os.path.join(args.savePath, 'pyframes')
	args.pyworkPath = os.path.join(args.savePath, 'pywork')
	args.pycropPath = os.path.join(args.savePath, 'pycrop')
	# if os.path.exists(args.savePath):
	# 	rmtree(args.savePath)
	os.makedirs(args.pyaviPath, exist_ok = True) # The path for the input video, input audio, output video
	os.makedirs(args.pyframesPath, exist_ok = True) # Save all the video frames
	os.makedirs(args.pyworkPath, exist_ok = True) # Save the results in this process by the pckl method
	os.makedirs(args.pycropPath, exist_ok = True) # Save the detected face clips (audio+video) in this process

	# Extract video
	args.videoFilePath = os.path.join(args.pyaviPath, 'video.avi')
	#If duration did not set, extract the whole video, otherwise extract the video from 'args.start' to 'args.start + args.duration'
	if args.duration == 0:
		command = ("ffmpeg -y -i %s -qscale:v 2 -threads %d -async 1 -r 25 %s -loglevel panic" % \
			(args.videoPath, args.nDataLoaderThread, args.videoFilePath))
	else:
		command = ("ffmpeg -y -i %s -qscale:v 2 -threads %d -ss %.3f -to %.3f -async 1 -r 25 %s -loglevel panic" % \
			(args.videoPath, args.nDataLoaderThread, args.start, args.start + args.duration, args.videoFilePath))
	subprocess.call(command, shell=True, stdout=None)
	sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Extract the video and save in %s \r\n" %(args.videoFilePath))
	
	# Extract audio
	args.audioFilePath = os.path.join(args.pyaviPath, 'audio.wav')
	command = ("ffmpeg -y -i %s -qscale:a 0 -ac 1 -vn -threads %d -ar 16000 %s -loglevel panic" % \
		(args.videoFilePath, args.nDataLoaderThread, args.audioFilePath))
	subprocess.call(command, shell=True, stdout=None)
	sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Extract the audio and save in %s \r\n" %(args.audioFilePath))

	#Extract the video frames
	command = ("ffmpeg -y -i %s -vf fps=25 -qscale:v 2 -threads %d -f image2 %s -loglevel panic" % \
		(args.videoFilePath, args.nDataLoaderThread, os.path.join(args.pyframesPath, '%06d.jpg'))) 
	subprocess.call(command, shell=True, stdout=None)
	sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Extract the frames and save in %s \r\n" %(args.pyframesPath))

	# Scene detection for the video frames
	scene = scene_detect(args)
	sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Scene detection and save in %s \r\n" %(args.pyworkPath))	

	# Face detection for the video frames
	faces = inference_video(args)
	sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Face detection and save in %s \r\n" %(args.pyworkPath))

	# # Face tracking
	allTracks, vidTracks = [], []
	for shot in scene:
		if shot[1].frame_num - shot[0].frame_num >= args.minTrack: # Discard the shot frames less than minTrack frames
			allTracks.extend(track_shot(args, faces[shot[0].frame_num:shot[1].frame_num])) # 'frames' to present this tracks' timestep, 'bbox' presents the location of the faces
	sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Face track and detected %d tracks \r\n" %len(allTracks))

	# Face clips cropping
	for ii, track in tqdm.tqdm(enumerate(allTracks), total = len(allTracks)):
		vidTracks.append(crop_video(args, track, os.path.join(args.pycropPath, '%05d'%ii)))
	savePath = os.path.join(args.pyworkPath, 'tracks.pckl')
	with open(savePath, 'wb') as fil:
		pickle.dump(vidTracks, fil)
	sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Face Crop and saved in %s tracks \r\n" %args.pycropPath)
	fil = open(savePath, 'rb')
	vidTracks = pickle.load(fil)

	# Active Speaker Detection by TalkNet
	files = glob.glob("%s/*.avi"%args.pycropPath)
	files.sort()
	final_labels = evaluate_network(files, args)
	savePath = os.path.join(args.pyworkPath, 'final_labels.pckl')
	with open(savePath, 'wb') as fil:
		pickle.dump(final_labels, fil)
	sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Scores extracted and saved in %s \r\n" %args.pyworkPath)

	# if args.evalCol == True:
	# 	evaluate_col_ASD(vidTracks, scores, args) # The columnbia video is too big for visualization. You can still add the `visualization` funcition here if you want
	# 	quit()
	# else:
	# Visualization, save the result as the new video	
	visualization(vidTracks, final_labels, args)	

if __name__ == '__main__':
    main()
