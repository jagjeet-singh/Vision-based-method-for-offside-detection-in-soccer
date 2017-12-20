# USAGE
# python deep_learning_object_detection.py --image images/example_01.jpg \
#	--prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel
#       --side L/R

# import the necessary packages
import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt
import utils
from scipy.cluster.vq import kmeans2,vq,whiten
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import KMeans

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-s", "--side", required=True,
        help="goal post side")
ap.add_argument("-c", "--confidence", type=float, default=0.05,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"person", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

'''"background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor '''
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

def centroid_histogram(clt):
	# grab the number of different clusters and create a histogram
	# based on the number of pixels assigned to each cluster
	numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
	(hist, _) = np.histogram(clt.labels_, bins = numLabels)
 
	# normalize the histogram, such that it sums to one
	hist = hist.astype("float")
	hist /= hist.sum()
 
	# return the histogram
	return hist

# FOR Histogram of Gradients

winSize = (64,64)
blockSize = (16,16)
blockStride = (8,8)
cellSize = (8,8)
nbins = 9
derivAperture = 1
winSigma = 4.
histogramNormType = 0
L2HysThreshold = 2.0000000000000001e-01
gammaCorrection = 0
nlevels = 64
hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                        histogramNormType,L2HysThreshold,gammaCorrection,nlevels)


def getColorHog(rgb_image):
	## Inputs ## 
	# RGB image
	
	## Outputs ##
	# colored histogram of size (3*324)x1

	color_hog = [];
	resized_image = cv2.resize(rgb_image, (64, 64))


        chans = cv2.split(resized_image)
        colors = ("b", "g", "r")
	
        for (chan, color) in zip(chans, colors):
                print color
                print chan.shape

                hist = hog.compute(chan)
                print hist
                print min(hist)
                #hist = hist/min(hist)
                color_hog.extend(hist)

        print len(color_hog)
	
	return color_hog

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
goal_post = args["side"]

# load the input image and construct an input blob for the image
# by resizing to a fixed 300x300 pixels and then normalizing it
# (note: normalization is done via the authors of the MobileNet SSD
# implementation)

r = 300;


image = cv2.imread(args["image"])


        
(h, w) = image.shape[:2]

if(w < 1000):
        r = 1000;


print h,w,r
blob = cv2.dnn.blobFromImage(cv2.resize(image, (r, r)), 0.007843, (r,r), 127.5)
# pass the blob through the network and obtain the detections and
# predictions
print("[INFO] computing object detections...")
net.setInput(blob)
detections = net.forward()

print detections
print detections.shape[2]

# loop for the clustering

dictionarySize = 2

#BOW = cv2.BOWKMeansTrainer(dictionarySize)
features = []
#Z = 
for i in np.arange(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    featuresrgb = []
    idx = int(detections[0, 0, i, 1])
    if confidence > args["confidence"] and ((idx == 15) or (idx == 5)):
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        new_image_str = "image"+str(i)
        new_image = image[startY:endY,startX:endX].copy()
        cv2.imwrite( new_image_str+".jpg", new_image )
        clt = KMeans(n_clusters = 2);
  #      new_image = cv2.resize(new_image, (64, 64))
        new_image = new_image.reshape((new_image.shape[0] * new_image.shape[1], 3))
        clt.fit(new_image);
        hist = centroid_histogram(clt)
        index_min = np.argmin(hist)
        #bar = utils.plot_colors(hist, clt.cluster_centers_)
        features.append(np.floor(clt.cluster_centers_[index_min]))
        #plt.figure()
        #plt.axis("off")
        #plt.imshow(bar)
        #plt.show()
        #color_hog = getColorHog(new_image);
        #chans = cv2.split(new_image)
        #colors = ("b", "g", "r")
        #hist = cv2.calcHist(new_image[:,:,0], [0], None, [32], [0, 256])
        #featuresrgb.extend(hist)
        #hist = cv2.calcHist(new_image[:,:,2], [0], None, [32], [0, 256])
        #featuresrgb.extend(hist)

        #x = np.arange(5292)

        #for (chan, color) in zip(chans, colors):
                # create a histogram for the current channel and
                # concatenate the resulting histograms for each
                # channel

        #        hist = cv2.calcHist([chan], [0], None, [32], [0, 256])
        #        featuresrgb.extend(hist)
        #        y = np.array(featuresrgb)
        #y = np.array(color_hog)
        #print y.shape
        #plt.bar(x, y[:,0])
        #plt.ylim(0,1000)
        #plt.savefig('figHist'+str(i))
        #plt.show()
        
        #features.append(np.array(color_hog))

#print features
features = np.array(features)
k=2
Z = linkage(features)
cluster_id = fcluster(Z,k,criterion='maxclust')
#features = np.transpose(features)
#print features.shape

#features = features[0,:,:].T
#print 'features shape : ' + str(features.shape)
#whitened = whiten(features)
#print '##### kmeans starting #####'
#centroid, cluster_id = kmeans2(whitened,2)
#print '##### kmeans completed ####'
#print cluster_id.shape
for i in range(0,cluster_id.shape[0]):
    print 'Person ' + str(i+1) + ' : ' + str(cluster_id[i]) + ' : ' + str(detections[0, 0, i, 2])

cluster_idx = 0
if goal_post == 'L':
    minX_cluster0 = w
    minX_cluster1 = w
else :
    maxX_cluster0 = 0
    maxX_cluster1 = 0

cluster0_mem = 0
cluster1_mem = 0
    
# loop over the detections
for i in np.arange(0, detections.shape[2]):
	# extract the confidence (i.e., probability) associated with the
	# prediction
	
    idx = int(detections[0,0,i,1])
    confidence = detections[0, 0, i, 2]
	# filter out weak detections by ensuring the `confidence` is
	# greater than the minimum confidence
    if confidence > args["confidence"] and ((idx == 15) or (idx == 5)):
	# extract the index of the class label from the `detections`,
	# then compute the (x, y)-coordinates of the bounding box for
	# the object
	box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
	(startX, startY, endX, endY) = box.astype("int")

        if cluster_id[cluster_idx] == 1 :
            cluster0_mem = cluster0_mem + 1

        if cluster_id[cluster_idx] == 2 :
            cluster1_mem = cluster1_mem + 1
            
	if (goal_post == 'L'):
            print "entered L"
            # check for min startX
            if (cluster_id[cluster_idx] == 1) :
                if (minX_cluster0 > startX) :
                    minX_cluster0 = startX
            if (cluster_id[cluster_idx] == 2) :
                if (minX_cluster1 > startX) :
                    minX_cluster1 = startX
        else:
            print "entered R"
            # check for min startX
            if (cluster_id[cluster_idx] == 1) :
                if (maxX_cluster0 < startX) :
                    maxX_cluster0 = startX
            if (cluster_id[cluster_idx] == 2) :
                if (maxX_cluster1 < startX) :
                    maxX_cluster1 = startX

	# display the prediction
	label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
	print("[INFO] {}".format(label))
	cv2.rectangle(image, (startX, startY), (endX, endY),
		COLORS[idx], 2)
	y = startY - 15 if startY - 15 > 15 else startY + 15
	cv2.putText(image, str(cluster_id[cluster_idx]), (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
        #cv2.putText(image, label, (startX, y),
	#	cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
        cluster_idx = cluster_idx + 1

cluster_idx = 0
# loop over the detections
for i in np.arange(0, detections.shape[2]):
	# extract the confidence (i.e., probability) associated with the
	# prediction
	
    idx = int(detections[0,0,i,1])
    confidence = detections[0, 0, i, 2]
	# filter out weak detections by ensuring the `confidence` is
	# greater than the minimum confidence
    if confidence > args["confidence"] and ((idx == 15) or (idx == 5)):
	# extract the index of the class label from the `detections`,
	# then compute the (x, y)-coordinates of the bounding box for
	# the object
	box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
	(startX, startY, endX, endY) = box.astype("int")

	if (goal_post == 'L'):
            print "entered L"
            # check for min startX
            if (cluster_id[cluster_idx] == 1) :
                if (cluster0_mem >= cluster1_mem) :
                    continue;
                if (minX_cluster0 != startX) :
                    continue;
            if (cluster_id[cluster_idx] == 2) :
                if (cluster1_mem > cluster0_mem) :
                    continue;
                if (minX_cluster1 != startX) :
                    continue;
        else:
            print "entered R"
            # check for min startX
            if (cluster_id[cluster_idx] == 1) :
                if (cluster0_mem >= cluster1_mem) :
                    continue;                
                if (maxX_cluster0 != startX) :
                    continue;
            if (cluster_id[cluster_idx] == 2) :
                if (cluster1_mem > cluster0_mem) :
                    continue;                    
                if (maxX_cluster1 != startX) :
                    continue

	# display the prediction
	roi = image[startY:endY,startX:endX]
	roi[:,:,2] = 0
	#cv2.rectangle(image, (startX, startY), (endX, endY),
	#	COLORS[idx], -1)
        cluster_idx = cluster_idx + 1
        
# LOGIC for offside
offside = ""
if cluster1_mem >= cluster0_mem :
    if (goal_post == 'L'):
        if (minX_cluster0 < minX_cluster1) and (cluster0_mem != 0) :
            offside = "OFFSIDE"
        else :
            offside = "NOT OFFSIDE"
    if (goal_post == 'R'):
        if (maxX_cluster0 > maxX_cluster1) and (cluster0_mem != 0) :
            offside = "OFFSIDE"
        else :
            offside = "NOT OFFSIDE"        
else :
    if (goal_post == 'L'):
        if (minX_cluster1 < minX_cluster0) and (cluster1_mem != 0) :
            offside = "OFFSIDE"
        else :
            offside = "NOT OFFSIDE"
    if (goal_post == 'R'):
        if (maxX_cluster1 > maxX_cluster0) and (cluster1_mem != 0) :
            offside = "OFFSIDE"
        else :
            offside = "NOT OFFSIDE"         

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (200, 200)
fontScale              = 1
fontColor              = (255,0,0)
lineType               = 2

cv2.putText(image,offside, 
    bottomLeftCornerOfText, 
    font, 
    fontScale,
    fontColor,
    lineType)

(h, w) = image.shape[:2]
# show the output image
print h,w
cv2.imshow("Output", image)
cv2.waitKey(0)
