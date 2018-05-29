from __future__ import print_function

import numpy as np
import cv2
import dlib
import matplotlib
import matplotlib.pyplot as plt
#matplotlib.use("TkAgg")

import skvideo.io

from data.video.video import Video
vpath = '/home/jake/classes/cs703/Project/data/grid/vid/s1/lgaz8p.mpg'
tpath = '/home/jake/classes/cs703/Project/data/grid/align/lgaz8p.align'

video = Video(vpath, tpath)

print(video.transcript.get_sentence())
print(video.transcript.aligns)
face_predictor_path = '/home/jake/classes/cs703/Project/dev/TellGAN/src/assests/predictors/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
feature_detector = dlib.shape_predictor(face_predictor_path)

#height = 128
#width = 128

def pilToMat(pil_img):
    pil_image = pil_img.convert('RGB')
    open_cv_image = np.array(pil_image)
    # Convert RGB to BGR
    return open_cv_image#[:, :, ::-1].copy()

def getFeaturePoints(img):
    faces = detector(img, 1)
    for k, rect in enumerate(faces):
        # print("k: ", k)
        # print("rect: ", rect)
        # shape = self.predictor(img_gray, rect)
        p0 = feature_detector(old_gray, rect)
        break

    face_points = np.asfarray([])
    for i, part in enumerate(p0.parts()):
        fpoint = np.asfarray([part.x, part.y])
        if i is 0:
            face_points = fpoint
        else:
            face_points = np.vstack((face_points, fpoint))
    face_points = np.expand_dims(face_points, axis=1)
    #print("face_points_shape: ", face_points.shape)
    #print("face_points: ", face_points)
    return face_points.astype(np.float32)

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
# Create some random colors
color = np.random.randint(0,255,(100,3))
# Take first frame and find corners in it
old_frame, word = video[0]
#old_frame.show()
old_frame = pilToMat(old_frame)
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = getFeaturePoints(old_gray)
#refp = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
#print("refp: ", refp)
#print("refp size: ", refp.shape)



vid_path = "opitcalflow.mp4"
writer = skvideo.io.FFmpegWriter(vid_path)

for i in range(1, len(video)):
    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)
    frame, word = video[i]
    frame = pilToMat(frame)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #p1 = getFeaturePoints(frame_gray)
    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]
    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 1)
        #frame = cv2.circle(frame,(a,b),1,color[i].tolist(),-1)
    img = cv2.add(frame,mask)

    writer.writeFrame(img)
    #plt.imshow(img)
    #plt.show()
    #k = cv2.waitKey(30) & 0xff
    #if k == 27:
    #    break
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)

writer.close()
#cv2.destroyAllWindows()
