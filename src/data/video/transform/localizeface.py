from __future__ import print_function

import dlib
import cv2
from PIL import Image
from scipy.misc import imresize
import numpy as np
import os.path

class LocalizeFace(object):
    """
    Crops the input PIL Image to the detected face.
    """


    def __init__(self, height=None, width=None, predictor_path=None, mouthonly=False, padding=0.19):
        """
        For GRID each frame is (288, 360, 3) by default, the height/width options force the bounding boxes
        to be a specific size.
        """
        predictor_path = 'shape_predictor_68_face_landmarks.dat'
        self.detector = FeaturePredictor(predictor_path)
        self.predictor_path = predictor_path
        self.height = height
        self.width = width
        self.isMouthOnly = mouthonly

        self.HORIZONTAL_PAD = padding
        x1 = self.width if self.width is not None else 0
        y1 = self.height if self.height is not None else 0

        self.prev_bb = (0,0,x1,y1)




        # Predictor only used to predict facial features, used for localizing mouth??
        #self.predictor = dlib.shape_predictor(self.predictor_path)


    def __call__(self, img):

        """
        Args:
            img (PIL Image): Image to be scaled.

        Returns:
            PIL Image with face extracted, returns image if no face detected.
        """

        if self.predictor_path is None:
            return self.naive_crop(img)

        localized, fpoints = self.localize(img, self.isMouthOnly)
        return self.detector.matToPil(localized)


    def localize(self, frame, mouthonly=False):
        normalize_ratio = None



        fpoints_of = self.detector.getFeaturePoints(frame, mouthonly)


        if fpoints_of is None:
            return frame, None, None

        fpoints = fpoints_of.squeeze().astype(np.int)

        #frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #mask = self.detector.create_mask(frame_gray, np.expand_dims(fpoints, axis=1))

        # Reverse coords, as np coords are in [y,x]
        centroid = np.mean(fpoints[:, -2:], axis=0)

        if normalize_ratio is None:
            if self.width > self.height:
                rightpt = np.max(fpoints[:, :-1]) * (1.0 + self.HORIZONTAL_PAD)
                leftpt = np.min(fpoints[:, :-1]) * (1.0 - self.HORIZONTAL_PAD)

                normalize_ratio = self.width / float(rightpt - leftpt)
            else:
                toppt = np.max(fpoints[:, :1]) * (1.0 + self.HORIZONTAL_PAD)
                bottompt = np.min(fpoints[:, :1]) * (1.0 - self.HORIZONTAL_PAD)

                normalize_ratio = self.height / float(toppt - bottompt)

        new_img_shape = (int(frame.shape[0] * normalize_ratio), int(frame.shape[1] * normalize_ratio))


        resized_img = imresize(frame, new_img_shape)
        #resized_mask = imresize(mask, new_img_shape)

        centroid_norm = centroid * normalize_ratio

        object_l = int(centroid_norm[0] - self.width / 2)
        object_r = int(object_l + self.width)
        object_t = int(centroid_norm[1] - self.height / 2)
        object_b = int(object_t + self.height)

        localized = resized_img[object_t:object_b, object_l:object_r]

        if localized.shape <= (self.height,self.width):
            # Pad
            if centroid_norm[0] >= self.width / 2:
                padx = (0, int(self.width - localized.shape[1]))  #(left,right)
            else:
                padx = (int(self.width - localized.shape[1]), 0)  #(left,right)

            if centroid_norm[1] >= self.height / 2:
                pady = (0, int(self.height - localized.shape[0]))  # (top,bottom)
            else:
                pady = (int(self.height - localized.shape[0]), 0)  # (top,bottom)

            localized = np.pad(localized, pad_width=(pady,padx,(0,0)), mode='mean')


        fpoints_norm = (fpoints_of * normalize_ratio).astype(np.int)
        fpoints_norm[:,:,1] -= object_t
        fpoints_norm[:,:,0] -= object_l

        return localized, fpoints_norm.astype(np.float32)


    def naive_crop(self, img):
        """
                Args:
                    img (PIL Image): Image to be scaled.

                Returns:
                    PIL Image with face extracted, returns image if no face detected.
                """
        # need grayscale for dlib face detection
        img_gray = np.array(img.convert('L'))
        faces = self.detector(img_gray, 1)

        # print("Faces: ", len(faces))
        bb = None
        for k, rect in enumerate(faces):
            # print("k: ", k)
            # print("rect: ", rect)
            # shape = self.predictor(img_gray, rect)
            bb = self.rect_to_bb(rect)
            break
            # i = -1
        # if shape is None: # Detector doesn't detect face, just return as is
        #    return img
        # shape = self.shape_to_np(shape)

        # face_crop = img[rect.left():rect.right(), rect.top():rect.bottom()]

        if bb is None:
            bb = self.prev_bb
            return img
        else:
            self.prev_bb = bb

        if self.width is not None:
            (x0, y0, x1, y1) = bb
            face_w = x1 - x0

            cx = x0 + face_w // 2

            new_x0 = cx - self.width // 2
            new_x1 = new_x0 + self.width

            bb = (new_x0, y0, new_x1, y1)

        if self.height is not None:
            (x0, y0, x1, y1) = bb

            face_h = y1 - y0

            cy = y0 + face_h // 2

            new_y0 = cy - self.height // 2
            new_y1 = new_y0 + self.height

            bb = (x0, new_y0, x1, new_y1)

        face_crop = img.crop(bb)

        # print("shape: ", shape.shape)

        return face_crop

    def shape_to_np(self, shape, dtype="int"):
        # initialize the list of (x, y)-coordinates
        coords = np.zeros((68, 2), dtype=dtype)

        # loop over the 68 facial landmarks and convert them
        # to a 2-tuple of (x, y)-coordinates
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)

        # return the list of (x, y)-coordinates
        return coords

    def rect_to_bb(self, rect):
        # take a bounding predicted by dlib and convert it
        # to the format (x, y, w, h) as we would normally do
        # with OpenCV
        x = rect.left()
        y = rect.top()
        w = rect.right()# - x
        h = rect.bottom()# - y

        # return a tuple of (x, y, w, h)
        return (x, y, w, h)


class FeaturePredictor(object):
    def __init__(self, feature_model=None):
        self.feature_model = feature_model
        self.face_detector = dlib.get_frontal_face_detector()

        self.feature_detector = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

        # Parameters for lucas kanade optical flow
        self.lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    def getFlowMask(self, old, new, features0=None):
        #mat0 = self.pilToMat(pil0)
        gray0 = cv2.cvtColor(old, cv2.COLOR_BGR2GRAY)
        #mat1 = self.pilToMat(pil1)
        gray1 = cv2.cvtColor(new, cv2.COLOR_BGR2GRAY)
        features1 = None

        if features0 is None:
            features0 = self.getFeaturePoints(gray0)

        # May fail to get features, return None and blank mask
        if features0 is not None:
            features1, features0 = self.getFlow(gray0, gray1, features0)

        features1 = features1.reshape(-1,1,2) if features1 is not None else None

        mask = self.create_mask(size=(gray1.shape[:2]), features=features1)

        return features1, self.matToPil(mask)

    def getFeatureMask(self, pil0, mouthonly=False):
        #gray0 = cv2.cvtColor(self.pilToMat(pil0), cv2.COLOR_BGR2GRAY)
        gray0 = cv2.cvtColor(pil0, cv2.COLOR_BGR2GRAY)
        features0 = self.getFeaturePoints(gray0, mouthonly)

        mask = self.create_mask(size=(gray0.shape[:2]), features=features0)
        return features0, self.matToPil(mask)


    def getFlow(self, old_gray, frame_gray, features0):
        features1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, features0,
                                                      None, ** self.lk_params)
        # Select good points
        good_new = features1[st == 1]
        good_old = features0[st == 1]

        return good_new, good_old

    def create_mask(self, size, features):

        mask = np.zeros(size, dtype=np.uint8)
        if features is None:
            return mask

        features_ = np.copy(features).astype(np.int)

        #features_ = features_[features_[:,0,0] < img.shape[0]]
        #features_ = features_[features_[:,0,1] < img.shape[1]]

        mask[features_[:,:,1], features_[:,:,0]] = 255

        return mask

    def locate(self, img_grey):
        faces = self.face_detector(img_grey, 1)
        return faces

    def extractMouthFeatures(self, features):
        feat_points = features[48:68]
        return feat_points

    def getFeaturePoints(self, img_grey, mouth_only=False):
        faces = self.locate(img_grey)
        features = None

        for k, rect in enumerate(faces):
            # print("k: ", k)
            # print("rect: ", rect)
            # shape = self.predictor(img_gray, rect)
            features = self.feature_detector(img_grey, rect)
            break

        feat_points = np.asfarray([])
        if features is None:
            return None

        for i, part in enumerate(features.parts()):
            fpoint = np.asfarray([part.x, part.y])
            # filter if index values larger than image
            if (fpoint < 0).any() or fpoint[0] >= img_grey.shape[1] or fpoint[1] >= img_grey.shape[0]:
                print("ignoring point: {} | imgsize: {}".format(fpoint,img_grey.shape))
                continue
            if i is 0:
                feat_points = fpoint
            else:
                feat_points = np.vstack((feat_points, fpoint))

        if mouth_only is True:
            # starting at 1, mouth indices are 49-68
            feat_points = self.extractMouthFeatures(feat_points)

        #if for_optical_flow:
        feat_points = np.expand_dims(feat_points, axis=1).astype(np.float32)


        # print("face_points_shape: ", feat_points.shape)
        # print("feat_points: ", feat_points)
        return feat_points

    def matToPil(self, mat_img):
        return Image.fromarray(mat_img)


    def pilToMatGrey(self, pil_img):
        pil_image = pil_img.convert('LA')
        open_cv_image = np.array(pil_image)
        # Convert RGB to BGR
        return open_cv_image  # [:, :, ::-1].copy()

    def pilToMat(self, pil_img):
        pil_image = pil_img.convert('RGB')
        open_cv_image = np.array(pil_image)
        # Convert RGB to BGR
        return open_cv_image  # [:, :, ::-1].copy()

def main():

    ppath = '/home/jake/classes/cs703/Project/dev/TellGAN/src/assests/predictors/shape_predictor_68_face_landmarks.dat'
    fpath = '/home/jake/classes/cs703/Project/me.jpg'
    transform = LocalizeFace()#ppath)
    img = Image.open(fpath)

    face = transform(img)
    face.show()


if __name__ == '__main__':

    main()