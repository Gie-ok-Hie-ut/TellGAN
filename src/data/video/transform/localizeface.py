
import dlib
from PIL import Image
import numpy as np

class LocalizeFace(object):
    """
    Crops the input PIL Image to the detected face.
    """


    def __init__(self): #, predictor_path):
        #self.predictor_path = predictor_path

        self.detector = dlib.get_frontal_face_detector()

        # Predictor only used to predict facial features, used for localizing mouth??
        #self.predictor = dlib.shape_predictor(self.predictor_path)


    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be scaled.

        Returns:
            PIL Image with face extracted, returns image if no face detected.
        """
        # need grayscale for dlib face detection
        img_gray = np.array(img.convert('L'))
        faces = self.detector(img_gray, 1)

        #print("Faces: ", len(faces))

        for k, rect in enumerate(faces):
            #print("k: ", k)
            #print("rect: ", rect)
            #shape = self.predictor(img_gray, rect)
            bb = self.rect_to_bb(rect)
            break
            #i = -1
        #if shape is None: # Detector doesn't detect face, just return as is
        #    return img
        #shape = self.shape_to_np(shape)

        #face_crop = img[rect.left():rect.right(), rect.top():rect.bottom()]

        if bb is None:
            return img

        face_crop = img.crop(bb)

        #print("shape: ", shape.shape)

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


def main():

    ppath = '/home/jake/classes/cs703/Project/dev/TellGAN/src/assests/predictors/shape_predictor_68_face_landmarks.dat'
    fpath = '/home/jake/classes/cs703/Project/me.jpg'
    transform = LocalizeFace()#ppath)
    img = Image.open(fpath)

    face = transform(img)
    face.show()


if __name__ == '__main__':

    main()