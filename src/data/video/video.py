import numpy as np
import skvideo.io
import cv2
from PIL import Image, ImageDraw

from .transcript import Transcript


class Video(object):
    def __init__(self, vid_path=None, trans_path=None):

        if vid_path is None:
            self.frames = []
            self.num_frames = 0
            self.frame_rate = 0
        else:
            self.from_video(vid_path)

        self.transcript = Transcript(trans_path)

    def from_video(self, path):
        #(self.frames, self.num_frames, self.frame_rate) = self.get_video_frames(path)
        self.frames = self.get_video_frames(path)
        return self

    def from_array(self, frames):
        self.frames = frames
        return self

    def get_video_frames(self, path):
        #videogen = skvideo.io.vread(path)
        videogen = skvideo.io.vreader(path)
        videometadata = skvideo.io.ffprobe(path)
        #num_frames = np.int(videometadata['video']['@nb_frames'])
        #frame_rate = videometadata['video']['@avg_frame_rate']

        frames = np.array([frame for frame in videogen])

        return frames #(frames, num_frames, frame_rate)

    def __getitem__(self, index):
        # Frame is loaded as RGB already, can just use as is
        frame = self.frames[index]

        # Get Word for next Frame
        next_frame_word = self.transcript.get_word_from_frame(index+1)

        pil_im = Image.fromarray(frame)

        return pil_im, next_frame_word


    def __len__(self):
        return len(self.frames)



def main():

    vpath = '/home/jake/classes/cs703/Project/data/grid/vid/s1/lgaz8p.mpg'
    tpath = '/home/jake/classes/cs703/Project/data/grid/anno/align/lgaz8p.align'

    video = Video(vpath, tpath)

    print(video.transcript.get_sentence())
    print(video.transcript.aligns)

    #print(video.num_frames)
    #print(video.frame_rate)

    for i in range(0, len(video.frames)):
        frame = cv2.cvtColor(video.frames[i],cv2.COLOR_BGR2RGB) #this code do color conversion
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cv2.destroyAllWindows()


if __name__ == '__main__':

    main()