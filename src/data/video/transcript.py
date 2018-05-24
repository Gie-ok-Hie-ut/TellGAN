from __future__ import print_function

class Transcript(object):
    def __init__(self, path=None):

        if path is None:
            self.aligns = []
            self.sentence = None
        else:
            self.from_file(path)

    def from_file(self, path):
        #reset
        self.sentence = None

        with open(path, 'r') as f:
            lines = f.readlines()
        self.aligns = [Align(int(y[0])/1000, int(y[1])/1000, y[2]) for y in [x.strip().split(" ") for x in lines]]

        # 'sil' means silence
        # 'sp' means there is video problem and should be skipped
        # remove buggy frames, but keep silence?
        self.aligns = self.strip(self.aligns, ['sp'])
        #self.aligns = self.strip(self.aligns, ["sp", "sil"])

        self.sentence = self.get_sentence()

        return self

    def from_align(self, align):
        self.aligns.append(align)
        return self

    def strip(self, aligns, items):
        return [sub for sub in aligns if sub.word not in items]

    def make_sentence(self, aligns=None):
        return " ".join([y.word for y in aligns])

    def get_sentence(self):
        if self.sentence is None:
            self.sentence = self.make_sentence(self.aligns)
        return self.sentence

    def get_word_from_frame(self, frame_num):
        for align in self.aligns:
            if frame_num >= align.start and frame_num < align.end:
                return align.word

        # Transcritps have startFrame and EndFrame Number,
        # We get words from start <= frame < end
        # FIX: last frame needs a transcript, give last word in aligns
        if frame_num == self.last_frame:
            return self.aligns[-1].word

        return None

    '''
    # Returns the length of the longest word 
    '''
    @property
    def word_length(self):
        return len(self.sentence.split(" "))

    '''
    # Number of first frame 
    '''
    @property
    def first_frame(self):
        return self.aligns[0].start if len(self.aligns) > 0 else 0


    '''
    # Number of last frame 
    '''
    @property
    def last_frame(self):
        return self.aligns[-1].end if len(self.aligns) > 0 else 0

    '''
    Returns total chars in sentence
    '''
    @property
    def sentence_length(self):
        return len(self.sentence)

class Align(object):
    def __init__(self, start, end, word):
        self.start = start
        self.end   = end
        self.word  = word

    def __str__(self):
        return str((self.start, self.end ,self.word))

    __repr__ = __str__


def main():
    path = '/home/jake/classes/cs703/Project/data/grid/anno/align/lgaz8p.align'

    transcript = Transcript(path)

    print(transcript.get_sentence())
    print(transcript.aligns)
    print(transcript.sentence_length)
    print(transcript.word_length)
    print(transcript.first_frame)
    print(transcript.last_frame)

if __name__ == '__main__':

    main()

