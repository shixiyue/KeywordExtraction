import multiprocessing
import logging
from optparse import OptionParser

import gensim

def read_corpus(path_to_corpus, output_path, min_count=10, size=300, window=5):
    workers = multiprocessing.cpu_count()
    sentences = gensim.models.word2vec.LineSentence(path_to_corpus)
    model = gensim.models.Word2Vec(sentences, min_count=min_count, size=size,
                                   window=window, sg=1, workers=workers)
    model.save(output_path)

def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)
    
    parser = OptionParser(usage="usage: %prog [options] corpus outputModel",
                          version="%prog 1.0")

    parser.add_option("-m", "--min_count",
                      action="store",
                      dest="min_count",
                      default=10,
                      type="int",
                      help="min number of apperances",)

    parser.add_option("-s", "--size",
                      action="store",
                      dest="size",
                      default=300,
                      type="int",
                      help="vectors size",)

    parser.add_option("-w", "--windows",
                      action="store",
                      dest="window",
                      default=10,
                      type="int",
                      help="window size",)

    (options, args) = parser.parse_args()

    if len(args) != 2:
        parser.error("wrong number of arguments")

    option_dict = vars(options)
    option_dict["path_to_corpus"] = args[0]
    option_dict["output_path"] = args[1]

    read_corpus(**option_dict)

if __name__ == "__main__":
    main()