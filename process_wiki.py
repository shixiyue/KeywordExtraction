#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Pan Yang (panyangnlp@gmail.com)
# Copyrigh 2017

from __future__ import print_function

import logging
import os.path
import six
import sys
import time

import nltk
from nltk.tag.perceptron import PerceptronTagger
import inflection
from gensim.corpora import WikiCorpus

tagger = PerceptronTagger()

def singularize_nouns(text):
    start = time.time()
    text = [word.decode('utf-8') for word in text]
    text = [inflection.singularize(word) if tag.startswith('NN') else word for word, tag in tagger.tag(text)]
    return text

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    # check and process input arguments
    if len(sys.argv) != 3:
        print("Using: python process_wiki.py enwiki.xxx.xml.bz2 wiki.en.text")
        sys.exit(1)
    inp, outp = sys.argv[1:3]
    space = " "
    i = 0

    output = open(outp, 'w', encoding='utf-8')
    wiki = WikiCorpus(inp, lemmatize=False, dictionary={})
    for text in wiki.get_texts():
        if six.PY3:
            text = singularize_nouns(text)
            output.write(' '.join(text) + '\n')
        #   ###another method###
        #    output.write(
        #            space.join(map(lambda x:x.decode("utf-8"), text)) + '\n')
        else:
            output.write(space.join(text) + "\n")
        i = i + 1
        if (i % 1000 == 0):
            logger.info("Saved " + str(i) + " articles")

    output.close()
    logger.info("Finished Saved " + str(i) + " articles")