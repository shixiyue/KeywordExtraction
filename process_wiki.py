# Based on http://textminingonline.com/training-word2vec-model-on-english-wikipedia-by-gensim
# Original Author: Pan Yang (panyangnlp@gmail.com)

import logging
import os.path
import sys

from custom_wikicorpus import CustomWikiCorpus

# Usage: python process_wiki.py enwiki.xxx.xml.bz2 wiki.en.text
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

    i = 0

    output = open(outp, 'w', encoding='utf-8')
    wiki = CustomWikiCorpus(inp, lemmatize=False, dictionary={})
    for text in wiki.get_texts():
        output.write(' '.join(text) + '\n')
        i += 1
        if (i % 1000 == 0):
            logger.info("Saved " + str(i) + " articles")

    output.close()
    logger.info("Finished Saved " + str(i) + " articles")