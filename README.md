# Keyword Extraction

## Algorithm Overview
https://docs.google.com/document/d/1K4neMHBZC0Y9b6x1Da1NVARnLcxl6mXT7du8yWCYd2c/edit?usp=sharing

## Requirement
- The project is written in python 3.
- Install numpy, scipy, gensim, networkx, spacy, inflection. To run the evaluation script, nltk is required as well.

(spacy is a natural language processing tool that is much faster and more accurate than nltk.
It provides word tokenization, sentence tokenization, part-of-speech tagging, lemmatization etc.
However, there is no build-in stemmer in spacy, while stemming is in standard keyword extraction
evaluation processure.)

## Files included

### Main Part (Compulsary!!!)
extract_keyword.py: 
- It is the main part.
- To run the scipt, call the function extract_keywords(text, num_of_keywords, threshold=0.6, word_similarity_weight=0.4)
- The method will return an array with num_of_keywords keywords
- The algorithm is based on what is described in the paper QALink
- The word_similarity_weights and thresholds are changed to optimize the result. Such a large word_similarity_weights (0.9)
is chosen because the value of point mutual information and the value of word similarity are not in the same scale: e.g. pmi = 2.807354922057604
and word similarity = 0.273369846168

utility.py:
- It provides some utility methods such as normalize_token and is_meaningful_token

word2vec/wiki.em.word2vec.model, word2vec/wiki.en.word2vec.model.syn1neg.npy, word2vec/wiki.en.word2vec.model.wv.syn0.npy: Please find them in the readpeer server /data/keyword/word2vec or use the above script to train a new model.

### Evaluation
evaluate.py:
- It calculates the precision, recall and f1 score of keyword extraction based on 2 datasets:
  * Inspect (paper abstracts, using the test set which consists of 500 documents with less than 200 tokens/doc)
  * SemEval-2010 (scientific papers, using the test set which consists of 100 documents with more than 5000 tokens/doc)
- To change num_of_keywords, word_similarity_weights, thresholds or evaluation documents, please change those corresponding variables
- Result: 
  * Inspect (num_of_keywords=15, word_similarity_weight=0.9, threshold=0.4): precision=0.497, recall=0.435, f1 score=0.437
  * Inspect (num_of_keywords=10, word_similarity_weight=0.9, threshold=0.4): precision=0.564, recall=0.340, f1 score=0.401
  * SemEval-2010 (num_of_keywords=15, word_similarity_weight=0.9, threshold=0.8): precision=0.477, recall=0.312, f1 score=0.372
  * SemEval-2010 (num_of_keywords=10, word_similarity_weight=0.9, threshold=0.8): precision=0.568, recall=0.249, f1 score=0.342
- document-long: It contains the documents of SemEval-2010
- keywords-long: It contains those annotator-assigned keywords of SemEval-2010
- document-short: It contains the documents of Inspect
- keywords-short: It contains those annotator-assigned keywords of Inspect

word2vec/process_wiki.py: Processes the wikipedia document, to prepare for training for word2vec
word2vec/custom_wiki_corpus: Overrides original tokenization method in wikicorpus.py. It will singularize plural nouns.
word2vec/train_word2vec_model: It trains word2vec model using the processed wikipedia corpus.