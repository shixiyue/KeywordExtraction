== Requirement ==
- The project is written in Python 3.
- Install numpy, scipy, gensim, networkx, spacy, inflection. To run the evaluation script, nltk is required as well.

(spacy is a natural language processing tool that is much faster and more accurate than nltk.
It provides word tokenization, sentence tokenization, part-of-speech tagging, lemmatization etc.
However, there is no build-in stemmer in spacy, while stemming is in standard keyword extraction
evaluation processure.)

== Files included ==

extract_keyword.py: 
- It is the main part.
- To run the scipt, call the function extract_keywords(text, num_of_keywords, threshold=0.6, word_similarity_weight=0.4)
- The method will return an array with num_of_keywords keywords
- The algorithm is based on what is described in the paper QALink

utility.py:
- It provides some utility methods such as normalize_token and is_meaningful_token

evaluate.py:
- It calculates the precision, recall and f1 score of keyword extraction based on 2 datasets:
  * Inspect (paper abstracts, using the test set which consists of 500 documents with less than 200 tokens/doc)
  * SemEval-2010 (scientific papers, using the test set which consists of 100 documents with more than 5000 tokens/doc)
- To change num_of_keywords, word_similarity_weights, thresholds or evaluation documents, please change those corresponding variables
- Result: 
  * Inspect (num_of_keywords=15, word_similarity_weight=0.6, threshold=0.4): precision=0.490, recall=0.427, f1 score=0.431
  * Inspect (num_of_keywords=10, word_similarity_weight=0.6, threshold=0.4): precision=0.553, recall=0.328, f1 score=0.389
  * SemEval-2010 (num_of_keywords=15, word_similarity_weight=0.4, threshold=0.6)
  * SemEval-2010 (num_of_keywords=10, word_similarity_weight=0.4, threshold=0.6)
- document-long: It contains the documents of SemEval-2010
- keywords-long: It contains those annotator-assigned keywords of SemEval-2010
- document-short: It contains the documents of Inspect
- keywords-short: It contains those annotator-assigned keywords of Inspect
- output/ : the output folder for keyword

word2vec/process_wiki.py: process the wikipedia document, to prepare for training for word2vec
word2vec/custom_wiki_corpus: Overrides original tokenization method in wikicorpus.py. It will singularize plural nouns.
word2vec/train_word2vec_model: It trains word2vec model using the processed wikipedia corpus.
word2vec/wiki.em.word2vec.model: Please download it from here: or use the above script to train a new model.

