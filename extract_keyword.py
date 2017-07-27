#!/usr/bin/python3

import os
import sys

'''
To set LD_PRELOAD so that Java program can call this part of code without exception.
Delete/comment out this part of code if you run the Python sript directly.
'''
'''LD_PRELOAD = 'LD_PRELOAD'
SYS_EXECUTABLE = '/usr/bin/python3'
SYS_ARGV = ['/data/word2vec/extract_keyword.py']
PYTHON = "/usr/lib64/libpython3.4m.so"

rerun = True
if not LD_PRELOAD in os.environ:
    os.environ['LD_PRELOAD'] = ":" + PYTHON
elif not PYTHON in os.environ.get(PYTHON):
    os.environ['LD_PRELOAD'] += ":" + PYTHON
else:
    rerun = False

if rerun:
     os.execve(SYS_EXECUTABLE, ['python3'] + SYS_ARGV, os.environ)

sys.path.append(os.path.abspath('/data/word2vec'))'''
# Until here

import io
import itertools
import math
import networkx as nx

from gensim.models.word2vec import Word2Vec

from utility import *

model = Word2Vec.load("word2vec/wiki.en.word2vec.model")

def extract_keywords(text, num_of_keywords, threshold=0.4, word_similarity_weight=0.6):
    """
    Returns a set of keywords. It is the main function of the script.
    """
    word_occurrence_dict = dict()
    num_of_sentence = 0
    sentences = [[token for token in sentence] for sentence in nlp(text).sents]
    
    for sentence_index, sentence in enumerate(sentences):
        num_of_sentence += 1
        meaningful_words_set = filter_words(sentence)
        update_word_occurrence_dict(word_occurrence_dict, meaningful_words_set, sentence_index)

    graph = build_graph(word_occurrence_dict, num_of_sentence, threshold, word_similarity_weight)

    return get_degree_centrality_keywords(graph, num_of_keywords)

def filter_words(sentence):
    """
    Removes words that are not meaningful for keyword extraction.
    """
    words_set = set()
    for token in sentence: 
    	if is_meaningful_token(token):
    		words_set.add(normalize_token(token))
    return words_set

def update_word_occurrence_dict(word_occurrence_dict, words_set, sentence_index):
    """
    Updates word occurrence dict of the sentence.
    """
    for word in words_set:
        if word not in word_occurrence_dict:
            word_occurrence_dict[word] = set()
        word_occurrence_dict[word].add(sentence_index)

def build_graph(word_occurrence_dict, num_of_sentence, threshold, word_similarity_weight):
    """
    Returns a networkx graph instance.
    """
    nodes = word_occurrence_dict.keys()
    gr = nx.Graph()  # initialize an undirected graph
    gr.add_nodes_from(nodes)
    node_pairs = list(itertools.combinations(nodes, 2))

    # add edges to the graph (weighted by tie strength)
    for pair in node_pairs:
        first_word = pair[0]
        second_word = pair[1]
        tie_strength = calculate_tie_strength(first_word, second_word, word_occurrence_dict, num_of_sentence, word_similarity_weight)
        if tie_strength >= threshold:
            gr.add_edge(first_word, second_word, weight=tie_strength)
    return gr

def calculate_tie_strength(first_word, second_word, word_occurrence_dict, num_of_sentence, word_similarity_weight):
    """
    Returns the tie strength between two words.
    """
    return word_similarity_weight * word_similarity(first_word, second_word) + \
           pmi(first_word, second_word, word_occurrence_dict, num_of_sentence) 

def word_similarity(first_word, second_word):
    """
    Returns the similarity between two words based on Word2Vec
    """
    try:
        return model.similarity(first_word, second_word)
    except:
        return 0

def pmi(first_word, second_word, word_occurrence_dict, num_of_sentence):
    """
    Calculates the pointwise mutual information of two words
    """
    first_word_count = len(word_occurrence_dict[first_word])
    second_word_count = len(word_occurrence_dict[second_word])
    cooccurrence_count = len(word_occurrence_dict[first_word].intersection(word_occurrence_dict[second_word]))
    if cooccurrence_count == 0:
        return 0

    p_first_second = cooccurrence_count / num_of_sentence
    p_first = first_word_count / num_of_sentence
    p_second = second_word_count / num_of_sentence
    return math.log2(p_first_second / (p_first * p_second))

def get_degree_centrality_keywords(graph, num_of_keywords):
    """
    Returns those most important words ranked by degree centrality.
    """
    keywords = []
    for i in itertools.repeat(None, num_of_keywords):
        if graph.size() == 0:
            return keywords
        degree_dict = nx.degree_centrality(graph)
        keyword = max(degree_dict.keys(), key=(lambda k: degree_dict[k]))
        keywords.append(keyword)
        graph.remove_node(keyword)
    return keywords
