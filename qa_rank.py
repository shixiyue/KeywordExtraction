"""Python implementation of the TextRank algoritm.
From this paper:
    https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf
Based on:
    https://gist.github.com/voidfiles/1646117
    https://github.com/davidadamojr/TextRank
"""
import io
import itertools
import networkx as nx
import nltk
import os
import inflection
import string
import math

from nltk.tag.perceptron import PerceptronTagger
from gensim.models.word2vec import Word2Vec
from parameters import *

meaningless_tokens = ['‘', '’', '”', '“']
meaningless_tokens.extend(string.punctuation)

tagger = PerceptronTagger()
model = Word2Vec.load("model.w2c")

def filter_for_tags(tagged, tags=['NN', 'JJ']):
    """Apply syntactic filters based on POS tags, 
    remove plurals for each tagged noun and apply case folding."""
    words_set = set()
    for word, tag in tagged:
        if word in meaningless_tokens:
            continue
        if tag.startswith('NN'):
             words_set.add(inflection.singularize(word).lower())
        elif tag.startswith('JJ'):
            words_set.add(word.lower())
    return words_set

def calculate_tie_strength(first_word, second_word, word_occurrence_dict, num_of_sentence):
    """Return the tie strength between two strings.
    """
    return word_similarity_weight * word_similarity(first_word, second_word) + \
           pmi(first_word, second_word, word_occurrence_dict, num_of_sentence) 

def word_similarity(first_word, second_word):
    try:
        return model.similarity(first_word, second_word)
    except:
        return 0

def pmi(first_word, second_word, word_occurrence_dict, num_of_sentence):
    first_word_count = len(word_occurrence_dict[first_word])
    second_word_count = len(word_occurrence_dict[second_word])
    cooccurrence_count = len(word_occurrence_dict[first_word].intersection(word_occurrence_dict[second_word]))
    if cooccurrence_count == 0:
        return 0

    p_first_second = cooccurrence_count / num_of_sentence
    p_first = first_word_count / num_of_sentence
    p_second = second_word_count / num_of_sentence
    return math.log(p_first_second / (p_first * p_second), 2)

def build_graph(word_occurrence_dict, num_of_sentence, pagerank):
    """Return a networkx graph instance.
    :param nodes: List of hashables that represent the nodes of a graph.
    """
    nodes = word_occurrence_dict.keys()
    print(len(nodes))
    gr = nx.Graph()  # initialize an undirected graph
    gr.add_nodes_from(nodes)
    nodePairs = list(itertools.combinations(nodes, 2))

    # add edges to the graph (weighted by Levenshtein distance)
    for pair in nodePairs:
        first_word = pair[0]
        second_word = pair[1]
        tie_strength = calculate_tie_strength(first_word, second_word, word_occurrence_dict, num_of_sentence)
        if pagerank or tie_strength >= threshold:
            gr.add_edge(first_word, second_word, weight=tie_strength)
    return gr

def update_word_occurrence_dict(word_occurrence_dict, words_set, index):
    for word in words_set:
        if word not in word_occurrence_dict:
            word_occurrence_dict[word] = set()
        word_occurrence_dict[word].add(index)

def extract_key_phrases(text, pagerank):
    """Return a set of key phrases.
    :param text: A string.
    """
    word_occurrence_dict = dict()

    # tokenize the text using nltk
    sentences = nltk.sent_tokenize(text)
    for index, sentence in enumerate(sentences):
        word_tokens = nltk.word_tokenize(sentence)
        # assign POS tags to the words in the text
        tagged = tagger.tag(word_tokens)
        meaningful_words_set = filter_for_tags(tagged)
        update_word_occurrence_dict(word_occurrence_dict, meaningful_words_set, index)

    graph = build_graph(word_occurrence_dict, len(sentences), pagerank)

    if pagerank:
        get_pagerank_keywords(graph)
    else:
        get_degree__centrality_keywords(graph)

def get_pagerank_keywords(graph):
    calculated_page_rank = nx.pagerank(graph, weight='weight')
    # most important words in ascending order of importance
    keywords = sorted(calculated_page_rank, key=calculated_page_rank.get,
                        reverse=True)
    print(keywords[0: 20 if len(keywords) >= 20 else len(keywords)])

def get_degree__centrality_keywords(graph):
    keywords = []
    for _ in range(20):
        if graph.size() == 0:
            break
        degree_dict = nx.degree_centrality(graph)
        keyword = max(degree_dict.keys(), key=(lambda k: degree_dict[k]))
        keywords.append(keyword)
        graph.remove_node(keyword)

    print(keywords)

with open('6.abstr') as document:
    text = ''
    for line in document:
        text += line.strip() + ' '
    extract_key_phrases(text, pagerank=True)