import io
import itertools
import networkx as nx
import spacy
import os
import inflection
import string
import math
import heapq

from gensim.models.word2vec import Word2Vec
from parameters import *
from utility import *

model = Word2Vec.load("word2vec/model.w2c")

def extract_key_words(text, pagerank, threshold, word_similarity_weight):
    """
    Returns a set of key phrases.
    """
    word_occurrence_dict = dict()
    num_of_sentence = 0
    sentences = nlp(text).sents
    
    for sentence_index, sentence in enumerate(sentences):
        num_of_sentence += 1
        meaningful_words_set = filter_words(sentence)
        update_word_occurrence_dict(word_occurrence_dict, meaningful_words_set, sentence_index)

    graph = build_graph(word_occurrence_dict, num_of_sentence, pagerank, threshold, word_similarity_weight)

    if pagerank:
        return get_pagerank_keywords(graph)
    else:
        try:
            keywords = get_degree_centrality_keywords(graph)
        except:
            # In case that keywords are not enough, we try to use another method.
            keywords = extract_key_words(text, pagerank=True, threshold=threshold, word_similarity_weight=word_similarity_weight)
        return keywords

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

def build_graph(word_occurrence_dict, num_of_sentence, pagerank, threshold, word_similarity_weight):
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
        if (pagerank and tie_strength > 0) or tie_strength >= threshold:
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
    return math.log(p_first_second / (p_first * p_second), 2)

def get_pagerank_keywords(graph):
    """
    Returns those most important words ranked by pagerank.
    """
    calculated_page_rank = nx.pagerank(graph, weight='weight')
    return get_tops(calculated_page_rank)

def get_tops(score_dictionary):
    score_heap = []
    # score_heap is a min-heap. So we need to negate score to simulate a 'max-heap'.
    for (node, score) in score_dictionary.items():
        heapq.heappush(score_heap, (-score, node))
    top_nodes = heapq.nsmallest(num_of_keywords, score_heap)
    return [x[1] for x in top_nodes]

def get_degree_centrality_keywords(graph):
    """
    Returns those most important words ranked by degree centrality.
    """
    keywords = []
    for _ in range(num_of_keywords):
        if graph.size() == 0:
            raise ValueError('Keywords are not enough')
        degree_dict = nx.degree_centrality(graph)
        keyword = max(degree_dict.keys(), key=(lambda k: degree_dict[k]))
        keywords.append(keyword)
        graph.remove_node(keyword)

    return keywords

def main(pagerank=False, threshold=0.6, word_similarity_weight=0.4, window=False):
    all_documents = os.listdir(document_directory)
    for document_name in all_documents:
        with open(document_directory + document_name, encoding='utf-8') as document:
            text = ''
            for line in document:
                text += line.strip() + ' '
        keywords = extract_key_words(text, pagerank, threshold, word_similarity_weight, window)
        with open(output_directory + document_name.split('.')[0] + '.out', mode='w') as document:
            document.write(' '.join(keywords))

if __name__ == "__main__":
    main()