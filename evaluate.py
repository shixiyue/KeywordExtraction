import os

from nltk import PorterStemmer
from extract_keyword import *

stemmer = PorterStemmer()
score_array = []

num_of_keywords = 15
document_directory = 'document-short/'
keywords_directory = 'keywords-short/'

output_directory = 'output/'
os.makedirs(os.path.dirname(output_directory), exist_ok=True)

def evaluate_output(document_name):
	with open(keywords_directory + document_name) as document:
		keywords = set()
		for line in document:
			for word in line.strip().split():
				keywords.add(stemmer.stem(word))
	with open(output_directory + document_name.split('.')[0] + '.out') as document:
		generated_keywords = set()
		for line_index, line in enumerate(document):
			if line_index == num_of_keywords:
				break
			for word in line.strip().split():
				generated_keywords.add(stemmer.stem(word))
	
	matched = len(generated_keywords.intersection(keywords))
	precision = matched / num_of_keywords
	if len(keywords) == 0:
		recall = 0
	else:
		recall = matched / len(keywords)
	if precision + recall > 0:
		f1 = 2 * precision * recall / (precision + recall)
	else:
		f1 = 0
	return precision, recall, f1

def evaluate(threshold, word_similarity_weight):
	parameters_settings = str(threshold) + '-' + str(word_similarity_weight)
	total_precision = 0
	total_recall = 0
	total_f1 = 0
	i = 0
	for document_name in os.listdir(keywords_directory):
		precision, recall, f1 = evaluate_output(document_name)
		total_precision += precision
		total_recall += recall
		total_f1 += f1
		i += 1
	print(total_precision / i)
	print(total_recall / i)
	print(total_f1 / i)
	print()

def try_parameters():
	word_similarity_weights = [0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
	thresholds = [0.4, 0.45, 0.5, 0.55, 0.6]
	for word_similarity_weight in word_similarity_weights:
		for threshold in thresholds:
			print(word_similarity_weight)
			print(threshold)
			run(threshold, word_similarity_weight)
			evaluate(threshold, word_similarity_weight)

def run(threshold, word_similarity_weight):
    all_documents = os.listdir(document_directory)
    for document_name in all_documents:
        with open(document_directory + document_name, encoding='utf-8') as document:
            text = ''
            for line in document:
                text += line.strip() + ' '
        keywords = extract_keywords(text, num_of_keywords, threshold, word_similarity_weight)
        with open(output_directory + document_name.split('.')[0] + '.out', mode='w', encoding='utf-8') as document:
            document.write('\n'.join(keywords))

if __name__ == "__main__":
	try_parameters()