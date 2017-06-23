import os
from parameters import *

all_documents = os.listdir(keywords_directory)
for document_name in all_documents:
    with open(keywords_directory + document_name) as document:
    	keywords = []
    	for line in document:
    		keywords.extend([word.lower() for word in line.split()])
    with open(keywords_directory + document_name, mode='w') as document:
        document.write(' '.join(keywords))