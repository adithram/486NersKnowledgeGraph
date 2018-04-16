from __future__ import print_function

from scoring.application import demo486
from kg import get_kg_topics
import time 


def score(answer, top_words, point_values, named_entities):
	score = 0.0
	missed_topics = []

	for idx, word in enumerate(top_words):
		if word.lower() in answer.lower():
			score += point_values[idx]
		else:
			missed_topics.append(top_words[idx])

	return score, missed_topics


def main():
	question_text = raw_input("Welcome, please type your question\n")
	source_doc = raw_input("Please give the filepath to the source document\n")
	document = None
	with open(source_doc) as f:
		document = f.readlines()

	document = [x.strip() for x in document]
	top_words, point_values, named_entities = demo486(document, 5)

	print('These are the top_words and point_values retrieved from the document {}'.format(
		zip(top_words, point_values)))

	print('These are the Named Entities extracted from your document {}'.format(
		named_entities))

	print('Now, the demo will switch to student mode...')
	# time.sleep(2)

	print(question_text)
	answer = raw_input("Please type in your answer now\n")

	points, missed_topics = score(answer, top_words, point_values, named_entities)
	get_kg_topics(missed_topics)

	print("Thank you for demoing!")

if __name__ == '__main__':
	while True:
		main()


