# 486NersKnowledgeGraph

### Dataset used to train NERS Model:
- Corpus features a tagged portiong from the Groningen Meaning Bank tagged specifically for training a model for named entity extraction.
- [Entity Annotated Corpus](https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus)
- [Groningen Meaning Bank](http://gmb.let.rug.nl/data.php)
- Directions:
	- Download data from kaggle, unzip and store both csvs in a data folder.

### Types of Named Entities:
1) geo = Geographical Entity
2) org = Organization
3) per = Person
4) gpe = Geopolitical Entity
5) tim = Time indicator
6) art = Artifact
7) eve = Event
8) nat = Natural Phenomenon
9) o = Other (Considered to be NOT a named entity

### Dependencies
- Numpy
- Pandas
- Sklearn
- Sklearn-CRF
- eli5
- nltk
- json
- urllib

### Running Files
1) ners.py generates two text files of named entities
	- File must be run using Python3
	- Currently uses text_files/raw_text.txt as source for raw text
	- Outputs to text_files/named_entities.txt and text_files/tagged_named_entities.txt 

2) kg.py queries the knowledge graph to find relationally relevant topics based on a seed list of missed topics
	- File must be run using Python2
	- Currently uses text_files/missed.txt as seed list
	- Outputs to text_files/suggested_topics.txt 