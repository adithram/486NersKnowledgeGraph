# kg.py
# Given a list of missed topics, this program will return related topics as indicated by 
#	the google knowledge graph

import json
import urllib

# Read in missed topics from file
queries = []
with open('text_files/missed.txt', 'r') as f:
	queries.append(f.readline())


api_key = open('.apikey').read()
print api_key
service_url = 'https://kgsearch.googleapis.com/v1/entities:search'

related_items = []

for query in queries:
	params = {
	    'query': query,
	    'limit': 3,
	    'indent': True,
	    'key': api_key,
	}
	url = service_url + '?' + urllib.urlencode(params)
	response = json.loads(urllib.urlopen(url).read())
	related_items.append(response)
for response in related_items:
	print response
	for element in response['itemListElement']:
	  print (element['result']['name'] + ' (' + str(element['resultScore']) + ')')
