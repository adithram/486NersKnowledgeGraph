# kg.py
# Given a list of missed topics, this program will return related topics as indicated by 
#	the google knowledge graph

import json
import urllib

# Read in missed topics from file
queries = []
with open('text_files/missed.txt', 'r') as f:
	queries = f.readlines()

# strip new line characters
queries = [x.strip() for x in queries] 


api_key = open('.apikey').read()
service_url = 'https://kgsearch.googleapis.com/v1/entities:search'

related_items = []


for query in queries:
	params = {
	    'query': query,
	    'limit': 10,
	    'indent': True,
	    'key': api_key,
	}
	url = service_url + '?' + urllib.urlencode(params)
	response = json.loads(urllib.urlopen(url).read())
	related_items.append(response)

output = ''
for i, response in enumerate(related_items):
	output += 'More information for ' + queries[i] + ' can be found by researching:\n'
	output += '--------------------\n'
	for element in response['itemListElement']:
	  output +=  (element['result']['name']).encode('utf-8') + ' (' + str(element['resultScore']) + ')\n'

	output += '\n'

with open('text_files/suggested_topics.txt', 'w') as f:
	f.write(output)

