# Import all the modules/packages needed first
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk

from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
set(stopwords.words('english'))
from nltk.tokenize.treebank import TreebankWordDetokenizer

from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import pymongo
from pymongo import MongoClient

# A MongoDB instance must be running before, activate with: mongod
# Connects to the default host and port
client = MongoClient()

# Uses the database name instead of test, which in this particular case is the db
db = client.test 
collection = db.articles                        

# Using Pandas DataFrame object to read the file              
dataFrame = pd.DataFrame(list(collection.find()))



# Look at the first 5 entries of the dataset with Pandas .head() method
dataFrame.head()

# Some spaces for clarity in the output
print(f'\n')
print(f'\n')

# See how many entries its got. .shape[0] gives rows, .shape[1] gives columns 
print(f'There are {dataFrame.shape[0]} entries in this dataset')

# How many unique authors, using the Pandas unique function
print(f'There are {len(dataFrame.author.unique())} unique authors in this dataset')

# Some spaces for clarity in the output
print(f'\n')
print(f'\n')

# Make another dataset called author, where all entries are grouped by author name and with only certain columns
author = dataFrame[["author", "claps", "reading_time", "title"]].groupby("author")
author.describe().head()

# Sort the dataframe showing only the highest 5 reading times. Notice that an article was posted 2 times.
dataFrame.sort_values(by = "reading_time", ascending = False).head()



# Using MathPlotLib we set the dimensions of the graph
plt.figure(figsize = (20, 10))

# With Pandas plot the top 20 reading times sorted by highest ones
dataFrame.sort_values(by = "reading_time", ascending = False)["reading_time"].head(20).plot.bar()

# Set label for the axes
plt.xlabel("Article #")
plt.ylabel("Max Reading Time Needed (minutes)")

plt.show(block = True)



# Let's make a WordCloud with the most common words from the first article. We can also use the set of stopwords provided by
# NLTK, or if undeclared, the default ones
firstArticle = dataFrame.text[0]
wordcloud1 = WordCloud(max_words = 50, background_color = "white").generate(firstArticle)

# Now show it with MathPlotLib
plt.figure(figsize = (15, 5))
plt.imshow(wordcloud1)
plt.show(block = True)



# Now let's use all the articles and make a very big cloud
# First we join all the articles
allArticles = " ".join(article for article in dataFrame.text)

# Let's check how many words are there
print(f'Between all articles, there are {len(allArticles)} words in total.')



# Let's set the NLTK default stopwords
stop_words_base = set(stopwords.words('english'))

# Now we add more, because the default ones were not enough
numbers_stop_words = set(('one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'first', 'second',
                         'third', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'))

more_stop_words = set(('like', 'would', 'use', 'also', 'much', 'us', 'get', 'way', 'new', 'could', 'much', 'many', 'could',
                      'make', 'even', 'want', 'see', 'still', 'need', 'used', 'learn', 'time', 'using'))

# Now we add them all together
stop_words = stop_words_base | numbers_stop_words | more_stop_words

# Revised stopwords without punctuation
no_punct = RegexpTokenizer(r'\w+')

# Tokenize all the articles, now without any punctuation
allArticles_withoutPunct = no_punct.tokenize(allArticles)

# Now let's create a new and clean list, without stopwords
articles_tokens_clean = []

# Loop to check for stopwords
for word in allArticles_withoutPunct:
    if word not in stop_words:
        articles_tokens_clean.append(word)

# Let's check how many USEFUL words are there after the cleaning
print(f'Between all articles, there are {len(articles_tokens_clean)} useful words after cleaning up.')

# Some spaces for clarity in the output
print(f'\n')
print(f'\n')

# Now let's detokenize the list to create a wordcloud later
clean_text = TreebankWordDetokenizer().detokenize(articles_tokens_clean)

# Now like before, we make a wordcloud
total_wordcloud = WordCloud(background_color = "white", max_words = 100).generate(clean_text)

plt.figure(figsize = (15, 5))
plt.imshow(total_wordcloud)
plt.show(block = False)



# Now lets use the tokenized list and make into a dictionary (clear of stopwords) to find the count of the 10 most used words
most_common_ten = dict(Counter(articles_tokens_clean).most_common(10))

# Now lets print the most common ten words first
print(f'The 10 most common words across all articles are:\n')



# Print frequencies and data from the dictionary
for item, count in most_common_ten.items(): 
    print(f'{item}: These skills appeared {count} times, or {((count*100)/len(articles_tokens_clean)):.4f}% of all the words')

# Some spaces for clarity in the output
print(f'\n')

# We make 2 lists for the 10 most common items and key of the dictionary and populate them
ten_most_common_words = []
ten_most_common_freqs = []

for item, count in most_common_ten.items(): 
    ten_most_common_words.append(item) 
    ten_most_common_freqs.append(count)
    
print(f'{ten_most_common_words}\n')
print(f'{ten_most_common_freqs}')

# Some spaces for clarity in the output
print(f'\n')
print(f'\n')

# Make lists for items and frequencies combinations
skill_combinations = []
skill_combination_freqs = []

# Now we create the combinations and append them to the lists
for i in range(5):
    skill_comb_i1 = ten_most_common_words[i] + '/' + ten_most_common_words[i+1]
    skill_comb_i2 = ten_most_common_words[i] + '/' + ten_most_common_words[i+2]
    skill_comb_i3 = ten_most_common_words[i] + '/' + ten_most_common_words[i+3]
    
    skill_comb_freqs_i1 = ten_most_common_freqs[i] + ten_most_common_freqs[i+1]
    skill_comb_freqs_i2 = ten_most_common_freqs[i] + ten_most_common_freqs[i+2]
    skill_comb_freqs_i3 = ten_most_common_freqs[i] + ten_most_common_freqs[i+3]
    
    skill_combinations.extend((skill_comb_i1, skill_comb_i2, skill_comb_i3))
    skill_combination_freqs.extend((skill_comb_freqs_i1, skill_comb_freqs_i2, skill_comb_freqs_i3))
    
print(f'{skill_combinations}\n')
print(f'{skill_combination_freqs}')

# Some spaces for clarity in the output
print(f'\n')
print(f'\n')

# Now we make them into a new dictionary and add items and frequencies as key and value
skills_freqs_combinations = {}

for i in range(10):
    skills_freqs_combinations.update({skill_combinations[i] : skill_combination_freqs[i]})

skills_freqs_combinations



# Now print frequencies and data from the dictionary like before
print(f'The 10 most common pairings of skills across all articles are:\n')

for combination, count in skills_freqs_combinations.items(): 
    print(f'{combination}: These skills appeared {count} times, or {((count*100)/len(articles_tokens_clean)):.4f}% of all the words')

# Some spaces for clarity in the output
print(f'\n')
print(f'\n')    