""" Scan top 10 headlines from various website and aggregate sentiment results """

import numpy as np
import flair
from collections import Counter
from datetime import date
import newspaper
import pickle

websites = [
    'https://www.nytimes.com/section/business',
    'https://www.reuters.com/',
    'https://www.ft.com/',
    'https://www.thestreet.com/',
    'https://www.cnbc.com/stocks/'
    'https://www.marketwatch.com/',
    'https://www.bloomberg.com/',
    'https://www.cbc.ca/news/business',
    'https://www.bbc.com/news/business',
    'https://news.yahoo.com/business/',
    'https://www.washingtonpost.com/business/',
    'https://www.nbcnews.com/business',
    'https://www.cnn.com/BUSINESS',
    
]

def check_if_english(string):
    """ check if string is ascii characters """
    return string.isascii()

# load sentiment model from flair
sentiment_model = flair.models.TextClassifier.load('en-sentiment')

headlines = []

for site in websites:
    try:
        # scan with newspaper
        n = newspaper.build(site, language='en', memoize_articles=False)
        articles = n.articles

        # top 10 articles only
        for article in articles[0:10]:
            article.download()
            article.parse()
            title = str(article.title)
            cleaned_title = title.strip()
            if check_if_english(cleaned_title) and cleaned_title is not None:
                if len(cleaned_title) > 10:
                    headlines.append(cleaned_title)
    except Exception as e:
        print(e)

# save todays headlines
with open(str(date.today()), 'wb') as f:
    pickle.dump(headlines, f)

# load todays headlines
with open(str(date.today()), 'rb') as f:
    headlines = pickle.load(f)

# remove duplicates
headlines = list(set(headlines))

sentiment = []
polarity = []

for title in headlines:
    sentence_rep = flair.data.Sentence(title)
    sentiment_model.predict(sentence_rep)
    prediction = sentence_rep.labels
    [curr_sentiment, curr_polarity] = str(prediction[0]).split()
    curr_polarity = float(curr_polarity[1:-1])
    sentiment.append(curr_sentiment)
    polarity.append(curr_polarity)

counter = Counter(sentiment)
total = sum(counter.values())
class_percent = {key: value/total * 100 for key, value in counter.items()}

print('Aggregated Sentiment Stats: ')
for sentiment, percentage in class_percent.items():
    print(sentiment, str(round(percentage,2)) + '%')
print(f'Avg Polarity: {round(np.mean(polarity),2)}')