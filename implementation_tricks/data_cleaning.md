```python
# Data Cleaning
import emoji
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = stopwords.words('english')

def only_alnum(s): 
    s1 = s.replace(u'’', u" ").replace("'"," ").replace("“"," ").replace("”"," ")
    s1 = re.sub(r'[^a-z0-9 ]+', ' ', s1.lower())
    return " ".join(s1.split())

def remove_emoji(text):
    allchars = [str for str in text]
    emoji_list = [c for c in allchars if c in emoji.UNICODE_EMOJI]
    clean_text = ' '.join([str for str in text.split() if not any(i in str for i in emoji_list)])

    return clean_text

def remove_punc(text):
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    no_punct = ""
    for char in text:
        if char not in punctuations: 
            no_punct = no_punct + char
        else: no_punct = no_punct + ' '
    return no_punct

def remove_stopwords(text):

    text_tokens = text.split()
    tokens_without_sw = [word for word in text_tokens if not word in stop_words]
    filtered_sentence = (" ").join(tokens_without_sw)

    return filtered_sentence

def normalize_whitespace(text, no_line_breaks=True):
    text = text.strip()
    text = " ".join(text.split())
    return text

def clean_data(data,stopwords=True):
    '''
    if you want stopwords to be removed put False
    '''
    data = [remove_emoji(sent) for sent in data]
    data = [re.sub('\s+', ' ', sent) for sent in data]

    data = [re.sub('http\S+', " ", sent) for sent in data]
    data = [re.sub('#\S+', " ", sent) for sent in data]
    data = [re.sub('@\S+', " ", sent) for sent in data]

    data = [remove_punc(sent) for sent in data]
    data = [only_alnum(sent) for sent in data]
    data = [normalize_whitespace(d) for d in data]

    data = [x.lower() for x in data]
    
    if not stopwords: data = [remove_stopwords(sent) for sent in data]

    data = [sent for sent in data if len(sent.split())!=0]

    return data
```

