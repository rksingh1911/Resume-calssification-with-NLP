import re
import string
string.punctuation
exclude = string.punctuation
from nltk.corpus import stopwords
stopwords.words('english')

def remove_html_tags(text):
    pattern=re.compile('<,*?>')
    return pattern.sub(r'',text)

def remove_url(text):
    pattern=re.compile(r'https?://\S+}www\.\S+')
    return pattern.sub(r'',text)

def remove_punc(text):
       return text.translate(str.maketrans('','',exclude))

def remove_stopwords(text):
    new_text = []
    for word in text.split():
        if word in stopwords.words('english'):
            new_text.append('')
        else:
            new_text.append(word)
    x = new_text[:]
    new_text.clear()
    return " ".join(x)
