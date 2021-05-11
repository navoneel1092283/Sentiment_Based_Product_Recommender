import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def text_process(msg):
    # 1. all lower case
    # 2. remove punctuation and stopwords
    # 3. lemmatization
    msg = msg.lower()
    nopunct = [char for char in msg if char not in string.punctuation]
    nopunct = ''.join(nopunct)
    a = ''
    list_of_words = nopunct.split()
    for i in range(len(list_of_words)):
        if list_of_words[i] not in stopwords.words('english'):
            b = WordNetLemmatizer().lemmatize(list_of_words[i], pos="v")
            a = a + b + ' '
    a = a.rstrip()
    return a