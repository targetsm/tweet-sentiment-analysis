import numpy as np 
import random
import translators as ts

def get_dataset():
    tweets = []
    labels = []
    def load_tweets(filename, label):
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                tweets.append(line.rstrip())
                labels.append(label)
    
    load_tweets('twitter-datasets/train_neg_full.txt', 0)
    load_tweets('twitter-datasets/train_pos_full.txt', 1)

    # Convert to NumPy array to facilitate indexing
    tweets = np.array(tweets)
    labels = np.array(labels)

    def get_test_set(filename):
        twt = []
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                twt.append(line.split(",", 1)[1].rstrip())

        return twt

    X_test = get_test_set('twitter-datasets/test_data.txt')

    return tweets, labels, X_test

# from https://gist.github.com/sebleier/554280

nltk_stopwords = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]
# remove [but, no, not, non, don('t)]
nltk_stopwords_filtered = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "should", "now"]

manually_filtering = ['<user>', '.', ',', '(', '<url>', '"', '-', ')',
       "i'm", ':', 'rt', 'like', '/', '&', 'lol', 'good', "'", "it's", 'x',
       'im', '*', '>', "i'll", "you're", 'us', '<', "i've", ';', '|', '^', '+', '#', '$', '~', '[', '_', ']']


def get_dataset_stopword():
    tweets = []
    labels = []
    def load_tweets(filename, label):
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                tweets.append(" ".join([x for x in line.rstrip().split(" ") if x not in nltk_stopwords]))
                labels.append(label)
    
    load_tweets('twitter-datasets/train_neg_full.txt', 0)
    load_tweets('twitter-datasets/train_pos_full.txt', 1)

    # Convert to NumPy array to facilitate indexing
    tweets = np.array(tweets)
    labels = np.array(labels)

    def get_test_set(filename):
        twt = []
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                twt.append(" ".join([x for x in line.split(",", 1)[1].rstrip().split(" ") if x not in nltk_stopwords]))

        return twt

    X_test = get_test_set('twitter-datasets/test_data.txt')

    return tweets, labels, X_test

def get_dataset_stopword(is_lower=False):
    tweets = []
    labels = []
    def load_tweets(filename, label):
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.lower()
                tweets.append(" ".join([x for x in line.rstrip().split(" ") if x not in nltk_stopwords]))
                labels.append(label)
    
    load_tweets('twitter-datasets/train_neg_full.txt', 0)
    load_tweets('twitter-datasets/train_pos_full.txt', 1)

    # Convert to NumPy array to facilitate indexing
    tweets = np.array(tweets)
    labels = np.array(labels)

    def get_test_set(filename):
        twt = []
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                twt.append(" ".join([x for x in line.split(",", 1)[1].rstrip().split(" ") if x not in nltk_stopwords]))

        return twt

    X_test = get_test_set('twitter-datasets/test_data.txt')

    return tweets, labels, X_test

def get_dataset_stopword_shuffle():
    tweets = []
    labels = []
    def load_tweets(filename, label):
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                tweets.append(" ".join([x for x in line.rstrip().split(" ") if x not in nltk_stopwords]))
                labels.append(label)
                if line.find(" , ") != -1:
                    split = line.rstrip().split(" , ")
                    random.shuffle(split)
                    shuffled = " , ".join(split).split(" ")
                    tweets.append(" ".join([x for x in shuffled if x not in nltk_stopwords]))
                    labels.append(label)
    
    load_tweets('twitter-datasets/train_neg_full.txt', 0)
    load_tweets('twitter-datasets/train_pos_full.txt', 1)

    # Convert to NumPy array to facilitate indexing
    tweets = np.array(tweets)
    labels = np.array(labels)

    def get_test_set(filename):
        twt = []
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                twt.append(" ".join([x for x in line.split(",", 1)[1].rstrip().split(" ") if x not in nltk_stopwords]))

        return twt

    X_test = get_test_set('twitter-datasets/test_data.txt')

    return tweets, labels, X_test

def get_dataset_stopword_trans(target_lang="de"):
    import queue
    import threading 
    import time
    tweets = []
    labels = []
    def load_tweets(filename, label):
        queueLock = threading.Lock()
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.lower()
                tweets.append(" ".join([x for x in line.rstrip().split(" ") if x not in nltk_stopwords]))
                labels.append(label)

        workQueue = queue.Queue(len(tweets))
        queueLock.acquire()
        for twt in tweets:
            workQueue.put(twt)
        queueLock.release()
        print("+++")

        class myThread (threading.Thread):
            def __init__(self, lbl):
                threading.Thread.__init__(self)
                self.label = lbl
            def run(self):
                while True:
                    queueLock.acquire()
                    if not workQueue.empty():
                        try:
                            twt = []
                            for i in range(100):
                                if not workQueue.empty():
                                    twt.append(workQueue.get())
                            #print(twt)
                            queueLock.release()
                            translated = ts.google(ts.google("\n".join(twt), "en", target_lang), target_lang, "en")
                            for twt_trans in translated.split("\n"):
                                twt_trans = twt_trans.lower().split(" ")
                                tweets.append(" ".join([x for x in twt_trans if x not in nltk_stopwords]))
                                labels.append(self.label)
                        except:
                            continue
                    else:
                        queueLock.release()
                        return 

        threads = []
        for i in range(500):
            thread = myThread(label)
            thread.start()
            threads.append(thread)


        while not workQueue.empty():
            print(workQueue.qsize())
            time.sleep(1)
            pass
        
        for t in threads:
            t.join()
    
    load_tweets('twitter-datasets/train_neg_full.txt', 0)
    load_tweets('twitter-datasets/train_pos_full.txt', 1)

    # Convert to NumPy array to facilitate indexing
    tweets = np.array(tweets)
    labels = np.array(labels)

    def get_test_set(filename):
        twt = []
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                twt.append(" ".join([x for x in line.split(",", 1)[1].rstrip().split(" ") if x not in nltk_stopwords]))

        return twt

    X_test = get_test_set('twitter-datasets/test_data.txt')

    return tweets, labels, X_test


def get_dataset_stopword_synonyms():
    tweets = []
    labels = []

    import json

    word2syn = {}

    # from here https://raw.githubusercontent.com/zaibacu/thesaurus/master/en_thesaurus.jsonl
    with open('./log/en_thesaurus.jsonl', 'r') as json_file:
        json_list = list(json_file)

    for json_str in json_list:
        result = json.loads(json_str)
        if result["word"] not in word2syn.keys():
            word2syn[result["word"]] = result["synonyms"]
        else:
            word2syn[result["word"]] += result["synonyms"]

    def load_tweets(filename, label):
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.lower()
                tweets.append(" ".join([x for x in line.rstrip().split(" ") if x not in nltk_stopwords]))
                labels.append(label)

                list_words = line.rstrip().split(" ")
                idx_word = list(enumerate(list_words))
                random.shuffle(idx_word)
                for idx, x in idx_word:
                    if x in word2syn.keys() and len(word2syn[x]):
                        list_words[idx] = random.choice(word2syn[x])
                        new_twt = " ".join(list_words)
                        tweets.append(" ".join([x for x in new_twt.rstrip().split(" ") if x not in nltk_stopwords]))
                        labels.append(label)
                        break # only change one word
    
    load_tweets('twitter-datasets/train_neg_full.txt', 0)
    load_tweets('twitter-datasets/train_pos_full.txt', 1)

    # Convert to NumPy array to facilitate indexing
    tweets = np.array(tweets)
    labels = np.array(labels)

    def get_test_set(filename):
        twt = []
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                twt.append(" ".join([x for x in line.split(",", 1)[1].rstrip().split(" ") if x not in nltk_stopwords]))

        return twt

    X_test = get_test_set('twitter-datasets/test_data.txt')

    return tweets, labels, X_test



def get_dataset_morestopword():
    tweets = []
    labels = []
    def load_tweets(filename, label):
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                tweets.append(" ".join([x for x in line.rstrip().split(" ") if x not in nltk_stopwords + manually_filtering]))
                labels.append(label)
    
    load_tweets('twitter-datasets/train_neg_full.txt', 0)
    load_tweets('twitter-datasets/train_pos_full.txt', 1)

    # Convert to NumPy array to facilitate indexing
    tweets = np.array(tweets)
    labels = np.array(labels)

    def get_test_set(filename):
        twt = []
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                twt.append(" ".join([x for x in line.split(",", 1)[1].rstrip().split(" ") if x not in nltk_stopwords + manually_filtering]))

        return twt

    X_test = get_test_set('twitter-datasets/test_data.txt')

    return tweets, labels, X_test

def get_dataset_commonword(max_dic_size=5000):
    tweets = []
    def load_tweets(filename, label):
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                tweets.append(line.rstrip())
    
    load_tweets('twitter-datasets/train_neg_full.txt', 0)
    load_tweets('twitter-datasets/train_pos_full.txt', 1)

    # Convert to NumPy array to facilitate indexing
    tweets = np.array(tweets)
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(max_features=max_dic_size)
    vectorizer.fit_transform(tweets)

    tweets = []
    labels = []
    def load_tweets_commonword(filename, label):
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                tweets.append(" ".join([x for x in line.rstrip().split(" ") if x in vectorizer.vocabulary_.keys()]))
                labels.append(label)
    
    load_tweets_commonword('twitter-datasets/train_neg_full.txt', 0)
    load_tweets_commonword('twitter-datasets/train_pos_full.txt', 1)
    tweets = np.array(tweets)
    labels = np.array(labels)

    def get_test_set(filename):
        twt = []
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                twt.append(" ".join([x for x in line.split(",", 1)[1].rstrip().split(" ") if x in vectorizer.vocabulary_.keys()]))

        return twt

    X_test = get_test_set('twitter-datasets/test_data.txt')

    return tweets, labels, X_test