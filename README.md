### Models

all baselines, and all data prepocessing methods

### Install

install required packages(mainly torch, sklearn, numpy, gensim)
- for synonyms, download https://raw.githubusercontent.com/zaibacu/thesaurus/master/en_thesaurus.jsonl to `./log/en_thesaurus.jsonl`

### Files

`./twitter-datasets`: contains all data files

`./embd/twt_data`: only copy train_neg_full.txt and train_pos_full.txt here
- if you want to train a word2vec model from scratch, this is required by gensim-API

### Reproducibility

baselines
- countvectorizer baseline  : `python project_2.py`
- word2vec baseline         : `python word2vec.twitter.py`
- word2vec lstm             : `python word2vec.twitter_lstm.py`
- word2embd lstm            : `python word2embd.twitter.py`

proprecessings:stop words
- nltk stop words: `python word2embd_stpwd.twitter.py`
- top5k: `python word2embd_commonword.twitter.py`
- nltk+: `python word2emnd_morestpwd.twitter.py`

proprecessings:data augmentations
- clause order: `python word2embd_stpwd.shuffle.twitter.py`
- synonyms: `python word2embd_stpwd.synonyms.twitter.py`
- translation: `python word2embd_stpwd.trans.twitter.py`