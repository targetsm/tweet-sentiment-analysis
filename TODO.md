### TODO

- get test data wrong, rerun all results
    - count tokenizer --> running --> done
    - glove.twitter --> running --> done
    - word2vec.twitter --> running
    - word2embd.twitter
- test some stop words
    - nltk --> running --> done
    - more stop work from nltk
        - manually filter some symbols from top 500 words
    - count --> less overfitting
        - 5000 most frequent running --> done
    - logistic regression --> does it work really?
- padding for training and test data?
    - at least the model learns the length of twitter
    - and yes, we cannot remove padding for test data, weird though
    - finally, the pack padded sequence is working
        - add a model then --> running

### baselines

- countvectorizer + logistic:       79.40
- word2vec, mean vector:            72.38
- word2vec, lstm:                   72.74

ours start from here
- wordembedding, lstm:              83.12
    - this is the "word2vec, lstm" with trainable embeddings
    - note everything is the same, including embedding size
- wordembedding, stpwd-nltk         84.06, 84.04 for lower case input
    - ignore words from nltk stopword list
- wordembedding, top5k words        82.78
    - train only on top 5k words, this is to compare with countvectorizer one, to show sequential prediction is better than bag-of-words
- wordembedding, stpwd-nltk+        82.34
    - I manually also filtered some stop words, but it does not work

ok stop word stops here
- questions from here: given 25 dim learnable word embedding, what is the best we can achieve from it?
    - 2 layers: 84.30
        - okay, 2 layers seems enough(?), what about the width
        - h64:  83.30
        - h128: 81.88
    - 3 layers: 84.26
- another direction: data augmentations!
    - altering some clause order    83.98
        - foo, bar --> bar, foo
        - ok, maybe some semantic information lost here
    - translation: EN-DE            84.62
        - working on multithreadings...running...
    - synonyms                      83.40
        - change one word in the sentence: enforce some robust embeddings maybe
        - running...

reproduce some baselines
- distillBERT with stop words
- fill in some citations
- intro & discussion