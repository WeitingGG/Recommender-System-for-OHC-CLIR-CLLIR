#coding:utf-8
import numpy as np
import string
import lda
import jieba
import codecs
import nltk
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
#nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer  

class LDA_v20161130(object):
    def __init__(self, topics=2):
        self.n_topic = topics
        self.corpus = None
        self.vocab = None
        self.ppCountMatrix = None
        # self.stop_words = [
        #     u'，', 
        #     u'。', 
        #     u'、', 
        #     u'（', 
        #     u'）', 
        #     u'·', 
        #     u'！', 
        #     u' ', 
        #     u'：', 
        #     u'“', 
        #     u'”', 
        #     u'\n'
        # ]
        self.model = None
        self.total_num = None
        self.stop = [
            u'!', u'@', u'#', u',', u'.', u'/', u';', u' ', u'[',
            u']', u'$', u'%', u'^', u'&', u'*', u'(', u')', u'"',
            u':', u'<', u'>', u'?', u'{', u'}', u'=', u'+', u'_',
            u'-', u''''''
        ]
        
    def loadCorpusFromFile(self, fn):
        # 中文分词
        f = open(fn, 'r',encoding="ISO-8859-1")
        text = f.readlines()
        self.total_num = len(text)
        print(self.total_num)
        text = r' '.join(text)
        #小写化
        text = text.lower()
        #去除特殊标点
        for c in string.punctuation:
            text = text.replace(c, ' ')
        #分词
        wordLst = nltk.word_tokenize(text)
        #去除停用词
        filtered = [w for w in wordLst if w not in stopwords.words('english')]
        #仅保留名词或特定POS
        refiltered =nltk.pos_tag(filtered)
        filtered = [w for w, pos in refiltered if pos.startswith('NN')]
        wordnet_lemmatizer = WordNetLemmatizer() 
        filtered = [wordnet_lemmatizer.lemmatize(w) for w in filtered]
        #词干化
        #ps = PorterStemmer()
        #filtered = [ps.stem(w) for w in filtered]
        seg_list = " ".join(filtered)
        #print(text)
#        seg_generator = jieba.cut(text)
#        seg_list = [i for i in seg_generator if i not in self.stop_words]
#        seg_list = r' '.join(seg_list)
        # 切割统计所有出现的词纳入词典
        seglist = seg_list.split(" ")
        mp = {}
        max_freq = 0
        for word in seglist:
            if (word != u' '):
                mp[word] = mp.get(word, 0) + 1
                max_freq = max(max_freq, mp[word])
        _mp = {
            k:v 
            for k,v in mp.items() 
            if v >= 5 and v < self.total_num * 0.7
        }
        _mp.pop("breast", None)
        _mp.pop("cancer", None)
        _mp.pop("n", None)
        _mp.pop("in", None)
        _mp.pop("s", None)
        _mp.pop("he", None)
        _mp.pop("m", None)
        _mp.pop("im", None)
        _mp.pop("can", None)
        _mp.pop("mso", None)
        _mp.pop("get", None)
        _mp.pop("one", None)
        _mp.pop("out", None)
        self.vocab = list(_mp.keys())
        print(len(self.vocab))
        CountMatrix = []
        f.seek(0, 0)
        # 统计每个文档中出现的词频
        for line in f:
            # 置零
            count = np.zeros(len(self.vocab),dtype=np.int)
            text = line.strip()
            # 但还是要先分词
            seg_generator = jieba.cut(text)
            seg_list = [i for i in seg_generator if i not in self.stop_words]
            seg_list = r' '.join(seg_list)
            seglist = seg_list.split(" ")
            # 查询词典中的词出现的词频
            for word in seglist:
                if word in self.vocab:
                    count[self.vocab.index(word)] += 1
            CountMatrix.append(count)            
        f.close()
        #self.ppCountMatrix = (len(CountMatrix), len(self.vocab))
        self.ppCountMatrix = np.array(CountMatrix)
        print ("load corpus from %s success!"%fn)
    def setStopWords(self, word_list):
        self.stop_words = word_list
    def fitModel(self, n_iter = 1500, _alpha = 0.1, _eta = 0.01):
        self.model = lda.LDA(
            n_topics=self.n_topic, 
            n_iter=n_iter, 
            alpha=_alpha, 
            eta= _eta, 
            random_state= 1
        )
        self.model.fit(self.ppCountMatrix)
    def printTopic_Word(self, n_top_word = 8):
        for i, topic_dist in enumerate(self.model.topic_word_):
            topic_words = \
                np.array(self.vocab)[
                    np.argsort(topic_dist)
                ][
                    :-(n_top_word + 1):-1
                ]
            print ("Topic:",i,"\t"),
            for word in topic_words:
                print (word),
            print
    def printDoc_Topic(self):
        for i in range(len(self.ppCountMatrix)):
            print ("Doc %d:((top topic:%s) topic distribution:%s)"%(
                i, self.model.doc_topic_[i].argmax(),self.model.doc_topic_[i]
            ))
    def printVocabulary(self):
        print ("vocabulary:")
        for word in self.vocab:
            print (word),
        print
    def saveVocabulary(self, fn):
        f = codecs.open(fn, 'w', 'utf-8')
        for word in self.vocab:
            f.write("%s\n"%word)
        f.close()
    def saveTopic_Words(self, fn, n_top_word = -1):
        if n_top_word==-1:
            n_top_word = len(self.vocab)
        f = codecs.open(fn, 'w', 'utf-8')
        for i, topic_dist in enumerate(self.model.topic_word_):
            topic_words = np.array(self.vocab)[
                np.argsort(topic_dist)
            ][
                :-(n_top_word + 1):-1
            ]
            f.write( "Topic:%d\t"%i)
            for word in topic_words:
                f.write("%s "%word)
            f.write("\n")
        f.close()
    def saveDoc_Topic(self, fn):
        f = codecs.open(fn, 'w', 'utf-8')
        nums = np.zeros(self.n_topic)
        for i in range(len(self.ppCountMatrix)):
            f.write("Doc %d:((top topic:%s) topic distribution:%s)\n"%(
                i, self.model.doc_topic_[i].argmax(), self.model.doc_topic_[i]
            ))
            nums[self.model.doc_topic_[i].argmax()] += 1
        for i in range(self.n_topic):
            print("%f"%(nums[i] * 100 / self.total_num))
        f.close()

# if __name__=="__main__":
#     _lda = LDA_v20161130(topics=20)
#     _lda.setStopWords(_lda.stop)
#     _lda.loadCorpusFromFile(u'../../all_posts_intial.txt')
#     _lda.fitModel(n_iter=1500)
#     _lda.printTopic_Word(n_top_word=10)
#     #_lda.printDoc_Topic()
#     _lda.saveVocabulary(u'./decsion_vocab.txt')
#     _lda.saveTopic_Words(u'./decision_topic_word.txt')
#     _lda.saveDoc_Topic(u'./decision_doc_topic.txt')

