import os
import pandas as pandas
import ijson
from nltk.stem.porter import PorterStemmer
from collections import defaultdict
import math
import time

START_TIME = time.time()
TEST_FOLDER_PATH = "/Users/vasanthmahendran/Workspace/Data/nbc_new/test_new_1"
TRAIN_FOLDER_PATH = "/Users/vasanthmahendran/Workspace/Data/nbc_new/train_new"
COLUMN_NAMES = ['Ratings', 'AuthorLocation', 'Title', 'Author', 'ReviewID', 'Content', 'Date']
STEMMER = PorterStemmer()
class NaiveBayesClassification(object):
    def __init__(self):
        print("--- Reading: %s minutes ---" % round(((time.time() - START_TIME) / 60), 2))
        self.pd_datas_train = self.parsefiles(TRAIN_FOLDER_PATH)
        self.pd_datas_test = self.parsefiles(TEST_FOLDER_PATH)
        print("--- Stemming: %s minutes ---" % round(((time.time() - START_TIME) / 60), 2))
        self.pd_datas_train['class'] = self.pd_datas_train['Ratings'].map(lambda x: 'positive' if float(x) > 3 else 'negative')
        self.pd_datas_train['Content'] = self.pd_datas_train['Content'].map(lambda x: self.pre_processing(x))
        self.pd_datas_test['class'] = self.pd_datas_test['Ratings'].map(lambda x: 'positive' if float(x) > 3 else 'negative')
        self.pd_datas_test['Content'] = self.pd_datas_test['Content'].map(lambda x: self.pre_processing(x))
        self.frequency = set()
        self.frequency_positive = defaultdict(int)
        self.frequency_negative = defaultdict(int)
        self.df_positive = defaultdict(int)
        self.df_negative = defaultdict(int)
        self.tf_idf_score_positive = defaultdict(int)
        self.tf_idf_score_negative = defaultdict(int)
        self.frequency_total = len(self.pd_datas_train.index)
        
        self.N_Value = len(self.pd_datas_train.index)
        pd_datas_train_positives= self.pd_datas_train.loc[self.pd_datas_train['class'] == 'positive']
        pd_datas_train_negatives= self.pd_datas_train.loc[self.pd_datas_train['class'] == 'negative']
        self.positive_probability_freq = len(pd_datas_train_positives.index)
        self.negative_probability_freq = len(pd_datas_train_negatives.index)
        print("--- Building TF_IDF model: %s minutes ---" % round(((time.time() - START_TIME) / 60), 2))
        pd_datas_train_positives['Content'].apply(lambda x: self.build_frequency_positive(x))
        pd_datas_train_negatives['Content'].apply(lambda x: self.build_frequency_negative(x))
        self.bin = len(self.frequency)
        self.build_tf_idf_positive();
        self.build_tf_idf_negative();
        print("--- NBC: %s minutes ---" % round(((time.time() - START_TIME) / 60), 2))
        self.pd_datas_test = self.pd_datas_test.merge(self.pd_datas_test.apply(self.process, axis=1), left_index=True, right_index=True)
        self.calculate_measures(self.pd_datas_test['class'], self.pd_datas_test['predicted_class'])
        print('accuracy-----',self.accuracy)
        print('fmeasure-----',self.fmeasure)
        print('precision-----',self.precision)
        print('recall-----',self.recall)
        print('fpr-----',self.fpr)
        print('no of train records-----',len(self.pd_datas_train.index))
        print('no of test records-----',len(self.pd_datas_test.index))
        print("--- Done: %s minutes ---" % round(((time.time() - START_TIME) / 60), 2))

    
    def build_tf_idf_positive(self):
        for x in self.frequency_positive:   
            tf_weight = (1 + math.log10(self.frequency_positive[x]))
            idf_weight = (math.log10(self.N_Value / self.df_positive[x]))
            tf_idf_weight = tf_weight * idf_weight
            self.tf_idf_score_positive[x] = tf_idf_weight
    
    def build_tf_idf_negative(self):
        for x in self.frequency_negative:   
            tf_weight = (1 + math.log10(self.frequency_negative[x]))
            idf_weight = (math.log10(self.N_Value / self.df_negative[x]))
            tf_idf_weight = tf_weight * idf_weight
            self.tf_idf_score_negative[x] = tf_idf_weight

    def parsefiles(self,folder_path):
        pd_datas = pandas.DataFrame([], columns=COLUMN_NAMES)
        for file in os.listdir(folder_path):
            data = []
            if file.endswith(".json"):
                file_path = os.path.join(folder_path, file)
                if os.path.isfile(file_path):
                    with open(file_path, 'r') as file:
                        objects = ijson.items(file, 'Reviews.item')
                        for row in objects:
                            selected_row = []
                            for item in COLUMN_NAMES:
                                if item == 'Ratings':
                                    selected_row.append(row[item]['Overall'])
                                else:
                                    selected_row.append(row[item])
                            data.append(selected_row)
                        pd_datas = pd_datas.append(pandas.DataFrame(data, columns=COLUMN_NAMES),ignore_index=True)
        return pd_datas
    
    def pre_processing(self, s):
        try:
            if isinstance(s, str):
                s = s.lower()
                escape_letters = ["$","  ","?",",","//","..","."," . "," / ","-"," \\"]
                for escape_letter in escape_letters:
                    s = s.replace(escape_letter," ")
                s = (" ").join([STEMMER.stem(z) for z in s.split(" ")])
                return s
            else:
                return " "
        except Exception as error:
            print("str causing error",s,repr(error))
            return " "
    
    def process(self,x):
        words = x['Content'].split(" ")
        positive_prob_product = 1
        negative_prob_product = 1
        for word in words:
            if word:
                positive_prob = (self.tf_idf_score_positive[word] + 1)/(self.bin+self.frequency_positive_total)
                negative_prob = (self.tf_idf_score_positive[word] + 1)/(self.bin+self.frequency_negative_total)
                positive_prob_product = positive_prob_product * positive_prob
                negative_prob_product = negative_prob_product * negative_prob
        
        positive_prob_product = positive_prob_product * (self.positive_probability_freq/self.frequency_total)
        negative_prob_product = negative_prob_product * (self.negative_probability_freq/self.frequency_total)

        if positive_prob_product > negative_prob_product :
            return pandas.Series(dict(predicted_class="positive"))
        else:
            return pandas.Series(dict(predicted_class="negative"))
    
    
    def build_frequency_positive(self,x):
        row = []
        word_set = x.split(" ")
        for word in word_set:
            if word:
                self.frequency_positive_total = +1
                self.frequency_positive[word] += 1
                if(word in self.frequency):
                    self.frequency.add(word)
                if word not in row:
                    row.append(word)
                    self.df_positive[word] += 1
    
    def build_frequency_negative(self,x):
        row = []
        word_set = x.split(" ")
        for word in word_set:
            if word:
                self.frequency_negative_total = +1
                self.frequency_negative[word] += 1
                if(word in self.frequency):
                    self.frequency.add(word)
                if word not in row:
                    row.append(word)
                    self.df_negative[word] += 1
    
    def calculate_measures(self,labeled_class, predicted_class):
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for idx, row in predicted_class.iteritems():
            if row == 'positive':
                if row == labeled_class[idx]:
                    tp += 1
                if row != labeled_class[idx]:
                    fp += 1
            if row == 'negative':
                if row == labeled_class[idx]:
                    tn += 1
                if row != labeled_class[idx]:
                    fn += 1
        
        self.accuracy =  (tp+tn) / len(labeled_class)
        self.precision = tp/(tp+fp)
        self.recall = tp/(tp+fn)
        self.fmeasure = (2*tp)/((2*tp)+fp+fn)
        self.fpr = fp/(fp+tn)
            
            
NBC = NaiveBayesClassification()