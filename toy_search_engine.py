'''
@author: Sivasubramanian Sivaramakrishnan
'''
import os
import math
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
doc_dictonary = {}
idf_dictionary = {}
dis_dict = {}
normalized_dict = {}
query_dict = {}
corpus_root = '/Users/Seshanth/Desktop/Programming assignment/presidential_debates'
for filename in os.listdir(corpus_root):
    file = open(os.path.join(corpus_root, filename), "r", encoding='UTF-8')
    doc = file.read()
    file.close()
    doc = doc.lower()
    doc = doc.lower()
    tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
    tokens = tokenizer.tokenize(doc)
    stops = set(stopwords.words('english'))
    word_dictionary = {}
    for word in tokens:
        if word.lower() not in stops:
            new_word = stemmer.stem(word)
            if new_word in word_dictionary.keys():
                word_dictionary[new_word] += 1
            else:
                word_dictionary[new_word] = 1
            
        doc_dictonary[filename] = word_dictionary
#Generating the inverse document frequency of the word
def getidf(token):
    df_value = 0
    doc_count=len(doc_dictonary)
    for name,another_dic in doc_dictonary.items():
        if token in another_dic.keys():
            df_value=df_value+1
    if df_value==0:
        return -1
    return math.log10(doc_count/df_value)
#Calculating the count value of the token
def getcount(token):
    count = 0
    for name, another_dic in doc_dictonary.items():
        if token in another_dic.keys():
            count+=another_dic[token]
    return count
#This function calculates the Tf-idf value for the token
def document_vector():
    for name, inner_dict in doc_dictonary.items():
        tf_idf_dict = {}
        for key,value in inner_dict.items():
            tf_value = 1 + math.log10(value)
            idf_value = getidf(key)
            tf_idf=tf_value*idf_value
            tf_idf_dict[key] = tf_idf
        idf_dictionary[name] = tf_idf_dict
#Euclidean distance is calculated with tfidf values
def euclidean():
    for filename , inner_dic in idf_dictionary.items():
        sqaure_plus = 0
        for word, tf_idf_value in inner_dic.items():
            sqaure_plus = sqaure_plus+(tf_idf_value*tf_idf_value)
        dis_dict[filename] = math.sqrt(sqaure_plus)
#IDf values are normalized in this function
def normalization():
    normalize = 0
    for filename,inner_dic in idf_dictionary.items():
        new_vector = {}
        for word,idf_value in inner_dic.items():
            g=dis_dict[filename]
            normalize = idf_value/g
            new_vector[word] = normalize
        normalized_dict[filename] = new_vector
def query_vector(token):
    tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
    tokens = tokenizer.tokenize(token)
    stops = set(stopwords.words('english'))
    query_dictionary = {}
    query_tfdict = {}
    tf_value=0
    for word in tokens:
        if word.lower() not in stops:
            new_word = stemmer.stem(word)
            if new_word in query_dictionary.keys():
                query_dictionary[new_word] += 1
            else:
                query_dictionary[new_word] = 1
    euc_dis=0
    for name,frequency in query_dictionary.items():
        tf_value = 1 + math.log10(frequency)
        query_tfdict[name] = tf_value
        euc_dis += tf_value*tf_value
    query_norm_tfdict = {}
    for word,un_normalized_val in query_tfdict.items():
        query_norm_tfdict[word] = un_normalized_val/math.sqrt(euc_dis)
    return query_norm_tfdict
#Returns the tf-idf weight of a token in a document name filename
def getweight(filename,token):
    new_term=0
    dot_pro=0
    doc1=normalized_dict[filename]
    query_tfdict = query_vector(token)
    for name,value_idf in query_tfdict.items():
        if name in doc1.keys():
            dot_pro = dot_pro + value_idf*doc1[name]
        else:
            return 0;
    return dot_pro
document_vector()
euclidean()
normalization()
print("%.12f" % getidf("health"))
print("%.12f" % getidf("agenda"))
print("%.12f" % getidf("vector"))
print("%.12f" % getidf("reason"))
print("%.12f" % getidf("hispan"))
print("%.12f" % getidf("hispanic"))
print("%.12f" % getweight("1960-10-21.txt","reason"))
print("%.12f" % getweight("1976-10-22.txt","agenda"))
print("%.12f" % getweight("2012-10-16.txt","hispan"))
print("%.12f" % getweight("2012-10-16.txt","hispanic"))
