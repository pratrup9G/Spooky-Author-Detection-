## Import required libraries
import numpy as np
import pandas as pa
import seaborn as sn 

import nltk
from nltk.tokenize import word_tokenize,sent_tokenize,TweetTokenizer
import string
import re
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split,KFold
import matplotlib.pyplot as plt


from sklearn.base import BaseEstimator,ClassifierMixin
from sklearn.utils.validation import check_is_fitted,check_X_y
from sklearn.linear_model import LogisticRegression
from scipy import sparse
from sklearn.naive_bayes import MultinomialNB
from sklearn import naive_bayes,metrics


import warnings 
warnings.filterwarnings('ignore')
tokneizer = TweetTokenizer()
lem = WordNetLemmatizer()
engstopwords = set(stopwords.words('english'))
random_state = 42




## Read data
train = pa.read_csv('train.csv')
test = pa.read_csv('test.csv')

##Meta Features 
def preprocessing(data):
    #word
    data['len_words'] = data['text'].apply(lambda x:len(str(x).split()))
    #charachter
    data['max_char'] = data['text'].apply(lambda x:len(str(x)))
    #punctuation
    data['total_punc'] = data['text'].apply(lambda x: len([punc for punc in str(x) if punc in string.punctuation]))
    #uppercase
    data['upper_case'] = data['text'].apply(lambda x: len([word for word in str(x).split() if word.isupper()]))
    #stopwords
    data['total_stopwords'] = data['text'].apply(lambda x: len([word for word in str(x).lower().split() if word in engstopwords]))
    #avg length of each words
    data['avg_length'] = data['text'].apply(lambda x: np.mean([len(word) for word in str(x).split()]))
    #unique words in text
    data['unique_words'] = data['text'].apply(lambda x:len(set(str(x).split())))
    #avg unique words in text
    data['avg_unique_words'] = data['text'].apply(lambda x:np.mean([len(word) for word in set(str(x).split())]))

    return data

##Add the meta features
train = preprocessing(train)
test = preprocessing(test)

APPO = {
"aren't" : "are not",
"can't" : "cannot",
"couldn't" : "could not",
"didn't" : "did not",
"doesn't" : "does not",
"don't" : "do not",
"hadn't" : "had not",
"hasn't" : "has not",
"haven't" : "have not",
"he'd" : "he would",
"he'll" : "he will",
"he's" : "he is",
"i'd" : "I would",
"i'd" : "I had",
"i'll" : "I will",
"i'm" : "I am",
"isn't" : "is not",
"it's" : "it is",
"it'll":"it will",
"i've" : "I have",
"let's" : "let us",
"mightn't" : "might not",
"mustn't" : "must not",
"shan't" : "shall not",
"she'd" : "she would",
"she'll" : "she will",
"she's" : "she is",
"shouldn't" : "should not",
"that's" : "that is",
"there's" : "there is",
"they'd" : "they would",
"they'll" : "they will",
"they're" : "they are",
"they've" : "they have",
"we'd" : "we would",
"we're" : "we are",
"weren't" : "were not",
"we've" : "we have",
"what'll" : "what will",
"what're" : "what are",
"what's" : "what is",
"what've" : "what have",
"where's" : "where is",
"who'd" : "who would",
"who'll" : "who will",
"who're" : "who are",
"who's" : "who is",
"who've" : "who have",
"won't" : "will not",
"wouldn't" : "would not",
"you'd" : "you would",
"you'll" : "you will",
"you're" : "you are",
"you've" : "you have",
"'re": " are",
"wasn't": "was not",
"we'll":" will",
"didn't": "did not",
"tryin'":"trying"
}

punctuation = string.punctuation

def data_cleaning(data):
    ##convert to lower case
    data_lower = data.lower()
    ##word tokenize using tweet tokenizer
    words = tokneizer.tokenize(data_lower)
    ##convert appos
    words = [APPO[word] if word in APPO else word for word in words]
    ##remove the stopwords
    words = [word for word in words if not word in engstopwords]
    ##lemmaize
    words = [lem.lemmatize(word,'v')  for word in words]
    ##remove punctuation
    words = [word for word in words if not word in punctuation]
    
    clean_sent = " ".join(words)
    return clean_sent

cleaned_train = train['text'].apply(lambda x:data_cleaning(x))
cleaned_test = test['text'].apply(lambda x:data_cleaning(x))

#cleaned train and test data
train['text'] = cleaned_train.values
test['text'] = cleaned_test.values

author = {'EAP':0,'HPL':1,'MWS':2}
train_y = train['author'].map(author)
train_id = train['id']


#Visualize number of used words using wordclouds
text = train['text'].values
wc = WordCloud(background_color='black',max_words=400,stopwords=engstopwords)
wc.generate(" ".join(text))
plt.axis('off')
plt.title("Word Frequency counting for all authors combined")
plt.imshow(wc.recolor(colormap='summer',random_state=42),alpha=0.98)

## Build model using only the meta features and check the accuracy and feature importance
meta_features = ['len_words','max_char','total_punc','upper_case','total_stopwords','avg_length','unique_words','avg_unique_words']
train_meta_features = train[meta_features]



best_param=[]
##split the data
def XGBoost(x_train,y_train,x_test,y_test,x_original_test=None,GRID=False):
    num_rounds = 2000
    
    if GRID==True:
         model = XGBClassifier(num_class=3,objective='multi:softprob',silent=1)
         #crossvaildation using gridsearch cv
         params ={'learning_rate':[0.09],
                  'max_depth':[5,6],
                  'n_estimators':[150,200],
                  'min_child_weight':[1,2,3],
                  'colsample_bytree':[0.3,0.5,1]}
    
         gridsearh = GridSearchCV(model,param_grid=params,cv=5,scoring='f1_weighted')
         grid_xgb = gridsearh.fit(x_train,y_train)
         print("Best scores is",grid_xgb.best_score_)
         print("Best parameter is",grid_xgb.best_estimator_)


         #best parameters
         best_param.append(grid_xgb.best_estimator_)
         best_param.append(grid_xgb.best_params_)
         return grid_xgb.best_estimator_

    else:
        
         #train the model
         xgtrain = xgb.DMatrix(x_train,label=y_train)
         xgtest = xgb.DMatrix(x_test,label=y_test)
         xgotest = xgb.DMatrix(x_original_test)
    
         watchlist = [(xgtrain,'train'),(xgtest,'test')]
         #tuned parameters of xgboost
         params = best_param[1]
         
         #default parameters of xgboost 
         params['objective'] = 'multi:softprob'
         params['num_class'] = 3
         params['eval_metric'] = 'mlogloss'
         params['silent'] = 1
         params['seed'] = 0
         
    
         #plst = list(params.items())
         xgmodel = xgb.train(params,xgtrain,num_rounds,watchlist,early_stopping_rounds=50,verbose_eval=20)
         pred_x_test = xgmodel.predict(xgtest,ntree_limit=xgmodel.best_ntree_limit)
         pred_xo_test = xgmodel.predict(xgotest,ntree_limit=xgmodel.best_ntree_limit)
    
         return pred_x_test,pred_xo_test,xgmodel   


#SPLIT THE DATA
X_train,X_test,Y_train,Y_test = train_test_split(train_meta_features,train_y,test_size=0.3,random_state=random_state)
xgb_model_best = XGBoost(X_train,Y_train,X_test,Y_test,GRID=True)


kf = KFold(n_splits=5,shuffle=True,random_state=42)
pred_test_full = 0
cv_score = []
i=1

#(Cross validation) custom cross validation

for train_index,test_index in kf.split(train_meta_features):
    print('{} in K Fold {}'.format(i,kf.n_splits))
    xtrain,xvalid = train_meta_features.loc[train_index],train_meta_features.loc[test_index]
    ytrain,yvalid = train_y[train_index],train_y[test_index]
    
    pred_valid_test,pred_original_test,xgmodel = XGBoost(xtrain,ytrain,xvalid,yvalid,test[meta_features],GRID=False)
    pred_test_full +=pred_original_test
    cv_score.append(xgmodel.best_score)
    i+=1


## PIPEINE 1--->
    
    
##Pipelines differnt combination for making new feaatures
#pipeline one for word vectoizer using TFIDF
pipeline_word_Tfidf = Pipeline([('vectorizer',TfidfVectorizer(strip_accents='unicode',analyzer='word',ngram_range=(1,3),stop_words='english')),
                                ('model',MultinomialNB())])
#finding the optimal parameters of tfidf vectorizer
parameters_tf_one = {'vectorizer__max_features':(500,1000,2000,5000),
                     'vectorizer__min_df':(2,3,4)}
#grid search on the whole train dataset
grid_piepeone = GridSearchCV(pipeline_word_Tfidf,parameters_tf_one)
grid_piepeone.fit(train['text'],train_y)


#parameters of tfidf word 
parms_tf_word = grid_piepeone.best_params_

parms_tf_word

##Vectorize the data using the best parameters of tfidf
tfid_vec = TfidfVectorizer(max_features=500,min_df=2,ngram_range=(1,3),stop_words='english',analyzer='word',strip_accents='unicode')
text_vec  = tfid_vec.fit_transform(cleaned_train.tolist() + cleaned_test.tolist())
train_vec = tfid_vec.transform(cleaned_train.tolist())
test_vec = tfid_vec.transform(cleaned_test.tolist())



#build model naive bayes with logistical regression as it performs much better in word based problems

train_y_dummies = pa.get_dummies(train_y)
train_y_dummies.columns =['EAP','HPL','MWS']


## Naive Bayes with Support Vector machine or logistic regression
class NbSvmClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, dual=False, n_jobs=1):
        self.C = C
        self.dual = dual
        self.n_jobs = n_jobs

    def predict(self, x):
        # Verify that model has been fit
        check_is_fitted(self, ['_r', '_clf'])
        return self._clf.predict(x.multiply(self._r))

    def predict_proba(self, x):
        # Verify that model has been fit
        check_is_fitted(self, ['_r', '_clf'])
        return self._clf.predict_proba(x.multiply(self._r))

    def fit(self, x, y):
        # Check that X and y have correct shape
        y = y.values
        x, y = check_X_y(x, y, accept_sparse=True)

        def pr(x, y_i, y):
            p = x[y==y_i].sum(0)
            return (p+1) / ((y==y_i).sum()+1)

        self._r = sparse.csr_matrix(np.log(pr(x,1,y) / pr(x,0,y)))
        x_nb = x.multiply(self._r)
        self._clf = LogisticRegression(C=self.C, dual=self.dual, n_jobs=self.n_jobs).fit(x_nb, y)
        return self

### Cross validation function for NB-LR/SVM


model = NbSvmClassifier(C=4, dual=True, n_jobs=-1)

kf = KFold(n_splits=5,shuffle=True,random_state=42)
pred_full_test = 0
cross_val = []
pred_full_test_list = []
pred_mat = np.zeros([test.shape[0],3])
pred_mat_t = np.zeros([train.shape[0],3])

pred_val_test_list = []
pred_val_test = 0
cross_val_test = []

for train_index,test_index in kf.split(cleaned_train):
    
    pred_full_test_list = []
    pred_val_test_list = []
    for target in train_y_dummies.columns:
        xtrain,xvalid = train_vec[train_index],train_vec[test_index]
        ytrain,yvalid = train_y_dummies[target][train_index],train_y_dummies[target][test_index]
        
        model.fit(xtrain,ytrain)

        
        pred_full_test = model.predict_proba(test_vec)[:,1]
        pred_test_y = model.predict_proba(xvalid)[:,1]
        
        pred_full_test_list.append(pred_full_test)   
        pred_val_test_list.append(pred_test_y)   
     
     
    #pred_mat_t = np.zeros([xvalid.shape[0],3])
    pred_mat_t[test_index,:] = np.array(pred_val_test_list).T
    test_y_pred = np.array(pred_val_test_list).T
    
    
    ##Calculate the loss class 0
    eap =  metrics.log_loss(yvalid,test_y_pred.T[0])
    print("loss class EAP",eap)
    cross_val_test.append(eap)
    
    
    ##Calculate the loss class 1
    hpl = metrics.log_loss(yvalid,test_y_pred.T[1])
    print("loss class HPL",hpl)
    cross_val_test.append(hpl)


    ##Calculate the loss class 2
    mws = metrics.log_loss(yvalid,test_y_pred.T[2])
    print("loss class MWS",mws)
    cross_val_test.append(mws)
    
    print("Average cross val on frist validation is ={}".format((eap+hpl+mws)/3))
    
    
    iterr = 0    
    index = 0
    
    for items in pred_full_test_list:
        index = 0    
        for i in items:
            if(index < 8392):
                pred_mat[index][iterr] += i
                #print(i)
                index +=1
        iterr +=1
        
                 
pred_mat = pred_mat/5


#Multinomial naive bayes with tuned parameters
def MUNbaiyes(train_x,train_y,test_x,test_y,X_Test):
    model = naive_bayes.MultinomialNB()
    model.fit(train_x,train_y)
    pred_test_y = model.predict_proba(test_x)
    pred_test_y2 = model.predict_proba(X_Test)
    return model,pred_test_y,pred_test_y2


#cross validataion 
kf = KFold(n_splits=5,shuffle=True,random_state=42)
pred_full_test = 0
cross_val = []

pred_train = np.zeros([train.shape[0],3])
cnt = 0
for train_index,test_index in kf.split(cleaned_train):
    xtrain,xvalid = train_vec[train_index],train_vec[test_index]
    ytrain,yvalid = train_y[train_index],train_y[test_index]
    naive_model,pred_test_y,pred_test_y2 = MUNbaiyes(xtrain,ytrain,xvalid,yvalid,test_vec)
    pred_full_test += pred_test_y2
    pred_train[test_index,:] = pred_test_y
    ##cv_scores
    cross_val.append(metrics.log_loss(yvalid,pred_test_y))
    print("Cross val score ={}".format(cross_val[cnt]))
    cnt +=1
############################################################################################################
#print("Cross validation scores ",cross_val)
print("Mean cross validation score on Multi naivebaiyes on word vecor using TFIDF Vectorizer tuned={}".format(np.mean(cross_val)))

pred_full_test /=5
## Confusion matrix
import itertools
from sklearn.metrics import confusion_matrix

### From http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py #
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    #else:
    #    print('Confusion matrix, without normalization')

    #print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


cnf_matrix = confusion_matrix(train_y, np.argmax(pred_train,axis=1))

np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure(figsize=(8,8))
plot_confusion_matrix(cnf_matrix, classes=['EAP', 'HPL', 'MWS'],
                      title='Confusion matrix, without normalization')
plt.show()


## Make features with the prediction of multi navie Baiyes
train['nb_tfid_weap'] = pred_train[:,0] 
train['nb_tfid_whwl'] = pred_train[:,1]
train['nb_tfid_wmws'] = pred_train[:,2]

test['nb_tfid_weap'] = pred_full_test[:,0]
test['nb_tfid_whwl'] = pred_full_test[:,1]
test['nb_tfid_wmws'] = pred_full_test[:,2]


## Apply Singluar Value Decomposition and compress the tfidf vectorizer into 20 features 
from sklearn.decomposition import TruncatedSVD
n_comp = 20
svdword_tf = TruncatedSVD(n_components=n_comp,algorithm='arpack')
svdword_tf.fit(text_vec)
train_svd = pa.DataFrame(svdword_tf.transform(train_vec))
test_svd = pa.DataFrame(svdword_tf.transform(test_vec))

train_svd.columns = ['svd_word_'+str(i)  for i in range(n_comp)]
test_svd.columns = ['svd_word_'+str(i)  for i in range(n_comp)]

train = pa.concat([train,train_svd],axis=1)

test = pa.concat([test,test_svd],axis=1)



# PIPELINE 2 ---> char vectoirzer using tfidf 

pipeline_char_tfidf = Pipeline([('vectorizer',TfidfVectorizer(ngram_range=(1,5),analyzer='char')),
                                 ('model',MultinomialNB())])
parameters_tfidf_char = {'vectorizer__max_features':(500,100,1500,5000),
                         'vectorizer__min_df':(3,4,5)}

grid_serach_char = GridSearchCV(pipeline_char_tfidf,parameters_tfidf_char)
grid_serach_char.fit(train['text'],train_y)

grid_serach_char.best_params_


tfidf_char_model = TfidfVectorizer(ngram_range=(1,5),analyzer='char',max_features=5000,min_df=3)
tfidf_char_model.fit_transform(cleaned_train.tolist() + cleaned_test.tolist())
train_vec_char = tfidf_char_model.transform(cleaned_train.tolist())
test_vec_char = tfidf_char_model.transform(cleaned_test.tolist())


cv_scores = []
pred_full_test = 0
pred_train = np.zeros([train.shape[0],3])
kf = KFold(n_splits=5,shuffle=True,random_state=42)
cnt=0
for train_index,test_index in kf.split(cleaned_train):
    xtrain,xvalid = train_vec_char[train_index],train_vec_char[test_index]
    ytrain,yvalid = train_y[train_index],train_y[test_index]
    naive_model,pred_test_y,pred_test_y2 = MUNbaiyes(xtrain,ytrain,xvalid,yvalid,test_vec_char)
    pred_full_test += pred_test_y2
    pred_train[test_index,:] = pred_test_y
    
    cv_scores.append(metrics.log_loss(yvalid,pred_test_y))
    print("Cross val score ={}".format(cv_scores[cnt]))
    cnt +=1

pred_full_test /=5

train['nb_tfid_ceap'] = pred_train[:,0]
train['nb_tfid_chpl'] = pred_train[:,1]
train['nb_tfid_cmws'] = pred_train[:,2]

test['nb_tfid_ceap'] = pred_full_test[:,0]
test['nb_tfid_chpl'] = pred_full_test[:,1]
test['nb_tfid_cmws'] = pred_full_test[:,2]


# PIPELINE 3 --->countvectorizer of word


#pipeline-char = Pipeline([('vectorizer',CountVectorizer())])


char_vec_word = CountVectorizer(stop_words='english', ngram_range=(1,3))
char_vec_word.fit(cleaned_train.tolist() + cleaned_test.tolist())
train_count_char = char_vec_word.transform(cleaned_train.tolist())
test_count_char = char_vec_word.transform(cleaned_test.tolist())


cv_scores = []
pred_full_test = 0
pred_train = np.zeros([train.shape[0],3])
kf = KFold(n_splits=5,shuffle=True,random_state=42)
cnt=0
for train_index,test_index in kf.split(cleaned_train):
    xtrain,xvalid = train_count_char[train_index],train_count_char[test_index]
    ytrain,yvalid = train_y[train_index],train_y[test_index]
    naive_model,pred_test_y,pred_test_y2 = MUNbaiyes(xtrain,ytrain,xvalid,yvalid,test_count_char)
    pred_full_test += pred_test_y2
    pred_train[test_index,:] = pred_test_y
    
    cv_scores.append(metrics.log_loss(yvalid,pred_test_y))
    print("Cross val score ={}".format(cv_scores[cnt]))
    cnt +=1
    

pred_full_test /=5

#add features here
train['nb_count_weap'] = pred_train[:,0]
train['nb_count_whpl'] = pred_train[:,1]
train['nb_count_wmws'] = pred_train[:,2]

test['nb_count_weap'] = pred_full_test[:,0]
test['nb_count_whpl'] = pred_full_test[:,1]
test['nb_count_wmws'] = pred_full_test[:,2]


    
# PIPELINE 4 --->count vectorizer of character


#pipeline check here
    
char_vec_word = CountVectorizer(ngram_range=(1,7),analyzer='char')
char_vec_word.fit(cleaned_train.tolist() + cleaned_test.tolist())
train_count_char = char_vec_word.transform(cleaned_train.tolist())
test_count_char = char_vec_word.transform(cleaned_test.tolist())


cv_scores = []
pred_full_test = 0
pred_train = np.zeros([train.shape[0],3])
kf = KFold(n_splits=5,shuffle=True,random_state=42)
cnt=0
for train_index,test_index in kf.split(cleaned_train):
    xtrain,xvalid = train_count_char[train_index],train_count_char[test_index]
    ytrain,yvalid = train_y[train_index],train_y[test_index]
    naive_model,pred_test_y,pred_test_y2 = MUNbaiyes(xtrain,ytrain,xvalid,yvalid,test_count_char)
    pred_full_test += pred_test_y2
    pred_train[test_index,:] = pred_test_y
    
    cv_scores.append(metrics.log_loss(yvalid,pred_test_y))
    print("Cross val score ={}".format(cv_scores[cnt]))
    cnt +=1
    

#add features here
pred_full_test /=5

train['nb_count_ceap'] = pred_train[:,0]
train['nb_count_chpl'] = pred_train[:,1]
train['nb_count_cmws'] = pred_train[:,2]

test['nb_count_ceap'] = pred_full_test[:,0]
test['nb_count_chpl'] = pred_full_test[:,1]
test['nb_count_cmws'] = pred_full_test[:,2]


####  Final Xgboost modelling here

#train = train.drop('nb_count_whpl',axis=1)
cols_to_drop = ['id','text']
train_X = train.drop(cols_to_drop+['author'], axis=1)
test_X = test.drop(cols_to_drop, axis=1)

cv_scores = []
pred_full_test = 0
pred_train = np.zeros([train.shape[0],3])
kf = KFold(n_splits=5,shuffle=True,random_state=42)
cnt=0
for train_index,test_index in kf.split(train_X):
    xtrain,xvalid = train_X.loc[train_index],train_X.loc[test_index]
    ytrain,yvalid = train_y[train_index],train_y[test_index]
    pred_test_y,pred_test_y2,xgmodel = XGBoost(xtrain,ytrain,xvalid,yvalid,test_X,GRID=False)
    pred_full_test += pred_test_y2
    pred_train[test_index,:] = pred_test_y
    
    cv_scores.append(metrics.log_loss(yvalid,pred_test_y))
    print("Cross val score ={}".format(cv_scores[cnt]))
    cnt +=1
    
    
    

