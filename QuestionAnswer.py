
import pandas as pd
import numpy as np
import nltk
import re

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import text_to_word_sequence

import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from string import punctuation
from nltk.stem import WordNetLemmatizer
from sklearn.neighbors import NearestNeighbors

from sklearn.metrics.pairwise import cosine_similarity
from numpy import dot
from numpy.linalg import norm


from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

# Deal with NLTK Data uploaded on an external drive
nltk.data.path.append("/Volumes/MGad/MachineLearning/NLTK/nltk_data/")
nltk.data.path


# SET DATA PATHS and load DATAS
#main_path = '/Users/matthieugadel/Desktop/MachineLearning/BV M&O - DIGITAL/'
main_path = '/Users/matthieugadel/Desktop/BV M&O - DIGITAL/'
path = 'data/'
data = pd.read_excel(main_path + path + 'LPO_comments.xlsx')

external_path = "/Volumes/MGad/MachineLearning/Embeddings/glove.840B.300d/"

EMBEDDING_FILE = external_path + r'glove.840B.300d.txt'


############################################
### DATA LOADING AND NULL VALUES
############################################

# On traite uniquementle texte et les LPOs
DATA_text = data[['Comment ID','Comment Text']]

# Drop na values
DATA_text=DATA_text.dropna()


########################################
## PROCESS DATA - SORT BY LANGUAGE
########################################

''' MEthode non parametrique utilisée pour l'apprentissage et la regression.
Calcule la distance euclidienne entre tous les points de l'espace et le nouveau point, eventuellement pondéré par la distance (1/d)
fonction de cout. on peut l'utiliser en regressione et en apprentissage
'''

''' Preparer les données pour le K-NEIREST, ie on calcule le nombre d'occurence d'un texte dans 
le dictionnaire.
C'est ce dictionnaire qu'on representera ensuite dans le K NEIREST NEIGHBOUR
'''



# define a function which turn language into its correct 
def nltk_language(x):
    
    token = nltk.word_tokenize(x)
   
    #df=pd.DataFrame(stopwords.fileids(),columns=['language'])
    #df['Score']=pd.Series(np.zeros((len(stopwords.fileids()))))
    
    dico={}
    
    # We keep the language with the best score
   
    for language in stopwords.fileids(): 
        stopwords_set=set(stopwords.words(language))
        text_set = set(token)
        intersect = text_set.intersection(stopwords_set)
        
        dico[language]= len(intersect)     
                
    maxi=max(zip(dico.values(), dico.keys())) 
    
    return maxi[1]

DATA_text['Language']=DATA_text['Comment Text'].apply(nltk_language)


# export Language in CSV file
out_df = DATA_text
DATA_text.to_csv('comment_with_language.csv', index=False)




##############################################################################
## PREPROCESS DATA - ENGLISH DATAS
##############################################################################

# 0 - Keep only English Data

DATA_eng = DATA_text[DATA_text['Language']=='english']
DATA_eng = DATA_eng.drop('Language',axis=1)

# 1 - CLEANING THE DATA 
'''
In regular sentences Noisy data can be defined as text file header,footer, 
HTML,XML,markup data.As these type of data are not meaningful and does not provide 
any information so it is mandatory to remove these type of noisy data. 
In python HTML,XML can be removed by BeautifulSoup library while markup,
header can be removed by using regular expression
'''

# REMOVE PUNCTUATION and NUMBERS



puncts = [',', '.', '"', ':', ')', '§','(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', 
'·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 
'“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 
'▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 
'∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]
   
#deal with words with adjacent punctuation 
def clean_text(x):
    x = str(x)
    for punct in list(puncts):
        x = x.replace(punct, f' {punct} ')
    return x

DATA_eng["Comment Text Clean"] = DATA_eng["Comment Text"].apply(clean_text)
  

    # Clean Numbers
def clean_numbers(x):
    x = re.sub('[0-9]{5,}', '', x)
    x = re.sub('[0-9]{4,}', '', x)
    x = re.sub('[0-9]{3,}', '', x)
    x = re.sub('[0-9]{2,}', '', x)
    x = re.sub('[0-9]{1,}', '', x)
    return x

DATA_eng["Comment Text Clean"] = DATA_eng["Comment Text Clean"].apply(clean_numbers)

  
    #ensure everithing is in low case
DATA_eng["Comment Text Clean"] = DATA_eng["Comment Text Clean"].apply(lambda x : x.lower())

  
    #remove words smaller than 3 caracters
def clean_small(x):
    x = re.sub(r"\b[a-zA-Z]\b", '', x)
    x = re.sub(r"\b[a-zA-Z]{,2}\b", '', x)
    return x
 
DATA_eng["Comment Text Clean"] = DATA_eng["Comment Text Clean"].apply(clean_small)



# 2 - TOKKENIZATION
''' In tokenization we convert group of sentence into token . It is also called text 
segmentation or lexical analysis. It is basically splitting data into small chunk of words.
 For example- We have sentence — “Ross 128 is earth like planet.Can we survive in that 
 planet?”. After tokenization this sentence will become -[‘Ross’, ‘128’, ‘is’, ‘earth’,
 ‘like’, ‘planet’, ‘.’, ‘Can’, ‘we’, ‘survive’, ‘in’, ‘that’, ‘planet’, ‘?’]. Tokenization 
 in python can be done by python’s NLTK library’s word_tokenize() function

TOKKENIZER
This tokenizer performs the following steps:
- split standard contractions, e.g. don't -> do n't and they'll -> they 'll
- treat most punctuation characters as separate tokens
- split off commas and single quotes, when followed by whitespace
- separate periods that appear at the end of line

'''

DATA_eng['token']= DATA_eng['Comment Text Clean'].apply(nltk.word_tokenize)


# 3 - NORMALISATION // STEMMING - LEMMATIZATION
'''
Before going to normalization first closely observe output of tokenization. 
Will tokenization output can be considered as final output? Can we extract 
more meanigful information from tokenize data ?
In tokenaization we came across various words such as punctuation,stop words
(is,in,that,can etc),upper case words and lower case words.After tokenization 
we are not focused on text level but on word level. So by doing stemming,lemmatization 
we can convert tokenize word to more meaningful words . For example — [‘‘ross’, ‘128’, 
‘earth’, ‘like’, ‘planet’ , ‘survive’, ‘planet’]. As we can see that all the punctuation 
and stop word is removed which makes data more meaningful

STEMMING = 'algorithms work by cutting off the end or the beginning of the word, 
taking into account a list of common prefixes and suffixes that can be found in an 
inflected word. This indiscriminate cutting can be successful in some occasions, but 
not always, and that is why we affirm that this approach presents some limitations.'
LEMMATIZATION = n the other hand, takes into consideration the morphological analysis of the 
words. To do so, it is necessary to have detailed dictionaries which the algorithm can 
look through to link the form back to its lemma. Again, you can see how it works with the
 same example words.
'''

    # REMOVE STOPWORDS AND PUNCTUATION
def remove_stopwords_and_punctuation(x,language='english'):
    stop_words = stopwords.words('english') + list(punctuation)
    x1 = [word for word in x if word not in stop_words]
    return x1

DATA_eng['token'] = DATA_eng['token'].apply(remove_stopwords_and_punctuation)


    # LEMMISATION

#is based on The Porter Stemming Algorithm
wordnet_lemmatizer = WordNetLemmatizer()

def lemming(x,language='english'):
    x1 = [wordnet_lemmatizer.lemmatize(word) for word in x]    
    return x1

DATA_eng['token'] = DATA_eng['token'].apply(lemming)

DATA_eng['token'].iloc[12]

#Controll text processing for a random sentence
#a=DATA_eng['token'].iloc[10]
#b=a.apply(lemming)
# d=set(a).difference(set(b))

# 4 - BAG OF WORDS METHOD / VECTORIZATION
''' It is basic model used in natural language processing. 
Why it is called bag of words because any order of the words in the document 
is discarded it only tells us weather word is present in the document or not 

So how a word can be converted to vector can be understood by simple word count 
example where we count occurrence of word in a document 

The approach which is discussed above is unigram because we are considering only 
one word at a time . Similarly we have bigram(using two words at a time- for example
 — There used, Used to, to be, be Stone, Stone age), trigram(using three words at a time
 - for example- there used to, used to be ,to be Stone,be Stone Age), ngram(using n words
 at a time)
Hence the process of converting text into vector is called vectorization.
By using CountVectorizer function we can convert text document to matrix of word count.
 Matrix which is produced here is sparse matrix. By using CountVectorizer on above document
 we get 5*15 sparse matrix of type numpy.int64.
 
TF-IDF stands for Term Frequency-Inverse Document Frequency which basically tells 
importance of the word in the corpus or dataset. TF-IDF contain two concept 
Term Frequency(TF) and Inverse Document Frequency(IDF)

Term Frequency
Term Frequency is defined as how frequently the word appear in the document or corpus. 
As each sentence is not the same length so it may be possible a word appears in long 
sentence occur more time as compared to word appear in sorter sentence. Term frequency 
can be defined as:

Inverse Document Frequency
Inverse Document frequency is another concept which is used for finding out importance 
of the word. It is based on the fact that less frequent words are more informative and 
important. IDF is represented by formula:

TF-IDF
TF-IDF is basically a multiplication between Table 2 (TF table) and Table 3(IDF table) .
 It basically reduces values of common word that are used in different document. 
 As we can see that in Table 4 most important word after multiplication of TF and IDF is 
 ‘TFIDF’ while most frequent word such as ‘The’ and ‘is’ are not that important
'''


##############################################################################
## TF IDF REPRESENTATION - VECTORIZE INFORMATION - SKLEARN
##############################################################################



# 1 - We define a VECTORIZER from skiilearn



# Minimum occurence to be in the dico = 10
#mn_df=5
#mx_df=3000

# define dummy to handle alreaty tokkenize data in the vectorizer
def dummy(tokens):
    return tokens

vectorizer = TfidfVectorizer(analyzer='word',stop_words='english',tokenizer=dummy,preprocessor=dummy)

data_vectorized = vectorizer.fit_transform(DATA_eng['token'])



# 2 - CHECK THE SPARCITY (number of ZEROS) of the MATRIX

# Materialize the sparse data
data_dense = data_vectorized.todense()

# Compute Sparsicity = Percentage of Non-Zero cells
print("Sparsicity: ", ((data_dense > 0).sum()/data_dense.size)*100, "%")






########################################
## K-NEIREST NEIGHBOURS ON DATA
########################################

'''
On transforme la QUERY grace à l'algo ci dessus
on cherche ses k voisins les plus proches
'''

# BUILD K-NEIREST NEIGHBOUR ALGORITHM
'''
on construit le classifier
'''

N_Voisins = 10


neighbors = NearestNeighbors(n_neighbors=N_Voisins)
neighbors.fit(data_vectorized) 

# INPUT QUERY AND TRANSFORM
'''
on transforme la question grace a TF_IDF
'''
QUERY='fire extinguisher hazardous'
Q=nltk.word_tokenize(QUERY)
Y=vectorizer.transform([Q])


# FIND THE K-NEIGHREST NEIGHBOUR
'''
on retourne les k neighbours de la query
'''
neighbors_index = neighbors.kneighbors(Y,return_distance=False)

'''
A partir de leur indice on ressort le texte originel
'''
a=neighbors_index.tolist()
ANSWER = DATA_eng['Comment Text'].iloc[a[0]]







########################################
## COSINE SIMILARITY TO CHECK SIMILAR DOCUMENTS
########################################
'''
The tfidf represents your documents in a common vector space. If you then calculate
 the cosine similarity between these vectors, the cosine similarity compensates for 
 the effect of different documents' length. The reason is that the cosine similarity
 evaluates the orientation of the vectors and not their magnitude. I can show you 
 the point with python: Consider the following (dumb) documents
'''



def cosine_sim(a,b):
    cos_sim = dot(a, b)/(norm(a)*norm(b))
    return cos_sim 


# take second element for sort
def takeSecond(elem):
    return elem[1]


def cosine_sim_texts(vector,tfidf_matrix):
    # return n closer texts cosine sim with corpus
       
    liste = []
    
    
    # transform vector to a 1D numpy array
    vect = np.asarray(vector.todense()).ravel()
    
    for i in range(0,tfidf_matrix.shape[0]):
        
        vectIFIDF_i=np.asarray(tfidf_matrix[i].todense()).ravel()       
        score=cosine_sim(vectIFIDF_i, vect)
        liste.append((i,score) )      
        liste.sort(key = takeSecond,reverse=True)
               
    return liste


dico_a = cosine_sim_texts(Y,data_vectorized)

def N_first_cosim(dico,N,DATA):
    #show N first 
    N_first=pd.DataFrame(columns=DATA.columns)

    for i in range (0,N):
        print(N_first)
        key=dico_a[i][0]
        N_first=N_first.append(DATA.iloc[key], ignore_index = True)
        print(DATA.iloc[key])

    return N_first


cos_liste=N_first_cosim(dico_a,10,DATA_eng).drop(['Comment Text Clean','token'], axis=1)
    





##############################################################################
## PRE TRAINED METHOD + SMOOTH INVERSE FREQUENCY + COSINE SIMILART
##############################################################################


print('Define and Load Embedding Matrix')

# ATTENTION ne pas LEMMISER

DATA_eng['token_emb']= DATA_eng['Comment Text Clean'].apply(nltk.word_tokenize)

def remove_punctuation(x,language='english'):
    stop_words = list(punctuation)
    x1 = [word for word in x if word not in stop_words]
    return x1

DATA_eng['token_emb'] = DATA_eng['token_emb'].apply(remove_punctuation)

vectorizer_emb = CountVectorizer(analyzer='word',stop_words=None,tokenizer=dummy,preprocessor=dummy)
data_vectorized = vectorizer_emb.fit_transform(DATA_eng['token_emb'])

vectorizer_emb.get_feature_names()


# Load GloVes Function

#define a max of value to work from the matrix embedding
max_features_embedding =10000
embed_size=300

def Load_GloVes(glove_file,word_index):
    '''
    On construit la MATRIX des embeddings de l'algo GloVes. 
    La dimension est (nb mots vocabulaire,tailles des embeddings)
    '''
    with open(glove_file,'r',encoding='utf8') as f:
        words= set()
        Glove_Index={}
        
        for lines in f:
            tmpLine = lines.split(" ")
            curr_word=tmpLine[0]
            words.add(curr_word)
            Glove_Index[curr_word] = np.array(tmpLine[1:], dtype='float32')
    
    #on prend le mean et la std
    #all_embs = np.stack(embeddings_index.values())
    words=list(Glove_Index.values())
    emb_mean,emb_std = np.array(words).mean(), np.array(words).std()
    
    #on reduit la taille de la matrix embedding pour diminuer le temps de calcu
    nb_words_emb=min(max_features_embedding,len(word_index))
    
    #On initialize la matrice avec la moyenne de tout // we add +1 to fit keras requirements
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words_emb, embed_size))
      
    # continue = on garde l'itération tant que la condition est respectée
    i=0
    dict_emb={}
    
    for word in word_index:
        if i >= max_features_embedding: continue
        embedding_vector = Glove_Index.get(word)
        # on traite le cas ou le mot ne soit pas dans le dico
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        dict_emb[word]= embedding_vector
            
    return embedding_matrix,dict_emb,Glove_Index


# Load Dict and Embeddings

embedding_matrix,dict_emb,Glove_Index =Load_GloVes(EMBEDDING_FILE,vectorizer_emb.get_feature_names())     



'''
Smooth Inverse Frequency
Taking the average of the word embeddings in a sentence (as we did just above) tends to give 
too much weight to words that are quite irrelevant, semantically speaking. 
Smooth Inverse Frequency tries to solve this problem in two ways:
Weighting: SIF takes the weighted average of the word embeddings in the sentence. 
Every word embedding is weighted by a/(a + p(w)), where a is a parameter that is typically 
set to 0.001 and p(w) is the estimated frequency of the word in a reference corpus.
Common component removal: SIF computes the principal component of the resulting embeddings 
for a set of sentences. It then subtracts from these sentence embeddings their projections 
on their first principal component. This should remove variation related to frequency and 
syntax that is less relevant semantically
'''

def Smooth_Inv_Frequency():
    
    return



def Text_Emb(token_vect,dict_emb,Glove_Index):
    '''
    Return vector. Sum of embeddings of each words of a given text (vect = tokkenize text)
    '''
    length = len(dict_emb['true'])
    text_representation=np.zeros((length,))
    
    for word in token_vect:
        if word in Glove_Index:
            text_representation += dict_emb[word]

    
    return text_representation


def Corpus_Emb(corpus,dict_emb,Glove_Index):
    '''
    Return vector. Sum of embeddings of each words of a given text (vect tokkenize text)
    '''
    length_emb = len(dict_emb['true'])
    length_corpus = corpus.shape[0]
    
    corpus_emb = np.zeros((length_corpus,length_emb))
    
    for i in range(0,length_corpus):
        corpus_emb[i,:]=Text_Emb(corpus.iloc[i],dict_emb,Glove_Index)
        
    
    return corpus_emb


corpus_sum_emb = Corpus_Emb(DATA_eng['token_emb'],dict_emb,Glove_Index)

Vect_emb = Text_Emb(Q,dict_emb,Glove_Index)


def cosine_sim_texts(vector,corpus_sum_emb):
    '''
    Cosine similarity between vector and a corpus. Vector tokkenize vectoe. dict emb dictionnary
    with each embedding
    '''
        
    liste = []

    # transform vector to a 1D numpy array
    vect = np.asarray(vector).ravel()
    
    for i in range(0,corpus_sum_emb.shape[0]):
        
        vectEMB_i=np.asarray(corpus_sum_emb[i]).ravel()       
        score=cosine_sim(vectEMB_i, vect)
        liste.append((i,score) )      
        liste.sort(key = takeSecond,reverse=True)
               
    return liste

liste_emb = cosine_sim_texts(Vect_emb,corpus_sum_emb)


def N_first_cosim(liste_emb,N,DATA_texte):
    #show N first 
    N_first=pd.DataFrame(columns=DATA_texte.columns)

    for i in range (0,N):
        print(N_first)
        key=liste_emb[i][0]
        N_first=N_first.append(DATA_texte.iloc[key], ignore_index = True)
        print(DATA_texte.iloc[key])

    return N_first

N_first = N_first_cosim(liste_emb,10,DATA_eng)




# INPUT QUERY AND TRANSFORM
'''
on transforme la question grace a TF_IDF
'''
QUERY='escape areas fore peak'
Q=nltk.word_tokenize(QUERY)

Vect_emb = Text_Emb(Q,dict_emb,Glove_Index)
liste_emb = cosine_sim_texts(Vect_emb,corpus_sum_emb)

N_first = N_first_cosim(liste_emb,10,DATA_eng)


########################################
## CLUSTERING TEXT DOCUMENTS USING KMEANS - SKII
########################################
'''
Two feature extraction methods can be used in this example:
TfidfVectorizer uses a in-memory vocabulary (a python dict) to map the most frequent words 
to features indices and hence compute a word occurrence frequency (sparse) matrix. 
The word frequencies are then reweighted using the Inverse Document Frequency (IDF) 
vector collected feature-wise over the corpus.
Two algorithms are demoed: ordinary k-means and its more scalable cousin minibatch k-means.
Additionally, latent semantic analysis can also be used to reduce dimensionality 
and discover latent patterns in the data.
LSA = ID-IDF + reduction de dimension de la matrice afin d'ameliorer tps de calcul, de
reduire le bruit (mot apparaissant peu)
'''


# 1 - Get the TF - IDF document
 
data_vect_cluster = vectorizer.fit_transform(DATA_eng['token'])


# DATA have already been proceset through TDIFFVectorizer
'''
 LSA TREAT ie dimensionnality reduction
 Vectorizer results are normalized, which makes KMeans behave as
 spherical k-means for better results. Since LSA/SVD results are
 not normalized, we have to redo the normalization.
 Therefore it's is identical as a cossin similarity for vectors

 We define a pipeline to perform at the SAME rime normalization and SVD

 In particular, truncated SVD works on term count/tf-idf matrices as returned by the 
 vectorizers in sklearn.feature_extraction.text. In that context, it is known as latent 
 semantic analysis (LSA).
'''


# Define number of parameters - after it will be hyperamater through pipeline
N_COMPONENTS = 5000

X =  data_vect_cluster
   
svd = TruncatedSVD(N_COMPONENTS)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)

X = lsa.fit_transform(X)


# check if X is normalized
print(norm(X[50,:]))


# ELBOW METHOD to define OPTIMAL K
'''
on fait varier K entre plusieurs valeurs pour lesquelles on compute le score
'''

def Elbow_KMeans(X,parameters):
    
    k_list=[]
    
    for K in parameters :
        print(K)
        kmeans = MiniBatchKMeans(n_clusters=int(K))
        kmeans.fit(X)

        k_list.append(kmeans.inertia_)

    return k_list

parameters = [10,40,70,90,100,120,140,160,200,300]


k_list=Elbow_KMeans(X,parameters)

import matplotlib.pyplot as plt


def Print_Elbow(parameters,k_list):
    
    plt.plot(parameters,k_list)
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    plt.show()
    
    return

Print_Elbow(parameters,k_list)


# Choice of a 90 parameter for K MEan

N_CLUSTERS = 90

kmeans = MiniBatchKMeans(n_clusters=N_CLUSTERS, init='k-means++', n_init=1,batch_size=100, verbose=True)
kmeans.fit(X)
kmeans.score(X)

from sklearn import metrics

print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, kmeans.labels_, sample_size=1000))


cluster = kmeans.predict(X)
DATA_Clustered=pd.Series(cluster,name='Cluster')
DATA_Clustered.value_counts()
DATA_Clustered=pd.DataFrame(df)
DATA_Clustered['Comment Text']=DATA_eng['Comment Text']
DATA_Clustered['Comment ID']=DATA_eng['Comment ID']


# check cluster, for example for LPO1405944 of above example
DATA_Clustered[DATA_Clustered['Comment ID']=='LPO1405944']
DATA_Clustered[DATA_Clustered['Cluster']==48]





'''
Hashing-based approaches

An alternative to the tree-based approach is the hash-based approach. Unlike trees, in hashes there's no recursive partitioning. The idea is to learn a model that converts an item into a code, where similar items will produce the same or similar code (hashing collision). This approach significantly reduces the memory that's needed. The expected query time is O(1), but can be sublinear in n, where n is the number of items (vectors) you have. Examples of hashing-based approaches include the following:

Locality-sensitive hashing (LSH)
PCA hashing
Min-hashing
Spectral hashing

There are several open source libraries that implement approximate similarity matching techniques, with different trade-offs between precision, query latency, memory efficiency, time to build the index, features, and ease of use.
The example solution described in this article uses Annoy (Approximate Nearest Neighbors Oh Yeah), a library built by Spotify for music recommendations. Annoy is a C++ library with Python bindings that builds random projection trees. An index is built with a forest of k trees, where k is a tunable parameter that trades off between precision and performance. It also creates large read-only, file-based data structures that are mapped into memory so that many processes can share the data.
Other widely used libraries are NMSLIB (non-metric space library) and Faiss (Facebook AI Similarity Search). The library you use to implement approximate similarity matching shouldn't affect the overall solution architecture or the workflow discussed in this article.


'
