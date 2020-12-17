import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import TruncatedSVD as SVD
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance
from wordcloud import WordCloud as WC
import random


def new_text(text, stop_words):
    
    lemmatizer=WordNetLemmatizer()
    
    words=word_tokenize(text)
    
    filtered_words=[]
    
    for word in words:
       
        if word.lower() not in stop_words and word.isalpha():
            
            filtered_words.append(lemmatizer.lemmatize(word.lower()))
    
    return filtered_words


def take_adj(lista):
    
    items=nltk.pos_tag(lista)
    
    adj_list=[]
    
    for item in items:
        
        if item[1]=="JJ":
            
            adj_list.append(item[0])
   
    return adj_list
        
    




def df_words_dict(list_agg,all_adjectives):
    
    d_adj={}
   
    for i in range(len(list_agg)):
    
        support_list=[]
        
        for element in all_adjectives:
            
            if element in list_agg[i]:
                
                support_list.append(1)
                
            else:
                
                support_list.append(0)
                
        
        d_adj[i]=support_list
                
        
    return d_adj


def variance_columns(df_adjectives,number_cluster):
    
    var_dict={}



    for col in df_adjectives:
    
    
        var=np.var(df_adjectives[col])
    
        var_dict[col]=var
    
    
    ordered_dict=dict(sorted(var_dict.items(), key=lambda x:x[1], reverse=True)[0:number_cluster])

    columns_list=pd.Series(list((ordered_dict.keys())))
    
    
    
    return columns_list





def cluster_labels(word_vector, iterations, cluster_number):

    
    
    centers= random.sample(word_vector,cluster_number)
    
    
    for i in range(0,iterations):
 
   
    
        label_vector=[]
    
    
        for vector in word_vector:
    
        
            distances=[]
        
        
    
            for center in centers:
    
               
                dist=distance.euclidean(center,vector)
             
                
                distances.append(dist)
            
            
            
            label_vector.append(np.argmin(distances))
        
        
        
        
        for k in range(cluster_number):
    
            
            cluster_sum=np.zeros(len(word_vector[0]))
    
            cluster_count=0
    
            
            for j in range(len(word_vector)):
        
                
                if k==label_vector[j]:
            
                    
                    cluster_sum=cluster_sum+word_vector[j]
                       
                     
                    cluster_count+=1
            
        
            
            if cluster_count!=0:
        
                centers[k]=(cluster_sum/cluster_count)
                
    
    
    return label_vector      
    
    
    

def score_distribution(df,number_cluster):
    
    d={}
    
    colors=["green",'yellow','orange','blue','brown','black','purple','red','lightyellow','lightblue','lightgreen','pink','lime','turquoise']

    for i in range(0,number_cluster):


        df_score=df[df['own_cluster']==i]

        df_score=df_score.Score
    
        mean=np.mean(df_score)
    
        d[i]=mean
    
        plt.figure(figsize=(16,9))

        plt.title(str(i))
    
        plt.xlim((0,5))
        
        plt.xlabel("score")
        
        plt.ylabel('density')

        plt.suptitle("score density for cluster".upper())

        plt.hist(df_score, color=colors[i], bins=5, density=True)

        plt.show()
        
    return d




def unique_users(df, number_cluster):

    users_count={}

    for j in range(number_cluster):

        df_unique=df[df['own_cluster']==j]

        single_users=len(pd.unique(df_unique["UserId"]))

        users_count[j]=single_users
    


    series_users=pd.Series(users_count)


    return series_users


def word_cloud(image,df1,number_clusters):

    for i in range(number_clusters):
        
    
        df_words=df1[df1['cluster']==i]

        
        df_words=df_words.taken_words

        
        flat_text=[item for sublist in df_words for item in sublist]

        
        text=""

        
        for word in flat_text:
    
            
            text=text+" "+word
        
       
        wordcloud = WC(mask=image,background_color="white",contour_width=3, contour_color="black").generate(text)

        plt.imshow(wordcloud, interpolation='bilinear')
        
        plt.suptitle("Most numerous words in cluster".upper())
        
        plt.title(str(i))

        plt.axis("off")

        plt.show()
        
    return