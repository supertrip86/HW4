import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import TruncatedSVD as SVD
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy.stats import ttest_ind_from_stats as ttest
from wordcloud import WordCloud as WC
import random


def new_text(text, stop_words):
    
    lemmatizer=WordNetLemmatizer()
    
    words=word_tokenize(text)
    
    
    
    filtered_words=[]
    
    for word in words:
       
        if word.lower() not in stop_words and word.isalpha() and word.lower()!='br': #br is an html tag
            
            filtered_words.append(lemmatizer.lemmatize(word.lower()))
    
    return filtered_words


def take_adj(lista):
    
    items=nltk.pos_tag(lista).   # here it is cretaed a speech tag for all the words in the list
    
    adj_list=[]
    
    for item in items:
        
        if item[1]=="JJ":    ## JJ stays for adjective 
            
            adj_list.append(item[0])
   
    return adj_list


def take_names(lista):
    
    items=nltk.pos_tag(lista).   # here it is cretaed a speech tag for all the words in the list
    
    nn_list=[]
    
    for item in items:
        
        if item[1]=="NN":    ## NN stays for noun
            
            nn_list.append(item[0])
   
    return nn_list


        
    




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



def variance_columns(df_adjectives,number_cluster):     # this function is used to retain the columns with more variance
    
    
    var_dict={}



    for col in df_adjectives:
    
    
        var=np.var(df_adjectives[col])
    
        var_dict[col]=var
    
    
    ordered_dict=dict(sorted(var_dict.items(), key=lambda x:x[1], reverse=True)[0:number_cluster])

    columns_list=pd.Series(list((ordered_dict.keys())))
    
    
    
    return columns_list



#the following function will be used for our k-means implementation

def cluster_labels(word_vector, iterations, cluster_number):

    
    
    centers= random.sample(word_vector,cluster_number).   # here the initial centers are choosen at random.
    
    
    for i in range(0,iterations):         # the algorithm is repeated for n iterations
 
   
    
        label_vector=[]      # a list to store the cluster for each item
    
    
        for vector in word_vector:    
    
        
            distances=[] 
        
            
    
            for center in centers:     # in this cycle for each element is computed the distances with the centers
    
               
                dist=distance.euclidean(center,vector)    
             
                
                distances.append(dist)
            
            
            
            label_vector.append(np.argmin(distances))   # as label is added the minimum distance index
        
        
        print(f'cluster labels for iteration {i} for the first 100 items are {label_vector[0:100]}')   # bonus question: dispaling the iteration process
        
        for k in range(cluster_number):      # in this cycle the centers are updated
    
            
            cluster_sum=np.zeros(len(word_vector[0]))      
    
            cluster_count=0
    
            
            for j in range(len(word_vector)):
        
                
                if k==label_vector[j]:
            
                    
                    cluster_sum=cluster_sum+word_vector[j]
                       
                     
                    cluster_count+=1
            
        
            
            if cluster_count!=0:     # in this case the center value is updated, otherwise the center stays the same
        
                centers[k]=(cluster_sum/cluster_count)
                
    
    
    return label_vector      
    
    
    

def score_distribution(df,number_cluster):
    
    d_mean={}
    
    d_std={}
    
    colors=["green",'yellow','orange','blue','brown','black','purple','red','lightyellow','lightblue','lightgreen','pink','lime','turquoise']

    for i in range(0,number_cluster):


        df_score=df[df['own_cluster']==i]

        df_score=df_score.Score
    
        mean=np.mean(df_score)
        
        std=np.std(df_score)
    
        d_mean[i]=mean
        
        d_std[i]=std
    
        plt.figure(figsize=(16,9))

        plt.title(str(i))
    
        plt.xlim((0,5))
        
        plt.xlabel("score")
        
        plt.ylabel('density')

        plt.suptitle("score density for cluster".upper())

        plt.hist(df_score, color=colors[i], bins=5, density=True)

        plt.show()
        
    return d_mean,d_std




def unique_users(df, number_cluster):

    users_count={}

    for j in range(number_cluster):

        df_unique=df[df['own_cluster']==j]

        single_users=len(pd.unique(df_unique["UserId"]))

        users_count[j]=single_users
    


    series_users=pd.Series(users_count)


    return series_users

#this function provides mean difference significancy between the dataset scores and the scores for each cluster

def mean_difference_main(alpha,means,stds,dataset,df_count):

    test=[]

    alpha=0.05

    significant=[]

    mean_dataset=np.mean(dataset.Score)
    
    std_dataset=np.std(dataset.Score)
    
    
    
    for i in range(14):

    
        t_test=list(ttest(means[i],stds[i],df_count[i],mean_dataset,std_dataset,len(dataset))).  #t-test for the mean difference
        
        test.append(t_test)
        
        if t_test[1]<alpha:
            
            significant.append(True)
            
        else:
            
            significant.append(False)
            
            
        
    df_stat=pd.DataFrame(test, columns=['t','p-value'])  

    df_stat['significant']=significant
    
    return df_stat

# this function provides mean difference significancy among each combination of cluster scores

def mean_difference(alpha,means,stds,df_count):

    test=[]

    alpha=0.05

    significant=[]



    for i in range(14):
    
    
    
        for k in range(i+1,14):

    
            t_test=list(ttest(means[i],stds[i],df_count[i],means[k],stds[k],df_count[k]))
        
            test.append(t_test)
        
            if t_test[1]<alpha:
            
                significant.append(True)
            
            else:
            
                significant.append(False)
            
            
        
    df_stat=pd.DataFrame(test, columns=['t','p-value'])  

    df_stat['significant']=significant

    return df_stat


def word_cloud(image,df1,number_clusters,food_list):

    for i in range(number_clusters):
        
    
        df_words=df1[df1['own_cluster']==i]

        
        df_words=df_words.taken_words

        
        flat_text=[item for sublist in df_words for item in sublist]

        
        text=""

        
        for word in flat_text:
    
            if word.lower() in food_list:
            
                text=text+" "+word
        
       
        wordcloud = WC(mask=image,background_color="white",contour_width=3, contour_color="black").generate(text)

        plt.imshow(wordcloud, interpolation='bilinear')
        
        plt.suptitle("Most numerous words in cluster".upper())
        
        plt.title(str(i))

        plt.axis("off")

        plt.show()
        
    return



def word_cloud_1(image,df1,number_clusters,selected_adjectives):

    for i in range(number_clusters):
        
    
        df_words=df1[df1['own_cluster']==i]

        
        df_words=df_words.adjectives

        
        flat_text=[item for sublist in df_words for item in sublist]

        
        text=""

        
        for word in flat_text:
    
            if word.lower() in selected_adjectives:
            
                text=text+" "+word
        
       
        wordcloud = WC(mask=image,background_color="white",contour_width=3, contour_color="black").generate(text)

        plt.imshow(wordcloud, interpolation='bilinear')
        
        plt.suptitle("Most numerous words in cluster".upper())
        
        plt.title(str(i))

        plt.axis("off")

        plt.show()
        
    return
