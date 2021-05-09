# Semantic Simiilarity Between Two blogs 

### Approach for the Problem :

* Step 1: I have taken simiilar sentences,blogs or any paragraph and  apply Text Processing techniques (stop-words removal , tokenization and lemmetization).
* Step 2:  calculating Similarity index calculation for each word see below the logic for calculating simmilarity index each word.

 
 ```
 for word1 in lemm_sent1:
    simi =[]
    for word2 in lemm_sent2:
        sims = []
        syns1 = wordnet.synsets(word1)
        
        syns2 = wordnet.synsets(word2)
        for sense1, sense2 in itertools.product(syns1, syns2):
            d = wordnet.wup_similarity(sense1, sense2)
            if d != None:
                sims.append(d)
            
            #print(sims)
    
        
        if sims != []:        
           max_sim = max(sims)
           
           simi.append(max_sim)
             
    if simi != []:
        max_final = max(simi)
        final.append(max_final)
 ```
* Step 4 : Final calculate the score by taking the mean of score value 

Note : I have used wordnet Package from NLTK library which provides me to calcluate senetnce simmilarity with the help of two main packages 
 
 `Synset` instances are the groupings of synonymous words that express the same concept
 
 `wup_similarity` Return a score denoting how similar two word senses are, based on the depth of the two senses in the taxonomy and that of their Least Common Subsumer (most specific ancestor node). For More study [Refer](https://www.nltk.org/howto/wordnet.html)
 
 
