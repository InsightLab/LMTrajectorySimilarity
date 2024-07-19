# Trajectory Embedding Similarity
Pipeline for Assessing Spatial Similarity between Trajectories using Embeddings obtained via NLP (Natural Language Processing) approaches. To train these approaches and obtain the embeddings, we used the same spatial grid as the t2vec work as a reference. For the similarity calculation involving classical methods, we use the raw data (lon, lat) of the trajectories.

PLN models, like t2vec, require trajectories in sequential cell format so that they can be trained. Thus, initially, we trained the t2vec model in order to obtain the same spatial grid for training the language models (Word2Vec, DOc2Vec and BERT).

As for the classical methods of spatial similarity between trajectories (EDR, DTW and LCSS), real trajectories were used (paths made up of geospatial points). When executing such methods, we first start with the EDR method notebook, as it contains the necessary code for the initial processing of this GPS data. Once the EDR notebook is executed, the processed trajectories can be used directly by the other two methods (DTW and LCSS).

NOTE: It is important that language models use the same /data directory created by t2vec!

## Used libraries:
Gensim 4.2.0 <br/>
Python 3.8.10 <br/>
Pytorch 1.13.1+cu117 <br/>
Transformers 4.30.2



