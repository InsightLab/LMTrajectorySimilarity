# Trajectory Embedding Similarity
Pipeline for Trajectory Embedding Similarity using NLP(Natural Language Processing) aproaches. These approaches use the spatial grid from  the t2vec work as a training reference for NLP models, as well as raw trajectory data for calculations involving standard similarity techniques.

The execution of this classical methods (EDR, LCSS and DTW) must be started in the EDR method notebook, as it is in this notebook that the real trajectories, containing points (lon, lat), are filtered for use in such notebook and the other two (LCSS and DTW).

## Used libraries:
Gensim 4.2.0 <br/>
Python 3.8.10 <br/>
Pytorch 1.13.1+cu117 <br/>
Transformers 4.30.2



