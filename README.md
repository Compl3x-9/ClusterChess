# ClusterChess
ClusterChess is my bachelor's thesis in Computer Science at the
University of Granada.

ClusterChess is a classification algorithm for chess errors,
which can be used to study common mistakes in future opponents
or in one's play, or to get training positions similar to
others.

It consists in three phases:
1. It extracts errors from chess games with the guidance
   of the [Stockfish chess engine](https://stockfishchess.org/), using concurrency and
   asyncronous input/output to make good use of our hardware
   resources.
2. Then it uses a novel method to extract a feature vector from
   each error using inter-layer values from the [DeepChess'](https://arxiv.org/abs/1711.09667)
   siamese neural network (we use [Bot-Benedict's implementation](https://github.com/Bot-Benedict/DeepChess)
   of the paper). In the thesis, we discuss four different ways
   of getting these features, which we implement by creating
   derived neural networks from the original.
3. Finally, we preprocess the data and classify it using
   a K-means model.

The original thesis is included in this repository, together with
the presentation (it's all in Spanish).

# Repository Structure

### main.py and src
The code source. From the main file you can replicate all
processes discussed on the thesis. Most of the chart's code isn't included,
as I feel that it's too messy to make public.

## requirements.txt
A list of the necessary python libraries to run the code, together with the
versions I used. They can be easily installed with `pip install -r requirements.txt`.
In the file `all_packages.txt`, you can find all the modules I had installed,
just in case I missed any dependency.

To extract errors from chess games, you must have Stockfish installed and
in your `PATH` variable.

### data
Here's where the data and the models are saved.
- `errors`: CSV files with chess mistakes. `final_errors.csv` contains the
  500.000 errors which served as the database for the project.
- `lichess_elite`: The chess game database from which the errors were extracted.
  You can download a similar one [here](https://database.nikonoel.fr/).
- `nn_networks`: Where all the neural networks are saved.
- `kmeans_models`: The four k-means models.
- `nn_vectors`: Where the feature vectors are saved.
- `cluster_labels`: It contains the four final classifications depending on the
  feature extraction process.
- `steinitz`: It contains all the files from the analysis of Steinitz's play.

### documents
It contains the original thesis and the presentation
