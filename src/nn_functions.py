# from keras.models import Model, load_model
import chess
import numpy as np
import tensorflow as tf
# import sys
# import pandas as pd
# import chess.pgn
from scipy.sparse import csr_array, lil_array, save_npz

from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Input, Concatenate

# https://github.com/Bot-Benedict/DeepChess
#

# Capa creada para multiplicar el output por los valores de una matriz
# elemento a elemento, por columnas y aumentando el tamaño
class MatrixMultLayer(tf.keras.layers.Layer):
    def __init__(self, matrix):
        super(MatrixMultLayer, self).__init__()
        self.matrix = tf.Variable(initial_value=matrix,
                               trainable=False)

    def call(self, inputs):
        # print(self.matrix)
        # print(self.matrix.shape)
        # print(inputs.shape)
        # ret = tf.empty((inputs.shape[2]*self.matrix.shape[1]))
        left = tf.math.multiply(self.matrix[:,0], inputs)
        right = tf.math.multiply(self.matrix[:,1], inputs)

        # print(left)

        return tf.concat([left, right], axis=2)

class MatrixMultLayer2(tf.keras.layers.Layer):
    def __init__(self, matrix):
        super(MatrixMultLayer2, self).__init__()
        self.matrix = tf.Variable(initial_value=matrix,
                               trainable=False)

    def call(self, inputs):
        # print(self.matrix)
        # print(self.matrix.shape)
        # print(inputs.shape)
        # ret = tf.empty((inputs.shape[2]*self.matrix.shape[1]))

        # print(left)

        return tf.math.multiply(self.matrix, inputs)

def make_mov2vec_model(default_model):

    # Weights of the layers
    default_weights = [layer.get_weights() for layer in default_model.layers]

    # Shell of the new model
    dbn_model = Sequential([
        Dense(773,activation='relu'),
        Dense(600,activation='relu'),
        Dense(400,activation='relu'),
        Dense(200,activation='relu'),
        Dense(100,activation='relu')
    ])
    left_input = Input((None, 773))
    right_input = Input((None, 773))
    encoded_l = dbn_model(left_input)
    encoded_r = dbn_model(right_input)

    concatenated = Concatenate()([encoded_l, encoded_r])
    deepchess = Dense(400, activation='relu')(concatenated)
    deepchess = Dense(200, activation='relu')(deepchess)
    deepchess = Dense(100, activation='relu')(deepchess)
    # deepchess_output = Dense(2, activation='softmax')(deepchess)
    deepchess = MatrixMultLayer(default_weights[-1][0])(deepchess)

    # Compiling the shell
    deconstr_model = Model(inputs=[left_input, right_input], outputs=deepchess)
    deconstr_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss=tf.keras.losses.binary_crossentropy, metrics=['acc']) #, run_eagerly=True)

    # Filling the weights
    for i in range(len(default_weights)-1):
        deconstr_model.layers[i].set_weights(default_weights[i])

    # print(default_weights[-1])

    return deconstr_model

def make_diff2vec_model(default_model):

    # Weights of the layers
    default_weights = [layer.get_weights() for layer in default_model.layers]

    # Shell of the new model
    dbn_model = Sequential([
        Dense(773,activation='relu'),
        Dense(600,activation='relu'),
        Dense(400,activation='relu'),
        Dense(200,activation='relu'),
        Dense(100,activation='relu')
    ])
    left_input = Input((None, 773))
    right_input = Input((None, 773))
    encoded_l = dbn_model(left_input)
    encoded_r = dbn_model(right_input)

    concatenated = Concatenate()([encoded_l, encoded_r])
    deepchess = Dense(400, activation='relu')(concatenated)
    deepchess = Dense(200, activation='relu')(deepchess)
    deepchess = Dense(100, activation='relu')(deepchess)
    # deepchess_output = Dense(2, activation='softmax')(deepchess)
    last_weights = default_weights[-1][0][:,0] - default_weights[-1][0][:,1]
    print(last_weights.shape)
    deepchess = MatrixMultLayer2(last_weights)(deepchess)

    # Compiling the shell
    deconstr_model = Model(inputs=[left_input, right_input], outputs=deepchess)
    deconstr_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss=tf.keras.losses.binary_crossentropy, metrics=['acc']) #, run_eagerly=True)

    # Filling the weights
    for i in range(len(default_weights)-1):
        deconstr_model.layers[i].set_weights(default_weights[i])

    # print(default_weights[-1])

    return deconstr_model

def make_pos2vec2_model(default_model):

    # Weights of the layers
    default_weights = [layer.get_weights() for layer in default_model.layers]

    # Shell of the new model
    dbn_model = Sequential([
        Dense(773,activation='relu'),
        Dense(600,activation='relu'),
        Dense(400,activation='relu'),
        Dense(200,activation='relu'),
        Dense(100,activation='relu')
    ])
    input = Input((None, 773))
    out = dbn_model(input)

    # Compiling the shell
    model = Model(inputs=input, outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss=tf.keras.losses.binary_crossentropy, metrics=['acc']) #, run_eagerly=True)

    # print(default_weights[2])
    # print(len(default_weights[2]))
    # print(len(model.layers))

    # Filling the weights
    model.layers[0].set_weights(default_weights[1])
    model.layers[1].set_weights(default_weights[2])

    return model


# Función que transforma un FEN en el bitboard apto para la red neuronal de DeepChess
def fen_to_bitboard(fen):
    chess_pieces = {
        'p': 0,
        'n': 1,
        'b': 2,
        'r': 3,
        'q': 4,
        'k': 5,
        'P': 6,
        'N': 7,
        'B': 8,
        'R': 9,
        'Q': 10,
        'K': 11
    }

    bitboard = [0]*773
    currIndex = 0
    [position, turn, castling, _, _, _] = fen.split(' ')
    for ch in position:
        if ch == '/':
            continue
        elif ch >= '1' and ch <= '8':
            currIndex += (ord(ch) - ord('0')) * 12
        else:
            bitboard[currIndex + chess_pieces[ch]] = 1
            currIndex += 12
    bitboard[768] = 1 if turn == 'w' else 0
    bitboard[769] = 1 if 'K' in castling else 0
    bitboard[770] = 1 if 'Q' in castling else 0
    bitboard[771] = 1 if 'k' in castling else 0
    bitboard[772] = 1 if 'q' in castling else 0
    return bitboard

# Devuelve el output de deepchess con dos posiciones.
# Suponemos que el modelo dado es Deepchess
def one_deepchess(model, fen1, fen2):
    l = []
    r = []

    l.append(fen_to_bitboard( fen1 ))
    r.append(fen_to_bitboard( fen2 ))

    l = np.array(l)
    l = l[:, np.newaxis, :]

    r = np.array(r)
    r = r[:, np.newaxis, :]

    out = model.predict(( l, r ))
    return out

# Devuelve el output de un modelo de una posición y movimiento
def one_mov2vec(model, fen, move=None):

    if move is None:
        i = [fen_to_bitboard( fen )]
        i = np.array(i)
        i = i[:, np.newaxis, :]
        out = model.predict(i)

    else:
        l = []
        r = []

        l.append(fen_to_bitboard( fen ))
        board = chess.Board(fen)
        board.push_san(move)
        r.append(fen_to_bitboard( board.fen() ))

        l = np.array(l)
        l = l[:, np.newaxis, :]
        r = np.array(r)
        r = r[:, np.newaxis, :]

        out = model.predict(( l, r ))

    return out

one_analisis = one_mov2vec

# Devuelve el output de un modelo al introducirle posiciones y movimientos en parejas.
# Mucho más eficiente que llamar muchas veces a la función de antes
def lots_mov2vec(model,  FENs, moves, out_len=200):
    i=0
    l=[]
    r=[]
    interval = 50000
    ret = lil_array((len(FENs), out_len))

    for fen, move in zip( FENs, moves ):
        l.append(fen_to_bitboard( fen ))
        board = chess.Board(fen)
        board.push_san(move)
        r.append(fen_to_bitboard( board.fen() ))
        i += 1
        if i%interval == 0:
            l = np.array(l)
            l = l[:, np.newaxis, :]

            r = np.array(r)
            r = r[:, np.newaxis, :]
            out = model.predict(( l, r ))
            ret[i-interval:i,:] = out[:,0,:]
            print(i)
            r = []
            l = []

    l = np.array(l)
    l = l[:, np.newaxis, :]

    r = np.array(r)
    r = r[:, np.newaxis, :]

    out = model.predict(( l, r ))
    ret[i-(i%interval):,:] = out[:,0,:]

    # save_npz( "./" + sys.argv[1].split('/')[-1].split('.')[0] + ".npz", output.tocsr() )

    return ret.tocsr()

def lots_pos2vec(model,  FENs, out_len=100):
    i=0
    inp = []
    interval = 50000
    ret = lil_array((len(FENs), out_len))

    for fen in FENs:
        inp.append(fen_to_bitboard( fen ))
        i += 1
        if i%interval == 0:
            inp = np.array(inp)
            inp = inp[:, np.newaxis, :]
            out = model.predict(inp)
            ret[i-interval:i,:] = out[:,0,:]
            print(i)
            inp = []

    inp = np.array(inp)
    inp = inp[:, np.newaxis, :]
    out = model.predict(inp)
    ret[i-(i%interval):,:] = out[:,0,:]
    print(i)

    # save_npz( "./" + sys.argv[1].split('/')[-1].split('.')[0] + ".npz", output.tocsr() )

    return ret.tocsr()

def lots_analisis(model, FENs, moves=None, out_len=100):
    if moves is None:
        return lots_pos2vec(model, FENs, out_len)
    else:
        return lots_mov2vec(model, FENs,moves, out_len)

nn_original_deepchess = "data/nn_networks/deepchess"
def load_model_original_deepchess():
    return tf.keras.models.load_model(nn_original_deepchess)

nn_mov2vec = "data/nn_networks/mov2vec"
nn_dif2vec = "data/nn_networks/dif2vec"
nn_pos2vec1 = "data/nn_networks/fpos2vec_v1/fpos2vec_v1"
nn_pos2vec2 = "data/nn_networks/pos2vec2"

def load_model_pos2vec1():
    dbn_model = Sequential([
        Dense(773,activation='relu'),
        Dense(600,activation='relu'),
        Dense(400,activation='relu'),
        Dense(200,activation='relu'),
        Dense(100,activation='relu')
    ])
    dbn_model.load_weights(nn_pos2vec1)
    # input = Input((None, 773))
    # out = dbn_model(input)
    #
    # Compiling the shell
    # model = Model(inputs=dbn_model, outputs=dbn_model)
    # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss=tf.keras.losses.binary_crossentropy, metrics=['acc']) #, run_eagerly=True)
    return dbn_model

def load_model_mov2vec():
    return tf.keras.models.load_model(nn_mov2vec)

def load_model_dif2vec():
    return tf.keras.models.load_model(nn_dif2vec)

def load_model_pos2vec2():
    return tf.keras.models.load_model(nn_pos2vec2)
def load_model_deepchess():
    return tf.keras.models.load_model(nn_original_deepchess)

if __name__ == "__main__":

    default_model = tf.keras.models.load_model(nn_original_deepchess)

    if False:
        mov2vec = make_mov2vec_model(default_model)
        dif2vec = make_diff2vec_model(default_model)
        pos2vec1 = load_model_pos2vec1()
        pos2vec2 = make_pos2vec2_model(default_model)

    else:
        mov2vec = load_model_mov2vec()
        dif2vec = load_model_dif2vec()
        pos2vec1 = load_model_pos2vec1()
        pos2vec2 = load_model_pos2vec2()

    if False:
        mov2vec.save(nn_mov2vec)
        dif2vec.save(nn_dif2vec)
        pos2vec2.save(nn_pos2vec2)

    if True:
        fen = "r4rk1/p4p2/4pq1p/5p2/1p3PP1/1Pn1R2Q/P6P/1B3R1K b - - 0 28"
        move = "f5g4"
        out = one_analisis(mov2vec, fen, move)
        print(out)
        print(out.shape)
        out = one_analisis(dif2vec, fen, move)
        print(out)
        print(out.shape)
        out = one_analisis(pos2vec1, fen)
        print(out)
        print(out.shape)
        out = one_analisis(pos2vec2, fen)
        print(out)
        print(out.shape)
