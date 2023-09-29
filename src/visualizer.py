import pandas as pd
import numpy as np


import chess
import chess.svg

import sys, os
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtQml import *
from PyQt5.QtQuick import *

class Backend(QObject):

    maxClusterChanged = pyqtSignal()
    maxPositionChanged = pyqtSignal()

    def __init__(self, db, labels):
        QObject.__init__(self)
        self.m_cluster = 0
        self.m_position = 0

        self.m_df = db.reset_index()
        self.m_df['labels'] = pd.Series(labels)
        # self.m_df.labels[self.m_df.labels == np.nan] = 0
        # self.m_labels = labels
        self.m_maxCluster = np.max(labels)

        self.db_cluster = self.m_df[self.m_df.labels == self.m_cluster].reset_index()
        self.m_maxPosition = self.db_cluster.shape[0]

    def svg_of_empty_board(self):
        return chess.svg.board(chess.Board("8/8/8/8/8/8/8/8"))

    def svg_of_board_and_move( self, board_fen, move ):
        board = chess.Board(board_fen)
        move = chess.Move.from_uci(move)

        return chess.svg.board(
            board,
            orientation=board.turn,
            lastmove=move,
            arrows=[chess.svg.Arrow(move.from_square, move.to_square)]
        )

    @pyqtProperty(int)
    def cluster(self):
        return self.m_cluster

    @pyqtProperty(int, notify=maxClusterChanged)
    def maxCluster(self):
        return self.m_maxCluster

    @pyqtProperty(int)
    def position(self):
        return self.m_position

    @pyqtProperty(int, notify=maxPositionChanged)
    def maxPosition(self):
        return self.m_maxPosition

    @pyqtSlot(int)
    def setCluster(self, cluster):
        if self.m_cluster == cluster:
            return
        self.m_cluster = cluster
        self.db_cluster = self.m_df[self.m_df.labels == self.m_cluster].reset_index()
        self.m_maxPosition = self.db_cluster.shape[0]
        self.maxPositionChanged.emit()

    @pyqtSlot(int)
    def setPosition(self, position):
        if self.m_position == position:
            return
        self.m_position = position

    @pyqtSlot(result=str)
    def refreshSVG(self):
        pos = self.m_position
        if pos >= self.db_cluster.shape[0]:
            pos = self.db_cluster.shape[0]-1

        fen = self.db_cluster.FEN[ pos ]
        move = self.db_cluster.move[ pos ]

        tmp = self.svg_of_board_and_move(fen, move)
        #file = open("last_pos.svg", "w")
        #file.write( tmp )
        #file.close()
        return "data:image/svg+xml;utf8,"+tmp

def main_visualizer(errors_fn, labels_fn):
    db = pd.read_csv( errors_fn, sep=';')
    labels = np.loadtxt( labels_fn, dtype=np.int64 )

    app = QGuiApplication([])
    backend = Backend(db, labels)
    engine = QQmlApplicationEngine()
    engine.quit.connect(app.quit)
    engine.rootContext().setContextProperty("backend", backend)

    path = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(path, 'visualizer_layout.qml')
    engine.load(path)

    if len(engine.rootObjects()) != 1:
        sys.exit(-1)

    sys.exit(app.exec())


if __name__ == "__main__":

    app = QGuiApplication(sys.argv)

    db = pd.read_csv( "data/errors/OUT_ERRORS_DEFINITIVE.csv", sep=';')
    labels = np.loadtxt( "data/cluster_labels/Cluster_Labels_dif2vec.txt", dtype=np.int64 )

    if False:
        vectors = np.load("data/nn_vectors/OUT_ERRORS_DEFINITIVE.npy")
        db = db[vectors[:,2] != 0]
        labels = np.zeros((db.shape[0]),dtype=np.int64)
    backend = Backend(db, labels)

    engine = QQmlApplicationEngine()
    engine.quit.connect(app.quit)
    engine.rootContext().setContextProperty("backend", backend)

    path = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(path, 'visualizer_layout.qml')
    engine.load(path)

    if len(engine.rootObjects()) != 1:
        sys.exit(-1)

    sys.exit(app.exec())
