import chess
import chess.svg

from PyQt5.QtSvg import QSvgWidget
from PyQt5.QtWidgets import QApplication, QWidget

def chessboard_svg( board ):
    return chess.svg.board(
        board
    )  

def move_svg( board_fen, move ):
    board = chess.Board(board_fen)
    move = chess.Move.from_uci(move)

    return chess.svg.board(
        board,
        orientation=board.turn,
        lastmove=move,
        arrows=[chess.svg.Arrow(move.from_square, move.to_square)]
    )

def save_svg( svg, path ):
    file = open(path, "w")
    file.write( svg )
    file.close()

class SvgWindow(QWidget):
    def __init__(self, svg):
        super().__init__()

        self.setGeometry(0, 0, 820, 820)

        self.widgetSvg = QSvgWidget(parent=self)
        self.widgetSvg.setGeometry(10, 10, 810, 810)

        self.widgetSvg.load(svg.encode("UTF-8"))

def show_svg( svg ):

    app = QApplication([])
    window = SvgWindow(svg)
    window.show()
    app.exec()

if __name__ == "__main__":
    # fen = input("FEN: ")
    # move = input("move: ")
    fen = "r4rk1/p4p2/4pq1p/5p2/1p3PP1/1Pn1R2Q/P6P/1B3R1K b - - 0 28"
    move = "f5g4"
    show_svg( move_svg( fen, move) )



