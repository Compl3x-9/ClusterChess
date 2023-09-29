import asyncio
import chess
import chess.engine
import chess.pgn
import sys


# Clase que une a más de un archivo PGN para su lectura de juegos, como si
# se tratase de un sólo archivo
class PGN_files:
    def __init__(self):
        self._name_files = []
        self._curr_pgn = None
        self._index = 0

    # Añade el path de un archivo PGN
    def add_pgn_file(self, file):
        self._name_files.append(file)

    # Lee un juego de los archivos PGN
    def read_game(self):
        if self._curr_pgn == None:
            if self._index >= len(self._name_files):
                return None
            self._curr_pgn = open(self._name_files[self._index])
            self._index += 1
        game = chess.pgn.read_game(self._curr_pgn)
        if game == None:
            self._curr_pgn.close()
            if self._index >= len(self._name_files):
                return None
            self._curr_pgn = open(self._name_files[self._index])
            self._index += 1
            game = chess.pgn.read_game(self._curr_pgn)
        return game

# Variables globales para los contadores de juegos, posiciones y errores leídos
class Info_progress:
    def __init__(self):
        self.games_counter = 0
        self.positions_counter = 0
        self.errors_counter = 0

# Función que determina si un movimiento es error o no
def is_error(peval, neval):
    return (neval - peval) < -50

# Función que ejecuta un motor de ajedrez y analiza partidas hasta que no
# quedan más que analizar
# pgn es de tipo PGN_files, eval_limits es de tipo chess.engine.Limit,
# out_file es un archivo de escritura y info sirve para hacer un contador de
# posiciones, partidas y errores evaluados
async def engine_eval(pgn, max_pos_eval, min_impact_move, moves_to_ignore, time_eval, depth_eval, out_file, info, index) -> None:

    eval_limits = chess.engine.Limit(
            time=None if time_eval<0 else time_eval,
            depth=None if depth_eval<0 else depth_eval
            )
    transport, engine = await chess.engine.popen_uci("/usr/bin/stockfish")

    while True:
        game = pgn.read_game()
        # End of loop
        if game == None:
            break
        board = game.board()
        moves_it = iter( game.mainline_moves() )

        # ignore first n moves
        i = 0

        while i<moves_to_ignore*2 and (move := next(moves_it, None)) is not None :
            board.push(move)
            i += 1

        # print("[DEBUG]", "Análisis,",move.uci()," && ",board.fen())
        eval_cp = (await engine.analyse(board, eval_limits))["score"]
        # print("[DEBUG]","Primer análisis hecho,", eval_cp)
        info.positions_counter += 1

        while (move := next(moves_it, None)) is not None:
            is_capture = board.is_capture(move)
            # print("[DEBUG]",move.uci()," && ",board.fen())
            board.push(move)
            tmp = await engine.analyse(board, eval_limits)
            info.positions_counter += 1
            prev_eval_cp = eval_cp
            eval_cp = tmp["score"]

            peval = prev_eval_cp.relative.score(mate_score=4000)
            neval = eval_cp.pov( prev_eval_cp.turn ).score(mate_score=4000)

            if neval > 2*max_pos_eval:
                continue
            if neval > max_pos_eval and (peval-neval) < 2*min_impact_move:
                continue
            if peval < -2*max_pos_eval:
                continue
            if peval < -max_pos_eval and (peval-neval) < 2*min_impact_move:
                continue

            if (peval-neval) >= min_impact_move:
                diff = peval - neval
                info.errors_counter += 1
                board.pop()
                out_file.write( board.fen()+";"+move.uci()+";"+str(is_capture)+";"+str(peval)+";"+str(diff)+"\n" )
                board.push(move)
        info.games_counter += 1

    await engine.quit()

# Una status bar del progreso del análisis de partidas
# El primer param. es la información de progreso, y el segundo,
# el intervalo de refresco en segundos
async def info1(info, interval) -> None:
    time_passed = 0
    prev_positions = info.positions_counter

    await asyncio.sleep(interval)
    while info.positions_counter - prev_positions > 0:
        time_passed += interval
        print("\r",info.games_counter," games (",round(info.games_counter/time_passed,2),"/s); ", sep='', end='')
        print(info.positions_counter," positions (",round(info.positions_counter/time_passed,2),"/s); ", sep='', end='')
        print(info.errors_counter, " errors (",round(info.errors_counter/time_passed,2),"/s);                          ", sep='', end='')
        prev_positions = info.positions_counter
        await asyncio.sleep(interval)


def errors_main(pgn_paths, max_pos_eval, min_impact_move, moves_to_ignore, time_eval, depth_eval, out_path, engine_number):

    pgn = PGN_files()

    for filename in pgn_paths:
        pgn.add_pgn_file( filename )

    out_file = open(out_path,"w") #File to write the errors
    out_file.write("FEN;move;is_capture;first_eval;move_impact\n")
    info = Info_progress()

    async def main() -> None:
        tasks = [info1(info, 1)]
        for i in range(engine_number):
            tasks.append(asyncio.create_task(
                engine_eval(pgn, max_pos_eval, min_impact_move, moves_to_ignore, time_eval, depth_eval, out_file, info, i)
            ))

        await asyncio.gather(*tasks)
    asyncio.run(main())
    print("\n")

def errors_default_main(pgn_paths, out_path, engine_number=6):

    errors_main(pgn_paths, 250, 50, 5, 0.25, 16, out_path, engine_number)

# IN CASE YOU WANT TO TEST IT
if __name__ == "__main__":
    errors_main([sys.argv[1]], "OUT_ERRORS.csv", engine_number=2)
