import sys
sys.path.append('src')

from tkinter import Tk     # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilenames, askopenfilename
import numpy as np
import matplotlib.pyplot as plt

Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing

def menu():
    print("\n","-"*50)
    print("Este es el Trabajo de Fin de Grado de Miguel Moles Mestre")
    print("Desde aquí podrá ejecutar todas las partes invididuales de él")
    print("")
    print("Opciones:")
    print("0) Ejecutar visualizador")
    print("1) Analizar movimientos de un archivo PGN en busca de errores")
    print("2) Transformar un archivo de errores en vectores de características usando los cuatro modelos diferentes")
    print("3) Obtener información sobre las características nulas de los vectores de características con la opción de aplicar filtros")
    print("4) Ver estadísticos de los datos cribados")
    print("5) Acumular características poco pobladas, orden ya establecido")
    print("6) Entrenar al algoritmo K-medias con los parámetros estudiados")
    print("7) Usar un modelo K-medias ya entrenado para clasificar datos ya preprocesados")
    print("-"*50)
    print("10) Aplicar todo el preprocesado y clasificar un archivo de vectores de características")
    print("-"*50)
    print("12) Regenerar mov2vec, dif2vec y pos2vec2 desde el DeepChess original")
    print("15) Ejecutar el algoritmo de preprocesamiento de acumulación de características")
    print("-"*50)
    print("x) SALIR")

    option = input("Elija una opción: --> ")

    return option

while True:
    option = menu()

    if option == "x":
        break

    if(option == "0"):
        import pandas as pd
        import visualizer as vsz

        print("Indique el archivo csv de errores")
        errors_fn = askopenfilename()
        print("Indique el archivo de etiquetas de los errores")
        labels_fn = askopenfilename()
        vsz.main_visualizer(errors_fn, labels_fn)

    if(option == "1"):
        import chess.engine
        import asyncio
        import get_errors as gerr

        print("Elija el archivo PGN a analizar")
        filename = askopenfilename()

        DEFAULT_VAL = 6
        print("¿Cuántas instancias del motor quieres ejecutar?")
        engine_number = input(f"({DEFAULT_VAL} por defecto) ---> ")
        engine_number = DEFAULT_VAL if engine_number == "" else round(float(engine_number))
        engine_number = 1 if engine_number<1 else engine_number

        DEFAULT_VAL = 250
        print("[Parámetro 1] CentiPeones para considerar que una posición es ganada")
        max_pos_eval = input(f"({DEFAULT_VAL} por defecto) ---> ")
        max_pos_eval = DEFAULT_VAL if max_pos_eval == "" else round(float(max_pos_eval))
        max_pos_eval = 0 if max_pos_eval<0 else max_pos_eval

        DEFAULT_VAL = 50
        print("[Parámetro 2] Impacto mínimo de un movimiento en la posición para ser considerado error (en CentiPeones)")
        min_impact_move = input(f"({DEFAULT_VAL} por defecto) ---> ")
        min_impact_move = DEFAULT_VAL if min_impact_move == "" else round(float(min_impact_move))
        min_impact_move = 0 if min_impact_move<0 else min_impact_move

        DEFAULT_VAL = 5
        print("[Parámetro 3] Cuántos turnos ignorar al comienzo de una partida")
        moves_to_ignore = input(f"({DEFAULT_VAL} por defecto) ---> ")
        moves_to_ignore = DEFAULT_VAL if moves_to_ignore=="" else round(float(moves_to_ignore))
        moves_to_ignore = 0 if moves_to_ignore<0 else moves_to_ignore

        DEFAULT_VAL=0.25
        print("[Parámetro 4] Tiempo límite de la evaluación de cada posición")
        time_eval = input(f"({DEFAULT_VAL} por defecto) ---> ")
        time_eval = DEFAULT_VAL if time_eval=="" else float(time_eval)

        DEFAULT_VAL=8
        print("[Parámetro 5] Profundidad máxima de movimientos en la evaluación de cada posición")
        depth_eval = input(f"({DEFAULT_VAL} por defecto) ---> ")
        depth_eval = DEFAULT_VAL*2  if depth_eval=="" else round(float(depth_eval)*2)

        out_path = "data/errors/OUT_ERRORS.csv"

        print(f"Los errores se escribirán en el archivo {out_path}")
        print("El análisis se puede cancelar prematuramente si lo desea")
        gerr.errors_main([filename], max_pos_eval, min_impact_move, moves_to_ignore, time_eval, depth_eval, out_path, engine_number)
        info = gerr.Info_progress()

    elif option == "2":
        import nn_functions as nn
        import pandas as pd
        from scipy.sparse import save_npz

        base_path = "data/nn_vectors/"
        print("Seleccione el archivo CSV de errores")
        filename = askopenfilename()
        data = pd.read_csv( filename, sep=';'
                            # skiprows=0, nrows=100000)
                            )

        mov2vec = nn.load_model_mov2vec()
        dif2vec = nn.load_model_dif2vec()
        pos2vec1 = nn.load_model_pos2vec1()
        pos2vec2 = nn.load_model_pos2vec2()

        print(f"Transformando con mov2vec:")
        out = nn.lots_analisis(mov2vec, data.FEN, data.move, 200)
        print(f"La forma del output es: {out.shape}")
        save_npz( base_path+"OUT_VECTORS_mov2vec.npz" , out )

        print(f"Transformando con dif2vec:")
        out = nn.lots_analisis(dif2vec, data.FEN, data.move, 100)
        print(f"La forma del output es: {out.shape}")
        save_npz( base_path+"OUT_VECTORS_dif2vec.npz" , out )

        print(f"Transformando con pos2vec1:")
        out = nn.lots_analisis(pos2vec1, data.FEN, None, 100)
        print(f"La forma del output es: {out.shape}")
        save_npz( base_path+"OUT_VECTORS_pos2vec1.npz" , out )

        print(f"Transformando con pos2vec2:")
        out = nn.lots_analisis(pos2vec2, data.FEN, None, 100)
        print(f"La forma del output es: {out.shape}")
        save_npz( base_path+"OUT_VECTORS_pos2vec2.npz" , out )

        print("Los vectores se han escrito en los archivos", base_path+"OUT_VECTORS_***.npz")

    elif option == "3":
        from scipy.sparse import load_npz
        import numpy as np

        print("Seleccione el archivo NPZ de vectores")
        filename = askopenfilename()
        db_raw = load_npz(filename)

        db = db_raw.todense()
        del(db_raw)

        print("Info de la DB:")
        print("Forma:",db.shape)

        print("Características que son todo 0s:", np.sum(np.all(db==0,axis=0)))
        print("Instancias que son todo 0s:",np.sum(np.all(db==0,axis=1)))
        print("Porcentaje de datos no 0s:", np.sum(db!=0)*100/db.size, "%")
        print("Mediana de características no 0s por instancia:",np.median(np.sum(db!=0,axis=1), axis=0) )
        print()

        print("Características que son todo 0s:", repr(np.where(np.all(db==0,axis=0))[1]))

        print("Si las quitásemos: ")

        any_value_cols = np.flatnonzero(np.any(db!=0,axis=0))
        # print(any_value_cols)
        db = db[:,any_value_cols]

        print("Características que son todo 0s:", np.sum(np.all(db==0,axis=0)))
        print("Instancias que son todo 0s:",np.sum(np.all(db==0,axis=1)))
        print("Porcentaje de datos no 0s:", np.sum(db!=0)*100/db.size, "%")
        print("Mediana de características no 0s por instancia:",np.median(np.sum(db!=0,axis=1), axis=0) )


        print("Forma final:",db.shape)

        if( input("¿Quieres sobreescribir los datos cribados? (si/no): ") == "si"):
            filename = filename.split('.')[0] + '_no_c0s.npy'
            np.save(filename, db)
            print("Se han guardado como", filename)
        print()

    if(option == "4"):
        import clustering as cl
        from scipy.sparse import load_npz
        import numpy as np

        filename = askopenfilename()
        db_init = np.load(filename)
        # Esto es para emparejar las características siamesas de mov2vec y dif2vec
        good_indexes = [(i//2)+(db_init.shape[1]//2)*(i%2) for i in range(db_init.shape[1])]
        print(good_indexes)
        db = db_init[:,good_indexes]

        general_mean = cl.medias_datos(db)
        general_var = cl.varianzas_datos(db)
        zeros = cl.valores_0_datos(db)
        nonzeros = db.shape[0] - zeros
        nonzeros_percentage = (nonzeros/db.shape[0])*100

        # Estadísticos característica a característica
        specific_mean = np.array(
                [ np.mean(db[np.nonzero(db[:,i])][i]) for i in range(db.shape[1]) ]
        )
        specific_var = np.array(
                [ np.var(db[np.nonzero(db[:,i])][i]) for i in range(db.shape[1]) ]
        )
        specific_min = np.array(
                [ np.min(db[np.nonzero(db[:,i])][i]) for i in range(db.shape[1]) ]
        )
        specific_max = np.array(
                [ np.max(db[np.nonzero(db[:,i])][i]) for i in range(db.shape[1]) ]
        )
        # Labels de las características
        indexes = [f"c{(i//2)+1}_{(i%2)+1}" for i in range(db.shape[1])]
        # Tabla final
        data_stats = np.stack((specific_mean,specific_var,specific_min,specific_max,nonzeros,nonzeros_percentage),axis=0).T

        from tabulate import tabulate
        print(tabulate(data_stats, headers=["Media","Varianza","Mínimo","Máximo","Valores no 0s","% no 0s"], tablefmt="grid", showindex=indexes))
        #print("Media de la DB:", general_mean)
        #print("Varianza de la DB:", general_var)
        #print("Porcentaje de datos no 0s:", nonzeros_percentage)

    if(option == "5"):
        import numpy as np
        np.set_printoptions(suppress=True)
        # La proporción máxima de datos no-zero para que la característica sea acumulada
        MIN_NONZERO = 0.05
        # Proporción de datos no-zero mínima que tendrán las características de acumulación
        TOP_ACC = 0.1
        filename = askopenfilename()
        db = np.load(filename)

        model = input("De cuál de los métodos de extracción de características vienen los datos?\n---> (mov2vec, dif2vec, pos2vec1, pos2vec2)\n---> ")
        orders = {
            "mov2vec":[[ 2, 5,14],[17, 20, 29]],
            "dif2vec":[[ 2, 5,14]],
            "pos2vec1":[[21,67,34,73,74,35,71,2,39,60,20,7,62,77],[9,8,18,44,51,79,52,6,13,49,
            61,65,59,33,69,41,84,12,26,30,48]],
            "pos2vec2":[[6,33,24,36,37,11,45,5,15,18,26,2,32,0,16,41,39,30,23,22,25,7,20,4,
            46,17,21,31,12,10,8,40,28]]
        }
        order = orders[model]

        # Eliminar características del orden
        base = np.delete(db, sum(order,[]), axis=1)
        print("Forma de la db sin las características a acumular:",base.shape)
        # Característica a acumular
        c = np.zeros((db.shape[0]))

        for l in order:
            for i in l:
                c[db[:,i] != 0] = db[db[:,i] != 0, i]
            base = np.c_[base, c]
            c = np.zeros((db.shape[0]))

        print("Forma de la db sin las características a acumular:",base.shape)

        filename = filename.split('.')[0] + '_refiltrado.npy'
        np.save(filename, base)
        print("Se han guardado los datos en "+filename)

    if(option == "6"):
        import clustering as cl
        filename = askopenfilename()
        clusters = {"mov2vec":20, "dif2vec":20, "pos2vec1":30, "pos2vec2":20}
        model = input("De cuál de los métodos de extracción de características vienen los datos?\n---> (mov2vec, dif2vec, pos2vec1, pos2vec2)\n---> ")
        out_path = "data/kmeans_models/Out_Kmeans_"+model+".joblib"
        label_path = "data/cluster_labels/Cluster_Labels_"+model+".txt"
        cl.train_save_kmeans(filename, clusters[model], out_path, label_path)
        print("El modelo se ha guardado como", out_path)
        print("Las etiquetas de los datos se encuentran en",label_path)

    if(option == "7"):
        from joblib import dump, load
        from sklearn.cluster import KMeans
        print("Indique con qué modelo quiere clasificar los vectores")
        model_fn = askopenfilename()
        print("Indique qué archivo de datos quiere clasificar")
        db_fn = askopenfilename()

        kmeans = load(model_fn)
        db = np.load(db_fn)

        labels = kmeans.predict(db)

        out_fn = "data/cluster_labels/OUT_LABELS.txt"
        np.save_txt(out_fn, labels)
        print("Las etiquetas se han guardado en", out_fn)

    if(option == "10"):
        import clustering as clust
        print("Indique con qué modelo quiere clasificar")
        model_fn = askopenfilename()

        print("Indique el archivo de vectores npz sin procesar")
        filename = askopenfilename()
        from scipy.sparse import load_npz
        db = load_npz(filename).todense()

        model = input("De cuál de los métodos de extracción de características vienen los datos?\n---> (mov2vec, dif2vec, pos2vec1, pos2vec2)\n---> ")

        # ELIMINACION DE CARACTERISTICAS NO 0
        db = clust.preproc_remove_zero_value_dims(db, model)

        # ACUMULACIÓN DE CARACTERÍSTICAS POCO POBLADAS
        db = clust.preproc_accum_low_density_dims(db, model)

        # CARGAMOS EL MODELO DE CLUSTERING
        from joblib import dump, load
        from sklearn.cluster import KMeans
        kmeans = load(model_fn)

        # Ejecutamos el labeling
        labels = kmeans.predict(db)

        # Vemos estadísticos
        values, counts = np.unique(labels, return_counts=True)
        sorted_index = counts.argsort()
        percentages = counts*100/labels.size

        for c, v in zip(counts[sorted_index],values[sorted_index]):
            print("Cluster",v,": ", c*100/labels.size,"%")

        print("¿Quieres visualizar las proporciones de cada cluster?")
        option = input("[y/N] --> ")
        if option in ["y", "Y"]:
            THRESHOLD=3
            others = np.sum(percentages[percentages<=THRESHOLD])

            plot_perc = percentages[percentages>THRESHOLD]
            graph_labels = ["Cl"+str(x+1) for x in values[percentages>THRESHOLD]]

            plot_perc = np.append(plot_perc, others)
            graph_labels.append("Otros")

            plt.pie(plot_perc,autopct='%1.1f%%', labels=graph_labels)
            plt.title("Porcentaje de pertenencia de los datos a los clusters de "+model)
            plt.show()

        print("Quieres guardar las etiquetas?")
        ans = input("[y/N] --> ")
        if ans in ["y", "Y"]:
            label_fn = "data/cluster_labels/OUT_LABELS.txt"
            np.save_txt(label_fn, labels)
            print("Se han guardado en el archivo",label_fn)

    if(option == "12"):
        # Funciones custom propias
        import nn_functions as nn
        import tensorflow as tf

        default_model = nn.load_model_deepchess()

        mov2vec = nn.make_mov2vec_model(default_model)
        dif2vec = nn.make_diff2vec_model(default_model)
        pos2vec2 = nn.make_pos2vec2_model(default_model)

        mov2vec.save(nn.nn_mov2vec)
        dif2vec.save(nn.nn_dif2vec)
        pos2vec2.save(nn.nn_pos2vec2)
        print("Listo")

    if(option == "15"):
        print("Indique qué archivo de datos quiere clasificar")
        db_fn = askopenfilename()
        db = np.load(db_fn)

        MIN_NONZERO = 0.05
        TOP_ACC = 0.1

        # Porcentaje de datos 0 por característica
        perc_0_per_car = np.sum(db!=0, axis=0)*100/db.shape[0]
        # Ordenarlos
        indexes = np.argsort(perc_0_per_car)

        # Índices de las características a eliminar
        indexes = (perc_0_per_car < (MIN_NONZERO*100)).nonzero()[0]
        final_order = indexes.copy()
        best_overriten = db.shape[0]

        #Algoritmo de prueba y error
        for times in range(500):
            # Orden mezclado
            np.random.shuffle(indexes)

            c = np.zeros((db.shape[0]))
            c_indexes = []
            out = []
            # Array booleano de los valores que serían sobreescritos
            overriten = np.full((db.shape[0]), False)
            overriten_value = 0

            for i in indexes:
                # Actualizamos valores sobreescritos
                overriten = overriten | ( (c != 0) & (db[:,i] != 0)  )
                # Añadimos valores no cero
                c[db[:,i] != 0] = db[db[:,i] != 0, i]
                c_indexes.append(i)

                # Cuando la característica de acumulación esté llena, contamos los valores sobreescritos y reseteamos las variables
                if np.sum(c != 0)/c.size > TOP_ACC:
                    overriten_value += np.sum(overriten)
                    c = np.zeros((db.shape[0]))
                    overriten = np.full((db.shape[0]), False)
                    out.append(c_indexes)
                    c_indexes = []

            overriten_value += np.sum(overriten)
            out.append(c_indexes)
            if overriten_value < best_overriten:
                best_overriten = overriten_value
                order = out.copy()
        print( "Final order:", order )
        print( "Overriten values:", best_overriten )
