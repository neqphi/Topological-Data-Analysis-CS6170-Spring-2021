import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import warnings
import math

from ripser import ripser
from sklearn import metrics
from sklearn.svm import SVC

#from gudhi import representations.vector_methods.PersistenceImage
from gudhi.representations.kernel_methods import PersistenceScaleSpaceKernel
from persim import plot_diagrams, wasserstein
from PIL import Image

def file2dgm(filename):
    datalist = []
    with open(filename,"r") as f:
        for line in f:
            points = line.strip("[)\n").split(",")
            points[0] = float(points[0])
            if points[1] == 'inf':
                points[1] = 100
            else:
                points[1] = float(points[1])
            datalist.append(points)
    return np.asarray(datalist)

def dgms2dist_matrix(dgms, metric):
    matrix = []
    size = len(dgms)
    print(f'Size:{size}')
    processStartTime = time.process_time()
    for i in range(size):
        rowStartTime = time.process_time()
        for j in range(size):
            if i == j:
                dist = 0
            elif i < j:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    dist = metric(dgms[i],dgms[j])
            else:
                dist = matrix[(size) * j + i] #By symmetry
            matrix.append(dist)
        print(f'Done:{i+1}/{size}({time.process_time()-rowStartTime:.2f}s)')
    print(f'{metric.__name__}-Done:{time.process_time()-processStartTime:.2f}s')
    return np.resize(np.asarray(matrix),(size,size))

def imgDist(img1_path, img2_path):
    with Image.open(img1_path) as img1, Image.open(img2_path) as img2:
        w1,h1 = img1.size
        w2,h2 = img2.size
        w = max(w1,w2)
        h = max(h1,h2)

        if (w1,h1) != (w2,h2):
            img1 = img1.resize((w,h))
            img2 = img2.resize((w,h))

        dist = 0
        for i in range(w):
            for j in range(h):
                dist += abs(img1.getpixel((i,j)) - img2.getpixel((i,j)))
    return dist/(w*h)

def plot_embedding(embedded_data, embedding, metric, dim):
    x = embedded_data[:, 0]
    y = embedded_data[:, 1]
    plt.figure()
    ax = plt.subplot(111)
    colors = cm.rainbow(np.linspace(0,1,8))
    cs = [colors[i//10] for i in range(len(x))]
    ax = plt.scatter(x,y,color=cs)
    if dim != "":
        dim_text = f'_dim{dim}'
    else:
        dim_text =""
    plt.title(f'{metric}_{embedding}{dim_text}')
    plt.savefig(f'plots/{metric}_{embedding}{dim_text}.png')
    return

def main():
    output_path_prefix = "barcodes/"

    input_path_prefix = "data/MPEG/"
    classList = ["camel",
                 "chicken",
                 "frog",
                 "horseshoe",
                 "key",
                 "octopus",
                 "pencil",
                 "sea_snake"]
    imageList = [f'{cl}-{num}.gif' for cl in classList for num in range(1,11)]

    startTime = time.process_time() #For measuring the total running time
    counter = 0
    #1. Barcode lists
    dgms = [[],[]]
    list4Label = []
    list8Label = []

    #Label list initializion
    for cl in classList:
        #list8Label
        for i in range(10):
            list8Label.append(cl)

        #list4Label
        if cl in ["key", "pencil"]:
            for i in range(10):
                list4Label.append("oneleg")
        elif cl in ["chicken", "frog"]:
            for i in range(10):
                list4Label.append("fewlegs")
        elif cl in ["camel", "octopus"]:
            for i in range(10):
                list4Label.append("manylegs")
        elif cl in ["horseshoe", "sea_snake"]:
            for i in range(10):
                list4Label.append("ushape")

    for imgname in imageList:
        for dim in [0,1]:
            dgm = file2dgm(f'{output_path_prefix}dim{dim}/{imgname}.txt')
            dgms[dim].append(np.asarray(dgm))

    print(dgms[0][0].shape)

    #Split dgms into train/test data set
    dgms_train = [[],[]]
    dgms_test = [[],[]]
    list4Label_train = []
    list4Label_test = []
    list8Label_train = []
    list8Label_test = []

    for idx in range(80):
        if idx % 10 <5:
            for dim in [0,1]:
                dgms_train[dim].append(dgms[dim][idx])
            list4Label_train.append(list4Label[idx])
            list8Label_train.append(list8Label[idx])
        else:
            for dim in [0,1]:
                dgms_test[dim].append(dgms[dim][idx])
            list4Label_test.append(list4Label[idx])
            list8Label_test.append(list8Label[idx])

    list4Label_train = np.asarray(list4Label_train)
    list4Label_test = np.asarray(list4Label_test)
    list8Label_train = np.asarray(list8Label_train)
    list8Label_test = np.asarray(list8Label_test)

    for dim in [0,1]:
        startTime = time.process_time()
        print(f'Preparing for PSSK for training data..-- dim{dim}')
        pssk_train = PersistenceScaleSpaceKernel()
        pssk_train.fit(dgms_train[dim])
        kmatrix_train = pssk_train.transform(dgms_train[dim])
        print(f'PSSK for training data generated!-- {time.process_time()-startTime:2f}s')

        prTime = time.process_time()
        print(f'Prediction Initiated..')

        clf_4 = SVC(kernel='precomputed')
        clf_4.fit(kmatrix_train,list4Label_train)
        pssk_test = PersistenceScaleSpaceKernel()
        pssk_test.fit(dgms_test[dim])
        kmatrix_test = pssk_test.transform(dgms_test[dim])
        predicted_4 = clf_4.predict(kmatrix_test)
        print(f'Prediction generated! -- {time.process_time() - prTime:.2f}s')

        primTime = time.process_time()
        #Prediction
        print(f'Accuarcy: {metrics.accuracy_score(list4Label_test, predicted_4)}')
        print(f"Classification report for classifier {clf_4}:\n"
              f"{metrics.classification_report(list4Label_test, predicted_4)}\n")

        clf_8 = SVC(kernel='precomputed')
        clf_8.fit(kmatrix_train,list8Label_train)
        pssk_test = PersistenceScaleSpaceKernel()
        pssk_test.fit(dgms_test[dim])
        kmatrix_test = pssk_test.transform(dgms_test[dim])
        predicted_8 = clf_8.predict(kmatrix_test)
        print(f'Prediction generated! -- {time.process_time() - prTime:.2f}s')

        primTime = time.process_time()
        #Prediction
        print(f'Accuarcy: {metrics.accuracy_score(list8Label_test, predicted_8)}')
        print(f"Classification report for classifier {clf_8}:\n"
              f"{metrics.classification_report(list8Label_test, predicted_8)}\n")
        #clf_8 = SVC(kernel='precomputed')
        #clf_8.fit(kmatrix_train,list8Label_train)
        #predicted_8 = clf_8.predict(dgms_test)

    #2 Compute distance matrices

    #Bottleneck
    #bottlenecks = [dgms2dist_matrix(dgms[i],bottleneck_distance) for i in range(2)]
#
    #for dim in [0,1]:
        #mds = manifold.MDS(n_components = 2, dissimilarity='precomputed')
        #transformed_data = mds.fit_transform(bottlenecks[dim])
        #plot_embedding(transformed_data, "MDS", "Bottleneck", dim)
#
        #tsne = manifold.TSNE(n_components = 2, metric= 'precomputed')
        #transformed_data = mds.fit_transform(bottlenecks[dim])
        #plot_embedding(transformed_data, "TSNE", "Bottleneck", dim)

    #Wasserstein
    #wassersteins = [dgms2dist_matrix(dgms[i],wasserstein) for i in range(2)]

    #for dim in [0,1]:
        #mds = manifold.MDS(n_components = 2, dissimilarity='precomputed')
        #transformed_data = mds.fit_transform(wassersteins[dim])
        #plot_embedding(transformed_data, "MDS", "Wasserstein", dim)
#
        #tsne = manifold.TSNE(n_components = 2, metric= 'precomputed')
        #transformed_data = mds.fit_transform(wassersteins[dim])
        #plot_embedding(transformed_data, "TSNE", "Wasserstein", dim)

    #Image Distance
    #imgs = [f'{input_path_prefix}{imgname}' for imgname in imageList]
    #imgDistances = dgms2dist_matrix(imgs,imgDist)
#
    #mds = manifold.MDS(n_components = 2, dissimilarity='precomputed')
    #transformed_data = mds.fit_transform(imgDistances)
    #plot_embedding(transformed_data, "MDS", "RawImageDist", "")
#
    #tsne = manifold.TSNE(n_components = 2, metric= 'precomputed')
    #transformed_data = mds.fit_transform(imgDistances)
    #plot_embedding(transformed_data, "TSNE", "RawImageDist", "")

    #embedding = MDS(n_components = 2)
    #X_transformed = embedding.fit_transform(dissimillarity_matrix)
    #2-1 Plot using plt

main()
