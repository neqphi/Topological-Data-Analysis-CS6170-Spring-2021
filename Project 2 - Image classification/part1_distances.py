import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import warnings
import math

from ripser import ripser
from sklearn import datasets, manifold

from gudhi import bottleneck_distance
from persim import plot_diagrams, wasserstein
from PIL import Image

def file2dgm(filename):
    with open(filename,"r") as f:
        datalist = [[float(point) for point in line.strip("[)\n").split(",")] for line in f]
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
    #1. Barcode lists
    dgms = [[],[]]
    for imgname in imageList:
        for dim in [0,1]:
            dgm = file2dgm(f'{output_path_prefix}dim{dim}/{imgname}.txt')
            dgms[dim].append(dgm)

    #2 Compute distance matrices

    #Bottleneck
    bottlenecks = [dgms2dist_matrix(dgms[i],bottleneck_distance) for i in range(2)]

    for dim in [0,1]:
        mds = manifold.MDS(n_components = 2, dissimilarity='precomputed')
        transformed_data = mds.fit_transform(bottlenecks[dim])
        plot_embedding(transformed_data, "MDS", "Bottleneck", dim)

        tsne = manifold.TSNE(n_components = 2, metric= 'precomputed')
        transformed_data = mds.fit_transform(bottlenecks[dim])
        plot_embedding(transformed_data, "TSNE", "Bottleneck", dim)

    #Wasserstein
    wassersteins = [dgms2dist_matrix(dgms[i],wasserstein) for i in range(2)]

    for dim in [0,1]:
        mds = manifold.MDS(n_components = 2, dissimilarity='precomputed')
        transformed_data = mds.fit_transform(wassersteins[dim])
        plot_embedding(transformed_data, "MDS", "Wasserstein", dim)

        tsne = manifold.TSNE(n_components = 2, metric= 'precomputed')
        transformed_data = mds.fit_transform(wassersteins[dim])
        plot_embedding(transformed_data, "TSNE", "Wasserstein", dim)

    #Image Distance
    imgs = [f'{input_path_prefix}{imgname}' for imgname in imageList]
    imgDistances = dgms2dist_matrix(imgs,imgDist)

    mds = manifold.MDS(n_components = 2, dissimilarity='precomputed')
    transformed_data = mds.fit_transform(imgDistances)
    plot_embedding(transformed_data, "MDS", "RawImageDist", "")

    tsne = manifold.TSNE(n_components = 2, metric= 'precomputed')
    transformed_data = mds.fit_transform(imgDistances)
    plot_embedding(transformed_data, "TSNE", "RawImageDist", "")

main()
