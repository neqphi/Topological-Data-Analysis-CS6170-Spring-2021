import numpy as np
import time

from ripser import ripser
from sklearn import datasets
from persim import plot_diagrams

def file2barcodes(filename):
    with open(filename,"r") as f:
        datalist = [[float(point) for point in line.strip().split(" ")] for line in f]
    return ripser(np.asarray(datalist))['dgms']

def ndarray2file(ndarray,filename="barcodes.txt"):
    with open(filename,"w") as f:
        for point in ndarray:
            f.write("[" + str(point[0]) + "," + str(point[1]) +")\n") #[birth,death) format
    print("File {} is succesfully created.".format(filename))

def lifespan(bdpair):
    return bdpair[1]-bdpair[0]

"""
I. Octa
"""
startTime = time.process_time() #For measuring the total running time
barcodes_octa = file2barcodes("data/octa.txt")
longest_eight = sorted(barcodes_octa[1], key=lifespan, reverse=True)[0:8] #Dimension 1 barcodes
ndarray2file(longest_eight,"part1_barcodes/octa_longest_8_barcodes.txt")
print(f'It took {time.process_time() - startTime:.2f}s to process octa.txt.')
#plot_diagrams(barcodes_octa, show=True)

"""
II. Cylinder
"""
startTime = time.process_time()
barcodes_cylinder = file2barcodes("data/cylinder.txt")
longest_one = sorted(barcodes_cylinder[1], key=lifespan, reverse=True)[0:1] #Dimension 1 barcode
ndarray2file(longest_one,"part1_barcodes/cylinder_longest_barcode.txt")
print(f'It took {time.process_time() - startTime:.2f}s to process cylinder.txt.')
#plot_diagrams(barcodes_cylinder, show=True)
