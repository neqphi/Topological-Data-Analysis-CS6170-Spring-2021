import time
import numpy as np
from ripser import ripser
from sklearn import datasets
from persim import plot_diagrams
from PIL import Image

def lifespan(bdpair):
    return bdpair[1]-bdpair[0]

def isBoundary(img,i,j):
    #returns true if any of eight adjacent pixels have different color from (i,j)-cell.

    myPixel = img.getpixel((i,j))
    surroundingIndices = [(i-1,j-1),(i-1,j),(i-1,j+1),(i,j-1),(i,j+1),(i+1,j-1),(i+1,j),(i+1,j+1)]
    pixelLists = []
    for pair in surroundingIndices:
        if -1 in pair: #We exclude index being -1
            continue
        else:
            try:
                pixelLists.append(img.getpixel(pair)) #We exclude index larger than height/width of the img.
            except:
                pass

    for pixel in pixelLists:
        if myPixel != pixel: #different pixel
            return True

    return False

def img2barcodes(imgpath):
    im = Image.open(imgpath)
    w,h = im.size
    boundaryList = []

    for i in range(0,w,2): #To reduce computational complexity 
        for j in range(0,h,2):
            if isBoundary(im,i,j):
                boundaryList.append((i,j))
    return ripser(np.asarray(boundaryList),maxdim=0)['dgms']

def ndarray2file(ndarray,filename="barcodes.txt"):
    with open(filename,"w") as f:
        for point in ndarray:
            f.write("[" + str(point[0]) + "," + str(point[1]) +")\n") #[birth,death) format
    print(f'File {filename} is succesfully created.')

def main():
    input_path_prefix = "data/MPEG/"
    output_path_prefix = "part2_barcodes(boundary)/"
    imageList = ["bat-19.gif",
                 "beetle-13.gif",
                 "butterfly-10.gif",
                 "camel-16.gif",
                 "cattle-3.gif",
                 "crown-10.gif",
                 "device3-5.gif",
                 "dog-7.gif",
                 "horse-18.gif",
                 "octopus-12.gif"]

    startTime = time.process_time() #For measuring the total running time
    counter = 0
    for imgname in imageList:
        print(f"Initiating to work on {imgname}..")
        counter += 1
        processStartTime = time.process_time() #For measuring the running time for each img processing
        output_filename = output_path_prefix + imgname + ".txt"
        barcodes_img = sorted(img2barcodes(input_path_prefix + imgname)[0], key=lifespan, reverse=True)
        ndarray2file(barcodes_img,filename=output_filename)
        print(f'It took {time.process_time() - processStartTime:.2f}s to process {imgname}. ({counter}/{len(imageList)})')

    print(f'Done! It took {time.process_time() - startTime:.2f}s to process the {len(imageList)} images.')

main()
