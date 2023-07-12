import math, time, operator
import numpy as np
from ripser import ripser
from sklearn import datasets
from persim import plot_diagrams
from PIL import Image

class Vertex:
    """
    Has Union-Find data structure.
    Represents each white pixel from the given image, with tracking its birth and death time.

    *Property:  pixel (stores the pixel as tuple)
                (for union-find DS) size, parent
                (for persistence barcodes) birth, death, height(height function value)
    *Method : Union, Find(find Ancestor)
    """
    def __init__(self, pixel):
        #Create a new Vertex from pixel information and the index of it from the data list
        self.pixel = pixel

        #set default values
        self.size = 1
        self.parent = self
        self.birth = -np.inf
        self.death = np.inf
        self.height = 0

    def __str__(self):
        #Specifies a string representation of a Vertex instance.
        return str(self.pixel)

    def __eq__(self, other):
        #Returns True if two have the same pixels
        return self.pixel == other.pixel

    def __ne__(self, other):
        #The opposite of __eq__.
        return not self.__eq__(other)

    #Union-Find
    def find(self):
        result = self
        while result != result.parent:
            result = result.parent
        self.parent = result #Path compression
        return result

    def union(self,other):
        a1 = self.find()
        a2 = other.find()

        if a1 != a2:
            l = sorted([a1,a2], key = operator.attrgetter('birth')) #older becomes parent
            l[1].parent = l[0]
            l[0].size += 1
            return True

        #Returns false if two vertices have already been in one component
        return False

def img2vertices(imgpath,thresh):
    with Image.open(imgpath) as im:
        w,h = im.size
        vertexSet = []
        step = math.floor(thresh/math.sqrt(2)) #
        for i in range(0,w,step): #Sample only in odd-indices pixels
            for j in range(0,h,step):
                if im.getpixel((i,j)) != 0: #non black
                    vertexSet.append(Vertex((i,j)))
    return vertexSet

def dist(v,w):
    #Returns the Euclidean distance between the vertices v and w.
    a = np.asarray(v.pixel)
    b = np.asarray(w.pixel)
    diff = a-b
    return math.sqrt(np.dot(a-b,a-b))

def height(v, b, d=(1,0), r=1):
    #Returns the value of the height function of the given vertex (defined as in [Hofer-Kwitt-Niethammer-Uhl])
    #Also, sets the birth of the vertex to be the function value.
    #v: vertex
    #b: barycenter of the image
    #d: direction (will be normalized)
    #r: normalizer of the result

    p = np.asarray(v.pixel)
    b = np.asarray(b)
    d = np.asarray(d)

    d = d/math.sqrt(np.dot(d,d)) #normalize direction
    result = np.dot(p-b,d)/r
    v.height = result
    return result

def vSet2eSet(vSet,thresh=4):
    #Form an edge between two vertices if the distance less than or equal to 4,
    #and returns the edge set.
    eSet = []
    for i in range(len(vSet)):
        for j in range(i+1,len(vSet)):
            if dist(vSet[i],vSet[j]) <= thresh:
                eSet.append((vSet[i],vSet[j]))
    return eSet

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

def img2barcodes(imgpath, b, d=(1,0), r=1, thresh=4):
    #b,d,r are the parameters for the height function. (barycenter, direction vector, normalizer)
    #thresh : threshold for distance between vertices, for edge formation.
    vSet = img2vertices(imgpath,thresh)
    vSet.sort(key=lambda v: height(v,b,d,r))

    for v in vSet:
        v.birth = v.height

    print(f'Number of vertices: {len(vSet)}')
    #As we have sorted with height, which are also the births of vertices,
    #vSet is now sorted from the youngest to the oldest.

    eSet = vSet2eSet(vSet,thresh)
    #By our earlier sort of vSet, every edge is written as pair of vertices of the form:
    #(younger vertex, older vertex).
    #We now have to sort eSet from the oldest to youngest.
    #This can be done by comparing the birth(=height) of the younger vertex (e[1]) in each edge e.
    eSet.sort(key=lambda e: e[1].birth)
    print(f'Number of edges: {len(eSet)}')
    #Now it's time to find the death of each vertex.
    #A vertex v dies when an edge (w,v) exists, where w is some younger vertex.
    #The death time of v should be the same as the same of its birth time,
    #because the filtration is driven by the height functions for vertices and edges,
    #and the height of an edge (v,w) is defined to be max(height(v), height(w)).
    #Therefore, one of vertices of an edge should die as soon as it borns.

    for e in eSet:
        u = e[0] #Older vertex of e
        v = e[1] #Younger vertex of e

        if u.find() != v.find(): #Same ancestors, i.e. connected
            u_ancestor = u.find()
            v_ancestor = v.find()
            u.union(v)

            l = sorted([u_ancestor,v_ancestor], key=operator.attrgetter('birth'))
            l[1].death = v.birth

        else:
            continue  #u,v were in the same component, so we don't have to update death.

    """
    Return the (birth, death) pairs of vertices in ndarray
    """
    return np.asarray([[v.birth,v.death] for v in vSet])

def ndarray2file(ndarray,filename="barcodes.txt"):
    with open(filename,"w") as f:
        for point in ndarray:
            f.write("[" + str(point[0]) + "," + str(point[1]) +")\n") #[birth,death) format
    print(f'File {filename} is succesfully created.')

def main():
    input_path_prefix = "data/MPEG/"
    output_path_prefix = "part2_barcodes(heightFtn)/"
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
    dList = [(1,0), #Assigns different pre-determined direction vectors for each image
             (0,1),
             (1,4),
             (0,-1),
             (1,-1),
             (0,1),
             (0,-1),
             (-1,0),
             (1,-1),
             (0,-1)]

    startTime = time.process_time() #For measuring the total running time
    counter = 0
    for imgname in imageList:
        counter += 1
        print(f"Started working on {imgname}..")
        processStartTime = time.process_time() #For measuring the running time for each img processing
        output_filename = output_path_prefix + imgname + ".txt"

        with Image.open(input_path_prefix + imgname) as im:
            w,h = im.size
            b = (math.floor(w/2), math.floor(h/2))
            r = max(w,h)
            d= dList[counter-1]

        barcodes_img = sorted(img2barcodes(input_path_prefix + imgname, b,d,r,thresh=4 * math.sqrt(2)), key=lifespan, reverse=True)
        ndarray2file(barcodes_img,filename=output_filename)
        print(f'It took {time.process_time() - processStartTime:.2f}s to process {imgname}. ({counter}/{len(imageList)})')

    print(f'Done! It took {time.process_time() - startTime:.2f}s to process the {len(imageList)} images.')

main()
