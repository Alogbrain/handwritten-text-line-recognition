import cv2
import numpy as np
import math

def lineSegmentation(img, sigmaY):
    ''' line segmentation '''
    img = 255 - img
    Py = np.sum(img, axis=1)

    y = np.arange(img.shape[0])
    expTerm = np.exp(-y**2 / (2*sigmaY**2))
    yTerm = 1 / (np.sqrt(2*np.pi) * sigmaY)
    Gy = yTerm * expTerm

    Py_derivative = np.convolve(Py, Gy)
    thres = np.max(Py_derivative) // 2
    # find local maximum
    res = (np.diff(np.sign(np.diff(Py_derivative))) < 0).nonzero()[0] + 1

    lines = []
    for idx in res:
        if Py_derivative[idx] >= thres:
            lines.append(idx)
    return lines


def createKernel(kernelSize, sigma, theta):
	"""create anisotropic filter kernel according to given parameters"""
	assert kernelSize % 2 # must be odd size
	halfSize = kernelSize // 2
	
	kernel = np.zeros([kernelSize, kernelSize])
	sigmaX = sigma
	sigmaY = sigma * theta
	
	for i in range(kernelSize):
		for j in range(kernelSize):
			x = i - halfSize
			y = j - halfSize
			
			expTerm = np.exp(-x**2 / (2 * sigmaX) - y**2 / (2 * sigmaY))
			xTerm = (x**2 - sigmaX**2) / (2 * math.pi * sigmaX**5 * sigmaY)
			yTerm = (y**2 - sigmaY**2) / (2 * math.pi * sigmaY**5 * sigmaX)
			
			kernel[i, j] = (xTerm + yTerm) * expTerm

	kernel = kernel / np.sum(kernel)
	return kernel

def getCurrentLine(yMin, yMax, lineAvg):
    currLine =0
    
    for line in lineAvg:
        if yMin<line and yMax>line:
            return currLine
        currLine+=1
    return currLine
def getAvgLine(yMin, yMax,lines):
    lineAvg =[]
    currentLine =0
    maxLine = len(lines)
    i=0
    while i <maxLine-1:
        lineAvg.append((lines[i]+lines[i+1])/2)
        i+=1
    avg = (yMax+yMin)/2
    if avg<=lineAvg[0]:
        return 0
    else:
        j=1
        while j< len(lineAvg):
            if avg>lineAvg[j-1] and avg<=lineAvg[j]:
                return j
            j+=1
        return j

def wordSegmentation(img, kernelSize, sigma, theta, minArea=0):
    ''' word segmentation '''
    sigma_X = sigma
    sigma_Y = sigma * theta
    # use gaussian blur and applies threshold
    imgFiltered = cv2.GaussianBlur(img, (kernelSize, kernelSize), sigmaX=sigma_X, sigmaY=sigma_Y)
    #kernel = createKernel(kernelSize, sigma, theta)
    #imgFiltered = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REPLICATE).astype(np.uint8)
    _, imgThres = cv2.threshold(imgFiltered, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    imgThres = 255 - imgThres
    # find connected components
    _,components, _ = cv2.findContours(imgThres, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    lines = lineSegmentation(img, sigma)

    items = []
    # for c in components:
    #     # skip small word candidates
    #     if cv2.contourArea(c) < minArea:
    #         continue
    #     # append bounding box and image of word to items list
    #     currBox = cv2.boundingRect(c)
    #     (x, y, w, h) = currBox
    #     currImg = img[y:y+h, x:x+w]
    #     items.append([currBox, currImg])
    rects = []
    rectsUsed = []
    for c in components:
        rectsUsed.append(False)
        rects.append(cv2.boundingRect(c))
    rects.sort(key = getXFromRect)
    acceptedRects = []
    xThr = 5
    idx1=0
    while idx1<len(rects):
        if (rectsUsed[idx1] == False):
            currRect = rects[idx1]
            currxMin = currRect[0] 
            currxMax = currRect[0] + currRect[2]
            curryMin = currRect[1]
            curryMax = currRect[1] + currRect[3]
            maxsize = curryMax- curryMin
            print("++++++++++",idx1)
            print("curr: ", currxMin, currxMax, curryMin, curryMax)
            for idx2 in range(idx1+1, len(rects)-1):
                candRect = rects[idx2]
                candxMin = candRect[0]
                candxMax = candRect[0] + candRect[2]
                candyMin = candRect[1]
                candyMax = candRect[1] + candRect[3]
                if candxMin<=currxMax :
                    if getAvgLine(candyMin,candyMax,lines)==getAvgLine(curryMin, curryMax, lines):
                        print("**************")
                        print("candyMIN: ", idx2, candxMin, candxMax,candyMin, candyMax)
                        curryMin = min(curryMin, candyMin)
                        curryMax = max(curryMax, candyMax)
                        #rectsUsed[idx2] = True
                        rects.remove(rects[idx2])
                    # if candyMin<currxMin &(candyMax-candyMin)<maxsize/3:
                    #     curryMin = min(curryMin, candyMin)
                    #     curryMax = max(curryMax, candyMax)
                    # if (candyMax-candyMin)<maxsize:
                    #     print("candyMIN: ", idx2, candxMin, candxMax,candyMin, candyMax)
                    #     miny = min(curryMin, candyMin)
                    #     maxy= max(curryMax, candyMax)
                    #     # if (maxy-miny)> maxsize*1.5:
                    #     #     curryMin = miny
                    #     #     curryMax = maxy
                    #     #     break
                    # if candyMin>=currxMin & candyMax<currxMax:
                    #     curryMin = min(curryMin, candyMin)
                    #     curryMax = max(curryMax, candyMax)
                
                else:
                    break
                idx2+=1
        acceptedRects.append([currxMin, curryMin, currxMax - currxMin, curryMax - curryMin])
        (x, y, w, h) = (currxMin, curryMin, currxMax - currxMin, curryMax - curryMin)
        currImg = img[y:y+h, x:x+w]
        items.append([(x, y, w, h), currImg])
        idx1+=1
        #if idx1 >5:
        #    break
        # print("index1: ", idx1)
    image = None
    for rect in acceptedRects:
        image = cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (121, 11, 189), 2)
    result = []
    for line in lines:
        print("****LINE: ", line)
        temp = []
        for currBox, currImg in items:
            if currBox[1] < line:
                temp.append([currBox, currImg])
        for element in temp:
            items.remove(element)
        # list of words, sorted by x-coordinate
        result.append(sorted(temp, key=lambda entry: entry[0][0]))
    return result

def getXFromRect(item):
    return item[0]

def prepareImg(img, height):
    ''' convert given image to grayscale image (if needed) and resize to desired height '''
    assert img.ndim in (2, 3)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h = img.shape[0]
    #factor = height / h
    #print("######################FACTOR: ", factor, height, h)
    factor = 0.5
    #factor =1
    return cv2.resize(img, dsize=None, fx=factor, fy=factor)