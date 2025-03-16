import numpy as np
import cv2 as cv
import glob
import os
import matplotlib.pyplot as plt
from fontTools.misc.arrayTools import normRect


def calibrate(showPics=True):
    root = os.getcwd()
    calibrationDir = os.path.join(root, 'calibration')
    imgPathList = glob.glob(os.path.join(calibrationDir, '*.jpg'))

    nRows = 9
    nCols = 6

    termCriteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    worldPtsCur = np.zeros((nRows * nCols, 3), np.float32)
    worldPtsCur[:,:2] = np.mgrid[0:nCols, 0:nRows].T.reshape(-1, 2)
    worldPtsList = []
    imgPtsList = []

    for curImgPath in imgPathList:
        imgBGR = cv.imread(curImgPath)
        imgGray = cv.cvtColor(imgBGR, cv.COLOR_BGR2GRAY)
        cornersFound, cornersOrg = cv.findChessboardCorners(imgGray, (nCols, nRows), None)

        if cornersFound == True:
            worldPtsList.append(worldPtsCur)
            cornersRefined = cv.cornerSubPix(imgGray, cornersOrg, (11, 11), (-1, -1), termCriteria)
            imgPtsList.append(cornersRefined)
            if showPics:
                cv.drawChessboardCorners(imgBGR, (nRows, nCols), cornersRefined, cornersFound)
                cv.imshow('Chessboard', imgBGR)
                cv.waitKey(500)

    cv.destroyAllWindows()

    repError, camMatrix,distCoeff,rvecs,tvces = cv.calibrateCamera(worldPtsList, imgPtsList, imgGray.shape[::-1], None, None)
    print('Camera Matrix: ', camMatrix)
    print("Reproj Error (pixels): {:.4f}".format(repError))

    curFolder = os.path.dirname(os.path.abspath(__file__))
    paramPath = os.path.join(curFolder, 'calibration.npz')
    np.savey(paramPath,
             repError = repError,
             camMatrix = camMatrix,
             distCoeff = distCoeff,
             rvecs = rvecs,
             tvecs = tvces)

    return camMatrix, distCoeff

def removeDistortion(camMatrix, distCoeff):
    root = os.getcwd()
    imgPath = os.path.join(root, 'demoImages//distortion2.jpg')
    img = cv.imread(imgPath)
    height, width = img.shape[:2]
    camMatrixNew, roi = cv.getOptimalNewCameraMatrix(camMatrix, distCoeff, (width, height), 1, (width, height))
    imgUndist = cv.undistort(img, camMatrix, None, camMatrixNew)

    cv.line(img, (1769, 103), (1780, 922), (255, 255, 255),2)
    cv.line(imgUndist,(1769,103), (1780, 922), (255, 255, 255),2)

    plt.figure()
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(imgUndist)
    plt.show()

def runCalibration():
    camMatrix, distCoeff = calibrate(showPics=False)
    removeDistortion(camMatrix, distCoeff)

if __name__ == '__main__':
    runCalibration()






