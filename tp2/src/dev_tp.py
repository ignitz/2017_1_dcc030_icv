'''
OReilly.Programming.Computer.Vision.with.Python.Jun.2012.RETAIL.eBook-ELOHiM.pdf
StackOverflow
'''

# for Augmented Reality
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import pygame, pygame.image
from pygame.locals import *

import pickle

# Marble Cake also the game
import cv2
import numpy as np

# pprint for dict
from pprint import *

# get saved intrinsic param
import yaml

import glob

'''
This code of this class I got from there
http://www.morethantechnical.com/2016/03/02/opencv-python-yaml-persistance/
'''
class YAML:
    # TODO try this after
    # Load previously saved data
    # with np.load('B.npz') as X:
    #     mtx, dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]
    def __init__(self, **kwargs):
        self.data = None
        self.file_name = kwargs.get('name', 'intrinsic.yml')
        yaml.add_constructor(u"tag:yaml.org,2002:opencv-matrix", self.opencv_matrix_constructor)
        yaml.add_representer(np.ndarray, self.opencv_matrix_representer)
        self.open_file()

    # A yaml constructor is for loading from a yaml node.
    # This is taken from: http://stackoverflow.com/a/15942429
    def opencv_matrix_constructor(self, loader, node):
        mapping = loader.construct_mapping(node, deep=True)
        mat = np.array(mapping["data"])
        mat.resize(mapping["rows"], mapping["cols"])
        return mat
     
    # A yaml representer is for dumping structs into a yaml node.
    # So for an opencv_matrix type (to be compatible with c++'s FileStorage) we save the rows, cols, type and flattened-data
    def opencv_matrix_representer(self, dumper, mat):
        mapping = {'rows': mat.shape[0], 'cols': mat.shape[1], 'dt': 'd', 'data': mat.reshape(-1).tolist()}
        return dumper.represent_mapping(u"tag:yaml.org,2002:opencv-matrix", mapping)

    def open_file(self):
        try:
            with open(self.file_name, 'r') as f:
                # read header
                if f.readline()[:5] != '%YAML':
                    print('Not YAML file')
                    return None
                self.data = yaml.load(f)
        except:
            print('Failed to open file ', )

    def get_data(self):
        return self.data

    def get_camera_matrix(self):
        return self.data['camera_matrix']

    def get_distortion_coefficients(self):
        return self.data['distortion_coefficients']

    def get_image_size(self):
        return self.data['image_width'], self.data['image_height']

    def show(self):
        pprint(self.data)

# Corner Harris
def get_corners(frame):
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners = cv2.cornerHarris(gray, 2, 3, 0.04)
    corners = cv2.dilate(corners,None)
    return corners

# # solvePnP
# def draw_cube(mtx, dist):
#     quad_3d = np.float32([[x0, y0, 0], [x1, y0, 0], [x1, y1, 0], [x0, y1, 0]])
#     ret, rvec, tvec = cv2.solvePnP(quad_3d, tracked.quad, mtx, dist)
#     return ret, rvec, tvec

# get corners and undistort video
def test_video1():
    camera = YAML()
    mtx = camera.get_camera_matrix()
    dist = camera.get_distortion_coefficients()

    cap = cv2.VideoCapture('entrada.avi')

    # Frames per second, Frame-rate
    fps = cap.get(cv2.CAP_PROP_FPS)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        h, w = frame.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            mtx,
            dist,
            (w,h), 1, (w,h)
        )

        # dst = cv2.undistort(frame, camera.get_camera_matrix(), dist, None, newcameramtx)
        # undistort
        mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
        dst = cv2.remap(frame,mapx,mapy,cv2.INTER_LINEAR)

        # crop the image
        x,y,w,h = roi
        dst = dst[y:y+h, x:x+w]

        # show corners in red on frame
        corners = get_corners(dst)
        dst[corners>0.01*corners.max()] = [0,0,255]

        # Display the resulting frame
        cv2.imshow('corners', dst)
        
        # Pay attention if Numlock actived
        # ord('q') maybe doesn't work
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    cv2.imshow('corners', dst) # XGH for close this window

# Test Canny Edge Detection in OpenCV
def test_video2():
    camera = YAML()
    mtx = camera.get_camera_matrix()
    dist = camera.get_distortion_coefficients()

    cap = cv2.VideoCapture('entrada.avi')

    # Frames per second, Frame-rate
    fps = cap.get(cv2.CAP_PROP_FPS)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        h, w = frame.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            mtx,
            dist,
            (w,h), 1, (w,h))

        # dst = cv2.undistort(frame, camera.get_camera_matrix(), dist, None, newcameramtx)
        # undistort
        mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h), 5)
        dst = cv2.remap(frame,mapx,mapy,cv2.INTER_LINEAR)

        # crop the image
        x,y,w,h = roi
        dst = dst[y:y+h, x:x+w]

        edges = cv2.Canny(dst,100,200)


        # Display the resulting frame
        cv2.imshow('edges', edges)        
        
        # Pay attention if Numlock actived
        # ord('q') maybe doesn't work
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    cv2.imshow('edges', edges) # XGH for close this window

# Draw XYZ canonical bases
def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

def cube_points(c,wid):
    """ Creates a list of points for plotting
    a cube with plot. (the first 5 points are
    the bottom square, some sides repeated). """
    p = []
    # bottom
    p.append([c[0]-wid,c[1]-wid,c[2]-wid])
    p.append([c[0]-wid,c[1]+wid,c[2]-wid])
    p.append([c[0]+wid,c[1]+wid,c[2]-wid])
    p.append([c[0]+wid,c[1]-wid,c[2]-wid])
    p.append([c[0]-wid,c[1]-wid,c[2]-wid]) # same as first to close plot
    # top
    p.append([c[0]-wid,c[1]-wid,c[2]+wid])
    p.append([c[0]-wid,c[1]+wid,c[2]+wid])
    p.append([c[0]+wid,c[1]+wid,c[2]+wid])
    p.append([c[0]+wid,c[1]-wid,c[2]+wid])
    p.append([c[0]-wid,c[1]-wid,c[2]+wid]) # same as first to close plot
    # vertical sides
    p.append([c[0]-wid,c[1]-wid,c[2]+wid])
    p.append([c[0]-wid,c[1]+wid,c[2]+wid])
    p.append([c[0]-wid,c[1]+wid,c[2]-wid])
    p.append([c[0]+wid,c[1]+wid,c[2]-wid])
    p.append([c[0]+wid,c[1]+wid,c[2]+wid])
    p.append([c[0]+wid,c[1]-wid,c[2]+wid])
    p.append([c[0]+wid,c[1]-wid,c[2]-wid])
    return np.array(p).T


# test with contours
def test_video3():
    camera = YAML()
    mtx = camera.get_camera_matrix()
    dist = camera.get_distortion_coefficients()

    cap = cv2.VideoCapture('entrada.avi')

    # Frames per second, Frame-rate
    fps = cap.get(cv2.CAP_PROP_FPS)

    ## aushuihausihuiahuish
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((6*7,3), np.float32)
    objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
    axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        h, w = frame.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            mtx,
            dist,
            (w,h), 1, (w,h))

        # dst = cv2.undistort(frame, camera.get_camera_matrix(), dist, None, newcameramtx)
        # undistort
        mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
        dst = cv2.remap(frame,mapx,mapy,cv2.INTER_LINEAR)

        # crop the image
        x,y,w,h = roi
        dst = dst[y:y+h, x:x+w]

        gray = cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5,5), 0)
        edges = cv2.Canny(gray, 100, 200)
        # cv2.imshow('edges', edges)
        im2, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        cv2.drawContours(dst, contours, -1, (0,255,0), 3)

        
        cv2.imshow('im2', dst)
        
        # Pay attention if Numlock actived
        # ord('q') maybe doesn't work
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    cv2.imshow('quad', edges) # XGH for close this window


# OPENGL STUFFs

# width, height = 2592 / 2, 1944 / 2
width, height = 640, 480

def set_projection_from_camera(K):
  glMatrixMode(GL_PROJECTION)
  glLoadIdentity()

  fx = float(K[0, 0])
  fy = float(K[1, 1])
  fovy = 2 * np.arctan(0.5 * height / fy) * 180 / np.pi
  aspect = (width * fy) / (height * fx)

  near, far = 0.1, 100
  gluPerspective(fovy, aspect, near, far)
  glViewport(0, 0, width, height)


def set_modelview_from_camera(Rt):
  glMatrixMode(GL_MODELVIEW)
  glLoadIdentity()

  # Rotate 90 deg around x, so that z is up.
  Rx = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])

  # Remove noise from rotation, make sure it's a pure rotation.
  R = Rt[:, :3]
  U, S, V = np.linalg.svd(R)
  R = np.dot(U, V)
  R[0, :] = -R[0, :]  # Change sign of x axis.

  print S
  t = Rt[:, 3]

  M = np.eye(4)
  M[:3, :3] = np.dot(R, Rx)
  M[:3, 3] = t

  m = M.T.flatten()
  glLoadMatrixf(m)


def draw_background(imname):
  bg_image = pygame.image.load(imname).convert()
  width, height = bg_image.get_size()
  bg_data = pygame.image.tostring(bg_image, "RGBX", 1)

  glEnable(GL_TEXTURE_2D)
  tex = glGenTextures(1)
  glBindTexture(GL_TEXTURE_2D, tex)
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0,
               GL_RGBA, GL_UNSIGNED_BYTE, bg_data)
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

  glBegin(GL_QUADS)
  glTexCoord2f(0, 0); glVertex3f(-1, -1, -1)
  glTexCoord2f(1, 0); glVertex3f( 1, -1, -1)
  glTexCoord2f(1, 1); glVertex3f( 1,  1, -1)
  glTexCoord2f(0, 1); glVertex3f(-1,  1, -1)
  glEnd()

  glDeleteTextures(tex)


def load_and_draw_model(filename):
  glEnable(GL_LIGHTING)
  glEnable(GL_LIGHT0)
  glEnable(GL_DEPTH_TEST)
  glClear(GL_DEPTH_BUFFER_BIT)
  glMaterialfv(GL_FRONT, GL_AMBIENT, [0, 0, 0, 0])
  glMaterialfv(GL_FRONT, GL_DIFFUSE, [0.5, 0.75, 1, 0])
  glMaterialfv(GL_FRONT, GL_SHININESS, 0.25 * 128)
  import objloader
  obj = objloader.OBJ(filename, swapyz=True)
  glScale(0.1, 0.1, 0.1)
  glCallList(obj.gl_list)


def setup():
  pygame.init()
  pygame.display.set_mode((width, height), OPENGL | DOUBLEBUF)
  pygame.display.set_caption('Look, an OpenGL window!')


def test_OPENGL():
    with open('ar_camera.pkl', 'rb') as f:
      K = pickle.load(f)
      Rt = pickle.load(f)

    glutInit()
    setup()
    draw_background('book_perspective.bmp')

    # FIXME: The origin ends up in a different place than in ch04_markerpose.py
    # somehow.
    set_projection_from_camera(K)
    set_modelview_from_camera(Rt)

    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_DEPTH_TEST)
    glClear(GL_DEPTH_BUFFER_BIT)
    glMaterialfv(GL_FRONT, GL_AMBIENT, [0, 0, 0, 0])
    glMaterialfv(GL_FRONT, GL_DIFFUSE, [0.5, 0, 0, 0])
    glMaterialfv(GL_FRONT, GL_SHININESS, 0.25 * 128)
    for y in range(0, 1):
      for x in range(0, 1):
        glutSolidTeapot(0.02)
        glTranslatef(0.04, 0, 0)
      glTranslatef(-3 * 0.04, 0, 0.04)
    #load_and_draw_model('out_toyplane.obj')
    pygame.display.flip()

    while True:
      event = pygame.event.poll()
      if event.type in (QUIT, KEYDOWN):
        break
      #pygame.display.flip()


def marbleCake():
    camera = YAML()
    mtx = camera.get_camera_matrix()
    dist = camera.get_distortion_coefficients()

    cap = cv2.VideoCapture('entrada.avi')

    # Frames per second, Frame-rate
    fps = cap.get(cv2.CAP_PROP_FPS)

    ## aushuihausihuiahuish
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((6*7,3), np.float32)
    objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
    axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        h, w = frame.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            mtx,
            dist,
            (w,h), 1, (w,h))

        # dst = cv2.undistort(frame, camera.get_camera_matrix(), dist, None, newcameramtx)
        # undistort
        mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
        dst = cv2.remap(frame,mapx,mapy,cv2.INTER_LINEAR)

        # crop the image
        x,y,w,h = roi
        dst = dst[y:y+h, x:x+w]

        gray = cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5,5), 0)
        edges = cv2.Canny(gray, 100, 200)
        # cv2.imshow('edges', edges)
        im2, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        cv2.drawContours(dst, contours, -1, (0,255,0), 3)

        
        cv2.imshow('im2', dst)
        
        # Pay attention if Numlock actived
        # ord('q') maybe doesn't work
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    cv2.imshow('quad', edges) # XGH for close this window


    MIN_MATCH_COUNT = 10

    img1 = cv2.imread('alvo.jpg',0)          # queryImage
    img2 = cv2.imread('export/samples029.png',0) # trainImage

    # Initiate SIFT detector
    # sift = cv2.xfeatures2d.SIFT_create()
    sift = cv2.ORB_create()

    # find the keypoints and descriptors with SIFT
    # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img1 = cv2.Canny(img1, 200, 200)
    img2 = cv2.Canny(img2, 200, 200)
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    # flann = cv2.FlannBasedMatcher(index_params, search_params)
    flann = cv2.BFMatcher()

    matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        h,w,d = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)

        img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

    else:
        print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
        matchesMask = None
        
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                       singlePointColor = None,
                       matchesMask = matchesMask, # draw only inliers
                       flags = 2)

    img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

    plt.imshow(img3, 'gray'),plt.show() 

# # for test
def main():
    # test_video1()
    # test_video2()
    # test_video3()
    # test_OPENGL()
    # marbleCake()
    

if __name__ == '__main__':
    main()

