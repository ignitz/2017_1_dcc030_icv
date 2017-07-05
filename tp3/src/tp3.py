# %matplotlib inline
import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
import time as t
print "OpenCV Version : %s " % cv2.__version__

class Image:
    """class Image for using in Jupyter with OpenCV 3.x.x"""
    name = str() # file name. ex: page1.jpg
    path_input = str() # input dir. ex: dados/
    path_output = str() # output data, if use save method
    color = str() # use 'black' or 'white' to detect points
    line_locations = list()
    img = None # image to do things
    img_original = None # original image
    img_output = None # output image with boxes

    def __init__(self, **kwargs):
        self.name = kwargs.get('name', 'input.jpg')
        self.path_input = kwargs.get('path_input', 'dados/')
        if len(self.path_input) > 0 and self.path_input[-1] != '/':
            self.path_input += '/'
        self.path_output = kwargs.get('path_output', 'export/')
        if len(self.path_output) > 0 and self.path_output[-1] != '/':
            self.path_output += '/'
        self.color = kwargs.get('color', 'black')

    def open_image(self): # Read image with opencv
        img_path = self.path_input + self.name
        self.img = cv2.imread(img_path)
        self.img_original = cv2.imread(img_path)
        if self.img is None:
            print('Arquivo nao carregado, verifique se o nome do arquivo esta correto')
            return False
        else:
            return True

    def threshold(self, div_value=180): # Convert to gray and make thresold of image
        # div_value: value between black and white
        img = self.img
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dummy, mask = cv2.threshold(img_gray, div_value, 255, cv2.THRESH_BINARY)
        img_bitwiseand = cv2.bitwise_and(img_gray, img_gray, mask=mask)
        if self.color == 'black':
            dummy, new_img = cv2.threshold(img_bitwiseand, div_value, 255, cv2.THRESH_BINARY_INV)
            # new_img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
        elif self.color == 'white':
            dummy, new_img = cv2.threshold(img_bitwiseand, div_value, 255, cv2.THRESH_BINARY)
        else:
            print('Set black or white in beginning of file')
            return None
        self.img = new_img
        return self.img

    def blur(self, dim_size_x = 13, dim_size_y = 11, num_iter = 1): # Remove noise and blur image
        # http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
        # TODO: Make a away to get a mean of space between chars
        
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (dim_size_x, dim_size_y))
        # array([[0, 0, 1, 0, 0],
        #    [0, 0, 1, 0, 0],
        #    [1, 1, 1, 1, 1],
        #    [0, 0, 1, 0, 0],
        #    [0, 0, 1, 0, 0]], dtype=uint8)
        
        self.img = cv2.dilate(self.img, kernel, iterations=num_iter)
        return self.img

    def draw_rectangles(self, w_max=15, h_max=15): # draw rectangles with boxes on detect pixels
        # for cv 3.x.x return 3 elements
        # cv 2.x.x return 2
        image, contours, hierarchy = cv2.findContours(self.img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

        self.img_output = self.img_original.copy()

        self.line_locations = list()
        self.line_locations.append('x\t\ty\t\tw\t\th\n')

        count = 1
        for contour in contours:
            line = [x, y, w, h] = cv2.boundingRect(contour)

            if w < w_max and h < h_max: # false points?
                continue

            string = ''
            for value in line:
                string += str(value)
                string += '\t\t' if value < 1000 else '\t'
            string += '\n'

            self.line_locations.append(string)

            # draw rectangle around contour on original image
            cv2.rectangle(self.img_output, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            count = count + 1
        
        print('Get ' + str(count - 1) + ' box of interest')
        return self.img_output

    def save(self): # save on file
        text_file = open(self.path_output + self.name + '.txt', 'w')
        if text_file is not None:
            for line in self.line_locations:
                text_file.write(line)
            text_file.close()
        return cv2.imwrite(self.path_output + self.name, self.img_output)

    def print_on_notebook(self, x=20, y=20, img=None):
    	# img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(x,y))
        plt.imshow(img)
        plt.show()
        
    def show_output(self, x=20, y=20):
        self.print_on_notebook(x=x, y=y, img=self.img_output)
        
    def show_img(self, x=20, y=20):
        self.print_on_notebook(x=x, y=y, img=self.img)
    
    def show_original_img(self, x=20, y=20):
        self.print_on_notebook(x=x, y=y, img=self.img_original)

def erode(img):
	dim_size_x, dim_size_y = (13, 11)
	kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (dim_size_x, dim_size_y))

	pprint.pprint(kernel)
	

def dilate(img):
	pass


def main():
	#  for output images
	export_path = './export/'
	if export_path[-1] != '/':
	    export_path += '/'

	# create export dir
	if not os.path.isdir(export_path):
	    os.mkdir(export_path)

	img = Image(name='sample.png')
	if not img.open_image():
	    raise Exception('failed to open file')
	img.threshold()
	img.show_img()
	img.blur()
	img.show_img()
	img.draw_rectangles()
	img.show_output()

def histogram():
	img = cv2.imread('dados/Lenna.png')
	hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
	hist = cv2.calcHist( [hsv], [0, 1], None, [180, 256], [0, 180, 0, 256] )
	plt.imshow(hist,interpolation = 'nearest')
	plt.show()

def equalize_gray(img):
    plt.imshow(img)
    plt.show()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.imshow(img)
    plt.show()
    equ = cv2.equalizeHist(img)
    res = np.hstack((img,equ)) #stacking images side-by-side
    plt.imshow(res)
    plt.show()

    # hist,bins = np.histogram(img.flatten(),256,[0,256])

    # cdf = hist.cumsum()
    # cdf_normalized = cdf * hist.max()/ cdf.max()

    # plt.plot(cdf_normalized, color = 'b')
    # plt.hist(img.flatten(),256,[0,256], color = 'r')
    # plt.xlim([0,256])
    # plt.legend(('cdf','histogram'), loc = 'upper left')
    # plt.show()
    # cdf_m = np.ma.masked_equal(cdf,0)
    # cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    # cdf = np.ma.filled(cdf_m,0).astype('uint8')
    # img2 = cdf[img]

def hisEqulColor(img):
	# hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
	# channels=cv2.split(hsv)
	# cv2.equalizeHist(channels[2],channels[2])
	# cv2.merge(channels, hsv)
	# cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR, img)
	# return img

	img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

	# equalize the histogram of the Y channel
	img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

	# convert the YUV image back to RGB format
	img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

	return img_output

    # ycrcb=cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
    # channels=cv2.split(ycrcb)
    # print len(channels)
    # cv2.equalizeHist(channels[0],channels[0])
    # cv2.merge(channels,ycrcb)
    # cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img)


if __name__ == '__main__':
	main()
	sys.exit(0)
	# histogram()
	img = cv2.imread('dados/sample.png')
	# plt.imshow(img)
	# plt.show()
	# img = hisEqulColor(img)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img = cv2.equalizeHist(img)
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	img = clahe.apply(img)
	for x in xrange(1,100):
		img = clahe.apply(img)
	
	
	img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
	# equalize_gray(img)
	# img = erode(img)
	# print(img)
	plt.imshow(img)
	plt.show()

	
	# plt.imshow()
	# plt.show()
	# plt.imshow(cv2.cvtColor(img2,cv2.COLOR_BGR2RGB))
	# plt.show()

# Equalizar
# Converter pra grayscale
# 