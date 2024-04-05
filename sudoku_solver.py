from skimage.segmentation import clear_border
import numpy as np
import pyautogui
import imutils
import cv2
######
from mnist_model import load_pretrained_model
from sudoku import Sudoku

## preprocess the image and find contours
def preprocess_image(img, show=False):
	bfilter = cv2.bilateralFilter(img, 13, 20, 20)
	edged = cv2.Canny(bfilter, 30, 180)
	keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE,
	cv2.CHAIN_APPROX_SIMPLE)
	contours = imutils.grab_contours(keypoints)
	contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]
	if show:
		newimg = cv2.drawContours(img.copy(), contours, -1, (0, 255, 0), 3)
		cv2.imshow("Preprocessed", newimg)
	return contours

# With the image and contours, it finds the quadrilateral grid
def grid_location(contours):
	location = None
	# Finds rectangular contour
	for contour in contours:
		approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
		if len(approx) == 4:
			distances = [np.linalg.norm(approx[i] - approx[(i + 1) % len(approx)]) for i in range(len(approx))]
			#distances_all.append(distances)
			if all_numbers_approx_equal(distances):
				location = approx
				return location
	return None

def all_numbers_approx_equal(arr, error=0.01):
    # Check if all numbers in the array are approximately equal within the specified error
    return np.allclose(arr, arr[0], atol=error)

# Crop the grid from the image
def crop_grid(img, location):
	x, y, w, h = cv2.boundingRect(location)
	return img[y:y+h, x:x+w]

# Split the grid into 81 cells
def split_cells(img):
	grid = clear_border(img)
	grid = 255 - grid
	height, _ = grid.shape
	square_size = height // 9
	squares = []
	for i in range(9):
		for j in range(9):
			square_img = grid[i*square_size:(i+1)*square_size, j*square_size:(j+1)*square_size]
			square_img = cv2.resize(square_img, (28, 28))
			squares.append(square_img)
	return squares

# prepare model input
def prepare_model_input(grid):
	model_input = np.array(grid)
	model_input = model_input / 255
	model_input = np.expand_dims(model_input, -1)
	return model_input
#####

if __name__ == "__main__":
	# Take a screenshot of the screen and find the sudoku
	pyautogui.hotkey("command", "tab", interval=0.25)
	img = pyautogui.screenshot()
	# DEBUG IMG: img = cv2.imread("/Users/cgonzalez/Downloads/test.png")
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	contours = preprocess_image(np.array(img))
	grid = grid_location(contours)
	if grid is None:
		raise ValueError("No Sudoku/grid found")

	grid = crop_grid(np.array(img), grid)
	cells_img = prepare_model_input(split_cells(grid))
	# Load the pretrained model
	model = load_pretrained_model()
	# Predict the digits
	sudoku_pred = model.predict(cells_img)
	sudoku_nums = np.argmax(sudoku_pred, axis=1)
	extracted_elements = np.max(sudoku_pred, axis=1)

	sudoku_board = np.where(extracted_elements < 0.5, 0, sudoku_nums)
	sudoku_board = sudoku_board.reshape(9, 9)
	# Solve the sudoku
	sudoku = Sudoku(sudoku_board)
	solved_sudoku = sudoku.solve()
	print(solved_sudoku)
