import cv2
import numpy as np

image = cv2.imread('test6.jpg')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blurred = cv2.GaussianBlur(gray, (9, 9), 0)
edges = cv2.Canny(blurred, 50, 150)
# _, binary_img = cv2.threshold(blurred, 128,255, cv2.THRESH_BINARY_INV)
thresh = cv2.adaptiveThreshold(edges, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10)
_, otsu = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
combined = cv2.bitwise_or(thresh, otsu)

# final_mask = cv2.bitwise_or(combined, edges)
# kernel = np.ones((3,3), np.uint8)
# final_mask = cv2.dilate(final_mask, kernel, iterations=1)

contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.imshow('thresh', thresh)
cv2.waitKey(0)
if contours:
    largest_contour = max(contours, key=cv2.contourArea)
    # epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    # approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    x, y, w, h = cv2.boundingRect(largest_contour)
    # cropped_maze = image[y:y+h, x:x+w]
    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)
    box = box.astype(np.intp)

cv2.drawContours(image, [box], 0, (0, 255, 0), 2)
cv2.imshow('Detected Maze', image)
cv2.waitKey(0)
# x, y, w, h = cv2.boundingRect(largest_contour)
# cropped_maze = image[y:y+h, x:x+w]
rect_points = np.array(box, dtype="float32")
s = rect_points.sum(axis=1)
diff = np.diff(rect_points, axis=1)
tl = rect_points[np.argmin(s)]
tr = rect_points[np.argmin(diff)]
br = rect_points[np.argmax(s)]
bl = rect_points[np.argmax(diff)]

(tl_x, tl_y) = tl
(tr_x, tr_y) = tr
(br_x, br_y) = br
(bl_x, bl_y) = bl

width_a = np.sqrt(((br_x - bl_x) ** 2) + ((br_y - bl_y) ** 2))
width_b = np.sqrt(((tr_x - tl_x) ** 2) + ((tr_y - tl_y) ** 2))
max_width = max(int(width_a), int(width_b))

height_a = np.sqrt(((tr_x - br_x) ** 2) + ((tr_y - br_y) ** 2))
height_b = np.sqrt(((tl_x - bl_x) ** 2) + ((tl_y - bl_y) ** 2))
max_height = max(int(height_a), int(height_b))

dst = np.array([[0, 0],
                [max_width, 0],
                [max_width, max_height],
                [0, max_height]], dtype="float32")

dst2 = np.array([[0, max_height - 1],
                 [max_width - 1, max_height - 1],
                 [max_width - 1, 0],
                 [0, 0]], dtype="float32")

M = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl], dtype="float32"), dst)

tilted_maze = cv2.warpPerspective(image, M, (max_width, max_height))

# angle = rect[-1]
# logging.debug(angle)
# angle = angle - 50

# 39 correct angle for horizontal 
# if angle < -45:
#     angle = -(90 +angle) 
# else:
#     angle = -angle

# logging.debug("after fix: ", angle)

# if angle < -45:
#     angle = 90 + angle
# elif angle > 45:
#     angle = angle - 90
# else:
#     angle = -angle

# distrotion
# pts1 = np.float32(box)  # Detected bounding box points
# pts2 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])  # Desired destination points

# M = cv2.getPerspectiveTransform(pts1, pts2)  # Compute transformation matrix
# aligned_maze = cv2.warpPerspective(image, M, (w, h))

# (h , w) = image.shape[:2]
# center = (w // 2 , h // 2)
# M = cv2.getRotationMatrix2D(center, angle, 1.0)

# box_center = (rect[0][0], rect[0][1])  # Use the rectangle's center
# M = cv2.getRotationMatrix2D(box_center, angle, 1.0)  # Rotate around the maze center

# rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


cv2.imshow('tilted Maze', tilted_maze)
cv2.waitKey(0)

gray_tilted = cv2.cvtColor(tilted_maze, cv2.COLOR_BGR2GRAY)
guassian_thresh = cv2.adaptiveThreshold(gray_tilted, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 5)
# cv2.imshow('guassian_thresh', guassian_thresh)
_, binary_tilted = cv2.threshold(guassian_thresh, 128, 255, cv2.THRESH_BINARY)

grid = (binary_tilted / 255).astype(int)
grid = grid[2:-2, 2:-2]
filter_grid = grid[np.any(grid == 0, axis=1)]
filter_grid = filter_grid[:, np.any(filter_grid == 0, axis=0)]

convert_filter_image = (filter_grid * 255).astype(np.uint8)
cv2.imshow('final image', binary_tilted)
cv2.waitKey(0)

# sharp_filter = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
# sharp_image = cv2.filter2D(binary_tilted, -1, sharp_filter)
# cv2.imshow('sharp_image', sharp_image)
# cv2.waitKey(0)

edges = cv2.Canny(guassian_thresh, 50, 150)
enhance_edge = cv2.bitwise_or(binary_tilted, edges)
cv2.imshow('enhance_edge', enhance_edge)
cv2.waitKey(0)

sharp_filter = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
sharp_image = cv2.filter2D(enhance_edge, -1, sharp_filter)
cv2.imshow('sharp_image', sharp_image)
cv2.waitKey(0)

_, binary_sharp = cv2.threshold(sharp_image, 128, 255, cv2.THRESH_BINARY)
cv2.imwrite('binary_sharp.jpg', binary_sharp)
cv2.imshow('binary_sharp', binary_sharp)
cv2.waitKey(0)

cv2.destroyAllWindows()
