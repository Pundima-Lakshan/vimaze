import cv2
import numpy as np

clicked_points = []

def get_corners(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
        cv2.circle(img, (x, y), 10, (0 , 0, 255), -1)
        cv2.imshow('image', img)
        clicked_points.append([x, y])


if __name__ == "__main__":
    image_path = "demo.jpeg"
    img = cv2.imread(image_path)
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', get_corners)
    cv2.waitKey(0)
    if len(clicked_points) != 4:
        print("Please click 4 points")
        exit    
    
    rect_points = np.array(clicked_points, dtype="float32")
    s = rect_points.sum(axis=1)
    diff = np.diff(rect_points, axis=1)
    tl = rect_points[np.argmin(s)]
    tr = rect_points[np.argmin(diff)]
    br = rect_points[np.argmax(s)]
    bl = rect_points[np.argmax(diff)]

    (tl_x, tl_y) = tl #tl
    (tr_x, tr_y) = tr #tr
    (br_x, br_y) = br #br
    (bl_x, bl_y) = bl #bl
    
    width_a = np.sqrt(((br_x - bl_x) ** 2) + ((br_y - bl_y) ** 2))
    width_b = np.sqrt(((tr_x - tl_x) ** 2) + ((tr_y - tl_y) ** 2))
    max_width = max(int(width_a), int(width_b))

    height_a = np.sqrt(((tr_x - br_x) ** 2) + ((tr_y - br_y) ** 2))
    height_b = np.sqrt(((tl_x - bl_x) ** 2) + ((tl_y - bl_y) ** 2))
    max_height = max(int(height_a), int(height_b))

    dst = np.array([[0, 0], 
                [max_width , 0], 
                [max_width , max_height ], 
                [0, max_height ]], dtype="float32")


    M = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl], dtype="float32"), dst)

    tilted_maze = cv2.warpPerspective(img, M, (max_width, max_height))
    gray_tilted = cv2.cvtColor(tilted_maze, cv2.COLOR_BGR2GRAY)
    guassian_thresh = cv2.adaptiveThreshold(gray_tilted, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, 20)
    _, binary_tilted = cv2.threshold(guassian_thresh, 128, 255, cv2.THRESH_BINARY)

    edges = cv2.Canny(guassian_thresh, 20, 150)
    enhance_edge = cv2.bitwise_or(binary_tilted, edges)

    sharp_filter = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    sharp_image = cv2.filter2D(enhance_edge, -1, sharp_filter)
  

    _, binary_sharp = cv2.threshold(sharp_image, 128, 255, cv2.THRESH_BINARY)

    noise_removed = cv2.medianBlur(binary_sharp, 3)

    sharp_final = np.array([[-1,-1,-1], [-1,9,-1],[-1,-1,-1]])
    final_image = cv2.filter2D(noise_removed, -1, sharp_final)

    kernel = np.ones((3,3), np.uint8) 
    inverted = cv2.bitwise_not(final_image)
    thickened_walls = cv2.dilate(inverted, kernel, iterations=2)
    inverted2 = cv2.bitwise_not(thickened_walls)
    cv2.imshow('binary2', inverted2)
    cv2.imwrite('binary2.jpg', inverted2)
    cv2.waitKey(0)

    cv2.destroyAllWindows()
