import cv2
import numpy as np
import os

class CornerDetectionUser:
    def __init__(self):
        self.max_width = 900
        self.max_height = 600

        self.clicked_points = []


    def wait_till_enter(self):
        while True:
            key = cv2.waitKey(1)  # Check for key press (wait for 1 ms)
            if key == 13:  # Enter key has ASCII value 13
                break  # Exit the loop when Enter is pressed

    def get_corners(self, event, x, y, flags, param):
        # Get the resized image and original image
        resized_img, original_img = param

        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Clicked coordinates on resized image: ({x}, {y})")

            # Map the resized image coordinates back to the original image coordinates
            original_x = int(x * (original_img.shape[1] / resized_img.shape[1]))
            original_y = int(y * (original_img.shape[0] / resized_img.shape[0]))

            print(f"Mapped coordinates on original image: ({original_x}, {original_y})")

            # Draw circle on the original image (to reflect clicked points)
            cv2.circle(original_img, (original_x, original_y), 10, (0, 0, 255), -1)

            # Update the resized image for display with the circle
            resized_display = self.resize_image(original_img, self.max_width, self.max_height)
            cv2.imshow("image", resized_display)

            self.clicked_points.append([original_x, original_y])

    def resize_image(self, image, max_width, max_height):
        # Get the original image dimensions
        h, w = image.shape[:2]

        # Calculate the scale factor
        scale_w = max_width / w
        scale_h = max_height / h

        # Choose the smaller scale factor to maintain aspect ratio
        scale = min(scale_w, scale_h)

        # Resize the image
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized_image = cv2.resize(image, (new_w, new_h))

        return resized_image

    def select_corners(self, image_path: str):
        img = cv2.imread(image_path)

        # Resize for display purposes with max width 600 and height 400
        resized_img = self.resize_image(img, self.max_width, self.max_height)

        # Show the resized image and pass both resized and original image to callback
        cv2.imshow("image", resized_img)
        cv2.setMouseCallback("image", self.get_corners, param=(resized_img, img))
        self.wait_till_enter()

        if len(self.clicked_points) != 4:
            print("Please click 4 points")
            exit()

        rect_points = np.array(self.clicked_points, dtype="float32")
        s = rect_points.sum(axis=1)
        diff = np.diff(rect_points, axis=1)
        tl = rect_points[np.argmin(s)]
        tr = rect_points[np.argmin(diff)]
        br = rect_points[np.argmax(s)]
        bl = rect_points[np.argmax(diff)]

        (tl_x, tl_y) = tl  # tl
        (tr_x, tr_y) = tr  # tr
        (br_x, br_y) = br  # br
        (bl_x, bl_y) = bl  # bl

        width_a = np.sqrt(((br_x - bl_x) ** 2) + ((br_y - bl_y) ** 2))
        width_b = np.sqrt(((tr_x - tl_x) ** 2) + ((tr_y - tl_y) ** 2))
        max_width = max(int(width_a), int(width_b))

        height_a = np.sqrt(((tr_x - br_x) ** 2) + ((tr_y - br_y) ** 2))
        height_b = np.sqrt(((tl_x - bl_x) ** 2) + ((tl_y - bl_y) ** 2))
        max_height = max(int(height_a), int(height_b))

        dst = np.array(
            [[0, 0], [max_width, 0], [max_width, max_height], [0, max_height]],
            dtype="float32",
        )

        M = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl], dtype="float32"), dst)

        tilted_maze = cv2.warpPerspective(img, M, (max_width, max_height))
        gray_tilted = cv2.cvtColor(tilted_maze, cv2.COLOR_BGR2GRAY)
        guassian_thresh = cv2.adaptiveThreshold(
            gray_tilted, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, 20
        )
        _, binary_tilted = cv2.threshold(guassian_thresh, 128, 255, cv2.THRESH_BINARY)

        edges = cv2.Canny(guassian_thresh, 20, 150)
        enhance_edge = cv2.bitwise_or(binary_tilted, edges)

        sharp_filter = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharp_image = cv2.filter2D(enhance_edge, -1, sharp_filter)

        _, binary_sharp = cv2.threshold(sharp_image, 128, 255, cv2.THRESH_BINARY)

        noise_removed = cv2.medianBlur(binary_sharp, 3)

        sharp_final = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        final_image = cv2.filter2D(noise_removed, -1, sharp_final)

        kernel = np.ones((3, 3), np.uint8)
        inverted = cv2.bitwise_not(final_image)
        thickened_walls = cv2.dilate(inverted, kernel, iterations=2)
        inverted2 = cv2.bitwise_not(thickened_walls)

        # Resize the binary image for display (max width 600, max height 400)
        resized_binary_image = self.resize_image(inverted2, max_width, max_height)

        # Display resized binary image
        cv2.imshow("binary2", resized_binary_image)

        # Get the directory and filename of the original image
        directory, original_filename = os.path.split(image_path)

        # Create a new filename for the binary image (adding _binary to the original name)
        name_without_extension, extension = os.path.splitext(original_filename)
        binary_filename = name_without_extension + '_binary' + extension

        # Construct the full path for saving the binary image
        binary_image_path = os.path.join(directory, binary_filename)

        # Save the binary image to the same directory with the new filename
        cv2.imwrite(binary_image_path, inverted2)

        print(f"Binary image saved at: {binary_image_path}")

        self.wait_till_enter()

        self.clicked_points = []

        cv2.destroyAllWindows()

        return binary_image_path
