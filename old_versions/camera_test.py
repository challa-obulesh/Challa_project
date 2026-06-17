import cv2

for i in range(6):

    print(f"Testing camera index {i}")

    cap = cv2.VideoCapture(i)

    if not cap.isOpened():
        print(f"Index {i} failed to open")
        continue

    ret, frame = cap.read()

    if ret:
        print(f"Camera index {i} works")

        cv2.imshow(f"Camera {i}", frame)
        cv2.waitKey(3000)

    else:
        print(f"Index {i} opened but no frame")

    cap.release()

cv2.destroyAllWindows()
