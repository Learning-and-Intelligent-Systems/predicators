# import the necessary packages
import apriltag
import argparse
import cv2
# initialize the camera
cam = cv2.VideoCapture(0)   # 0 -> index of camera

while True:
    s, img = cam.read()
    if s:    # frame captured without any errors
        cv2.namedWindow("cam-test")
        # cv2.imshow("cam-test",img)
        # cv2.waitKey(0)
        cv2.destroyWindow("cam-test")
        cv2.imwrite("image.jpg",img) #save image

    # load the input image and convert it to grayscale
    # print("[INFO] loading image...")
    image = cv2.imread("image.jpg")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # define the AprilTags detector options and then detect the AprilTags
    # in the input image
    # print("[INFO] detecting AprilTags...")
    options = apriltag.DetectorOptions(families="tag36h11")
    detector = apriltag.Detector(options)
    results = detector.detect(gray)
    # print("[INFO] {} total AprilTags detected".format(len(results)))

    camera_params = (
        329.52725219726562,
        328.67050170898438,
        312.82241821289062,
        239.58528137207031,
    )

    detections = results
    try:
        detection = [d for d in detections if d.tag_id == 401][0]
    except:
        print("[Out of View]")
        continue
    pose = detector.detection_pose(detection, camera_params)[0]
    tx, ty, tz, tw = pose[:, -1]
    print("Pose in inches:", 3*tx, -3*ty, 4.1*tz)

    # loop over the AprilTag detection results
    for r in results:
        # extract the bounding box (x, y)-coordinates for the AprilTag
        # and convert each of the (x, y)-coordinate pairs to integers
        (ptA, ptB, ptC, ptD) = r.corners
        ptB = (int(ptB[0]), int(ptB[1]))
        ptC = (int(ptC[0]), int(ptC[1]))
        ptD = (int(ptD[0]), int(ptD[1]))
        ptA = (int(ptA[0]), int(ptA[1]))
        # draw the bounding box of the AprilTag detection
        cv2.line(image, ptA, ptB, (0, 255, 0), 2)
        cv2.line(image, ptB, ptC, (0, 255, 0), 2)
        cv2.line(image, ptC, ptD, (0, 255, 0), 2)
        cv2.line(image, ptD, ptA, (0, 255, 0), 2)
        # draw the center (x, y)-coordinates of the AprilTag
        (cX, cY) = (int(r.center[0]), int(r.center[1]))
        cv2.circle(image, (cX, cY), 5, (0, 0, 255), -1)
        # draw the tag family on the image
        tagFamily = r.tag_family.decode("utf-8")
        cv2.putText(image, tagFamily, (ptA[0], ptA[1] - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # print("[INFO] tag family: {}".format(tagFamily))
    # show the output image after AprilTag detection
    cv2.imshow("Image", image)
    # cv2.waitKey(0)