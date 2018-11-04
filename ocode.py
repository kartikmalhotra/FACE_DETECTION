from __future__ import division
import dlib
import cv2
import openface
import numpy as np
import face_recognition

def resize(img, width=None, height=None, interpolation=cv2.INTER_AREA):
    global ratio
    w, h = img.shape

    if width is None and height is None:
        return img
    elif width is None:
        ratio = height / h
        width = int(w * ratio)
        resized = cv2.resize(img, (height, width), interpolation)
        return resized
    else:
        ratio = width / w
        height = int(h * ratio)
        resized = cv2.resize(img, (height, width), interpolation)
        return resized

def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords

camera = cv2.VideoCapture(0)

predictor_path = 'F://shape_predictor_68_face_landmarks.dat'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
face_aligner = openface.AlignDlib(predictor_path)
win = dlib.image_window()

while True:

    ret, frame = camera.read()
    if ret == False:
        print('Failed to capture frame from camera. Check camera index in cv2.VideoCapture(0) \n')
        break

    frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    frame_resized = resize(frame_grey, width=120)

    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    
    dets = detector(frame_resized, 1)

    print("Found {} faces in image".format(len(dets)))

    if len(dets) > 0:
        for qk, d in enumerate(dets):
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy array
            print("- Face {} found at Left: {} Top: {} Right: {} Bottom: {}".format(len(dets), d.left(), d.top(), d.right(), d.bottom()))
            shape = predictor(frame_resized, d)
            
            shape = shape_to_np(shape)
            print (shape)
            r=0

            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw them on the image
            r=0
            #win.set_image(frame)
            for (x, y) in shape:
                
               #cv2.circle(frame, (int(x/ratio), int(y/ratio)), 3, (255, 255, 255), -1)
               cv2.rectangle(frame, (int(d.left()/ratio), int(d.top()/ratio)),(int(d.right()/ratio), int(d.bottom()/ratio)), (0, 255, 0), 1)

            
            encodings = face_recognition.face_encodings(frame_rgb, shape)
            print(encodings)
            alignedFace = face_aligner.align(534, frame_grey, d, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
            cv2.imwrite("alignedface{}.jpg".format(r),alignedFace)
            r +=1
            cv2.imshow("image", frame)
            
                        
                
    if cv2.waitKey(1) & 0xFF == ord('q'):
        camera.release()
        cv2.destroyAllWindows()
        break
