import numpy as np
import cv2
import cv2.aruco as aruco
import sys
import time
import math

marker_size = 100


# CHeck if a matrix is a valid rotation matrix
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 12-6
# Calculates rotation matrix to euler angles
# The result is the same as Matlab except the order
# of the euler angles (x and z are sqapped)

def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))
    
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0,0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z])


calib_path=""
camera_matrix =np.loadtxt(calib_path +'cameraMatrix.txt', delimiter=',')
camera_distortion =np.loadtxt(calib_path +'cameraDistortion.txt', delimiter=',')

#--- 180 deg rotation matrix around the x axis
R_flip = np.zeros((3,3), dtype=np.float32)
R_flip[0,0] =1.0
R_flip[1,1] =1.0
R_flip[2,2] =1.0


aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)

cap = cv2.VideoCapture(-1)

camera_width = 1280
camera_height = 720
camera_frame_rate = 60

cap.set(2, camera_width)
cap.set(4, camera_height)
cap.set(5, camera_frame_rate)
#--Set the camera size as the ONE it was calibrated with
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

#-- FOnt for the text in the image
font = cv2.FONT_HERSHEY_PLAIN

while True: 
    ret, frame = cap.read()

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #-- Find all the aruco markers in the image
    corners, ids, rejected = aruco.detectMarkers(gray_frame, aruco_dict, camera_matrix, camera_distortion)


    if ids is not None:
        aruco.drawDetectedMarkers(frame, corners)
    
        rvec, tvec, _objPoints = aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, camera_distortion)
        ret2 = aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, camera_distortion)


        #-- Unpack the output, get only the first
        #rvec = ret[0][0,0,:]
        #tvec = ret[1][0,0,:]
        rvec2, tvec2= ret2[0][0,0,:], ret2[1][0,0,:]


        #--Print thw tag position in camera frame
        str_position = "MARKER POSITION x=%4.0f y=%4.0f z=%4.0f"  %(tvec2[0], tvec2[1], tvec2[2])
        cv2.putText(frame, str_position, (0,100), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        #--OBTAIN THE ROTATION matrix in camera
        R_ct  = np.matrix(cv2.Rodrigues(rvec2)[0])
        R_tc  = R_ct.T
        
        # Get the attitrude in terms of euler 321
        roll_marker, pitch_marker, yaw_marker = rotationMatrixToEulerAngles(R_flip*R_tc)
        
        #--Print the marker's attitude respect to camera frame
        str_attitude = "MARKER Attitude r=%4.0f p=%4.0f y=%4.0f"%(math.degrees(roll_marker),math.degrees(pitch_marker),
                                                                  math.degrees(yaw_marker))
        cv2.putText(frame, str_attitude, (0,250), font, 1, (0,255,0), 2, cv2.LINE_AA)        

        #-- Now get Position and attitude f the camera respect to the marker
        pos_camera = -R_tc*np.matrix(tvec2).T

        str_position = "CAMERA Position x=%4.0f  y=%4.0f  z=%4.0f"%(pos_camera[0], pos_camera[1], pos_camera[2])
        cv2.putText(frame, str_position, (0, 200), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

        #-- Get the attitude of the camera respect to the frame
        roll_camera, pitch_camera, yaw_camera = rotationMatrixToEulerAngles(R_flip*R_tc)
        str_attitude = "CAMERA Attitude r=%4.0f  p=%4.0f  y=%4.0f"%(math.degrees(roll_camera),math.degrees(pitch_camera),
                            math.degrees(yaw_camera))
        cv2.putText(frame, str_attitude, (0, 250), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
   



        for marker in range(len(ids)):
            aruco.drawAxis(frame, camera_matrix, camera_distortion, rvec[marker], tvec[marker], 100)
            cv2.putText(frame, str(ids[marker][0]), ( int(corners[marker][0][0][0]) -30, int(corners[marker][0][0][1])), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 2, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    #-- use 'q' to quit
    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()




#----------------
#str_position = "MARKER POSITION x=%4.0f y=%4.0f z=%4.0f"  %(tvec[0], tvec[1], tvec[2])
#cv2.putText(frame, str_position, (0,100), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
#-----------------
#cv2.waitKey(0)

#-- use 'q' to quit
#key = cv2.waitKey(10) & 0xFF
#if key == ord('q'):
#    cap.release()
#    cv2.destroyAllWindows()
#    break

#cap.release()
#cv2.destroyAllWindows()
