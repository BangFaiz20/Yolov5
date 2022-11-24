import numpy as np
import cv2
import torch
import time
import serial
import joblib


model = torch.hub.load('./', 'custom', path='best.pt',source='local') # or yolov5m, yolov5l, yolov5x, custom

cap = cv2.VideoCapture('Pertama.mp4')

frame_rate= 3.5
prev = 0
prev_frame_time = 0
new_frame_time = 0

arduino = serial.Serial(port='/dev/ttyACM0', baudrate=9600, timeout=.1)       #'/dev/ttyUSB0'
'''
def write_read(x):
    time.sleep(0.05)
    data = arduino.readline()
    return data'''

while (cap.isOpened()):
    
    # Capture frame-by-frame
    
    ret,frame = cap.read()
    frame = cv2.resize(frame,(800,640),interpolation = cv2.INTER_AREA)
    if not ret:
        break
    #cv2.line(frame,(800,320),(0,320),(0,255,0),2)
    #cv2.line(frame,(800,400),(0,400),(0,255,0),2)
    gray = frame
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    fps =float(fps)
    print(abs(fps))
    time_elapsed = time.time() - prev
    #s1 = []
    #s2 = []
    if time_elapsed>1./frame_rate:
        detections = model(frame)
        results = detections.pandas().xyxy[0].to_dict(orient="records")
        for result in results:   
                    con = result['confidence']
                    cs = result['name']
                    x1 = int(result['xmin'])
                    y1 = int(result['ymin'])
                    x2 = int(result['xmax'])
                    y2 = int(result['ymax'])

                    wx =(x2+x1)//2
                    wy = (y2+y1)//2
                    cent = (wx,wy)
                    print(wy)
                    #cv2.line(frame,(800,380),(0,380),(0,255,0),1)
                    #cv2.line(frame,(800,400),(0,400),(0,255,0),1)
                    g1=380
                    g2=400
                    #cv2.rectangle(frame, (x1, y1), (x2, y2),(0, 255, 255), 2)
                    #cv2.circle(frame,(cent) ,radius=2,color=(0, 0, 255),thickness= 2)
                    #s1.append(tot)
                    #joblib.dump(tot,'simpen.pkl')
                    #print(tot)

                    if cent[1] > g1 and cent[1] < 390  :
                        xx1 = fps
                        joblib.dump(xx1,'data1.pkl')          
                        
                    if cent[1] > 390 and cent[1] < g2  :
                        xx2 = fps
                        joblib.dump(xx2,'data2.pkl')
                        
                    #if cent[1]>400 :    
                
                    #print(joblib.load('data1.pkl'),joblib.load('data2.pkl'))
                                
                    if con> 0.5 :
                                
                        if cs == "motor" :
                                cv2.rectangle(frame, (x1, y1), (x2, y2),(0, 255, 255), 2)
                                cv2.circle(frame,(cent) ,radius=2,color=(0, 0, 255),thickness= 2)
                                cv2.putText(frame,str(cs)+" ", (x1,y1 + 30), cv2.FONT_HERSHEY_PLAIN,2, (0,255,2),2)
                                if cent[1]> g2 and cent[1]<420: 
                                    d1 =joblib.load('data1.pkl')
                                    d2 =joblib.load('data2.pkl')
                                    tot = (d1+d2)
                                    S=tot/28
                                    Kec = (5.5/1000)/(S/3600)/16.5
                                    cv2.putText(frame,str(round((Kec), 2))+"km/h", (x1,y1 -20), cv2.FONT_HERSHEY_PLAIN,2, (0,0,255),2)
                                    arduino.write(bytes(str(round(((5.5/1000)/(S/3600)/16.5), 2)), 'utf-8'))
                                    #cv2.putText(frame,str(round(tot*750/25, 2))+"km/h", (x1,y1 -30), cv2.FONT_HERSHEY_PLAIN,2, (0,0,255),2)
                                    if Kec > 50 :
                                        img = frame[y1:y2, x1:x2]
                                        cv2.imshow('Tertangkap', img)
                                        Speed = round((Kec), 2)
                                        #img = cv2.resize(img, None, fx=1, fy=1,interpolation=cv2.INTER_CUBIC)
                                        cv2.imwrite(f"captures/{str(Speed)} kmh.jpg", img)
                                #cv2.putText(frame,str(cs)+" ", (x1,y1 + 30), cv2.FONT_HERSHEY_PLAIN,2, (0,255,255),2)
                                #cv2.putText(frame,str(round(con,2))+" ", (x1-20,y1 + 50), cv2.FONT_HERSHEY_PLAIN,2, (0,255,255),2)
                                #cv2.putText(frame,(i)+" ", (x1-20,y1 + 50), cv2.FONT_HERSHEY_PLAIN,2, (0,255,255),2)
                                        
                        else :
                                cv2.rectangle(frame, (x1, y1), (x2, y2),(0, 255, 0), 2)
                                cv2.circle(frame,(cent) ,radius=2,color=(0, 0, 255),thickness= 2)
                                cv2.putText(frame,str(cs)+" ", (x1,y1 + 30), cv2.FONT_HERSHEY_PLAIN,2, (0,255,2),2)
                                if cent[1]> g2 and cent[1]<420: 
                                    d1 =joblib.load('data1.pkl')
                                    d2 =joblib.load('data2.pkl')
                                    tot = (d1+d2)
                                    S=tot/28
                                    Kec = (5.5/1000)/(S/3600)/16.5
                                    cv2.putText(frame,str(round((Kec), 2))+"km/h", (x1,y1 -20), cv2.FONT_HERSHEY_PLAIN,2, (0,0,255),2)
                                    arduino.write(bytes(str(round(((5.5/1000)/(S/3600)/16.5), 2)), 'utf-8'))
                                    #cv2.putText(frame,str(round(tot*600/25, 2))+"km/h", (x1,y1 -30), cv2.FONT_HERSHEY_PLAIN,2, (0,0,255),2)
                                    if Kec > 50 :
                                        img = frame[y1:y2, x1:x2]
                                        cv2.imshow('tertangkap', img)
                                        Speed = round((Kec), 2)
                                        #img = cv2.resize(img, None, fx=1, fy=1,interpolation=cv2.INTER_CUBIC)
                                        cv2.imwrite(f"captures/{str(Speed)} kmh.jpg", img)
                                #cv2.putText(frame,str(round(con,2))+" ", (x1-20,y1 + 50), cv2.FONT_HERSHEY_PLAIN,2, (0,255,255),2)
    frame = cv2.resize(frame,(400,400),interpolation = cv2.INTER_AREA)                        #cv2.putText(frame,(i)+" ", (x1-20,y1 + 50), cv2.FONT_HERSHEY_PLAIN,2, (0,255,255),2)     
    
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
                
    