import cv2
import pandas as pd
print(cv2.__version__)

def main():
    cascade_src = '../data/cars.xml'
    video_src = '../data/car_test.avi'
    #video_src = 'dataset/video2.avi'

    cap = cv2.VideoCapture(video_src)
    car_cascade = cv2.CascadeClassifier(cascade_src)

    t=0
    f = 1278.351
    c = (659.588, 324.347)

    # FLANN 파라미터 1
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, tress=5)
    search_params = dict(checks=50)

    detector = cv2.xfeatures2d.SIFT_create()

    while True:
        ret, img = cap.read()
        if (type(img) == type(None)):
            break
        t+=1
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if t==1:
            first = gray
            cars = car_cascade.detectMultiScale(gray, 1.1, 1) # cars=[x,y,w,h]
            cars_df = pd.DataFrame(cars)
            middle = []
            for i in range(len(cars_df)):
                middle.append([((cars[i][0]+(0.5*cars[i][2]))/2-c[0]), cars[i][1], cars[i][2], cars[i][3]])
            middle = pd.DataFrame(middle)
            car = cars_df.loc[middle.idxmin()[0]]
            print(car)

            cv2.rectangle(img, (car[0],car[1]), (car[0]+car[2], car[1]+car[3]), (0,0,255), 2)

            detected = first[car[1]:car[1]+car[3], car[0]:car[0]+car[2]]
            kp1, des1 = detector.detectAndCompute(detected, None)

        if t>1:
            # second = gray
            # visual feature 감지기
            kp2, des2 = detector.detectAndCompute(gray, None)

            # flann
            matcher = cv2.FlannBasedMatcher(index_params, search_params)
            matches = matcher.knnMatch(des1, des2, k=2)

            good = []
            pts = []
            for i, (m, n) in enumerate(matches):
                if m.distance < 0.6 * n.distance:
                    good.append(m)
                    pts.append(kp2[m.trainIdx].pt)
            avr_w = 1.7
            avr_h = 1.6 #차량 w=1.7, h=1.6 설정
            df = pd.DataFrame(pts)
            distance = f * (avr_w / (df.max()[0] - df.min()[0]))
            distance = str(distance)
            x_min = int(df.min()[0])
            y_min = int(df.min()[1])
            x_max = int(df.max()[0])
            y_max = int(df.max()[1])
            cv2.rectangle(img, (x_min,y_min), (x_max,y_max), (0,0,255), 2)
            cv2.putText(img, distance, (x_max, y_min), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0),2)

            # dst = cv2.drawMatches(detected, kp1, second, kp2, good, None)
            # dst = cv2.resize(dst, dsize=(0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
            # cv2.imshow('dst', dst)
            # print(df)


        cv2.imshow('video', img)
        if cv2.waitKey(33) == 27:
            break


if __name__=="__main__":
    main()