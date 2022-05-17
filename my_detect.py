

import yolov5detect

from yolov5detect import detect, annotation

#annotation()




import torch

# Model
#model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5m, yolov5l, yolov5x, etc.
model = torch.hub.load('ultralytics/yolov5', 'custom', 'yolov5/runs/train/exp5/weights/best.pt')  # custom trained model
# Images
#im = 'https://ultralytics.com/images/zidane.jpg'  # or file, Path, URL, PIL, OpenCV, numpy, list
x = 0

import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # Inference
    results = model(img)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    annotator = annotation.Annotator(img.copy())
    xy = results.pandas().xyxy  # im predictions (pandas)
    #print(xy)
    for sd in xy:
        #print('len= ', sd)
        for row in sd.itertuples():
            print(row)
            box = [row.xmin, row.ymin, row.xmax, row.ymax]

        #print(sd.confidence)
            annotator.box_label(box, label=f'{row.name} {round(row.confidence, 2)}', color=(255,182,193), txt_color=(255, 255, 255))
            if row.confidence < 0.27:
                results.crop(box)

#    cv2.imshow("camera", img)

    cv2.imshow("camera", annotator.result())
    if cv2.waitKey(10) == 27: # Клавиша Esc
        break


cap.release()
cv2.destroyAllWindows()





#   annotator = annotation.Annotator(img.copy())
#   for *xyxy, conf, cls in results:
#       annotator.box_label(xyxy, f"{names[int(cls)]} {conf:.2f}")
#   cv2.imwrite(annotator.result(), str(output / item.name))
# print('результ:',results)
# Results
#    results.print()  # or .show(), .save(), .crop(), .pandas(), etc.