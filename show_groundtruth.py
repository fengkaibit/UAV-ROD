import cv2
import numpy as np
import os
import xml.etree.ElementTree as ET

pi = 3.1415926535

def xywht_to_xyxyxyxy(center_coordinate):
    bboxes = np.empty((0, 8),dtype=np.int32)
    for rect in center_coordinate:
        bbox = cv2.boxPoints(((rect[0], rect[1]), (rect[2], rect[3]), rect[4] * 180 / pi))
        bbox = np.int0(bbox)
        bbox = np.reshape(bbox, [-1, ])
        bboxes = np.vstack((bboxes, [bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5], bbox[6], bbox[7]]))
    return bboxes

def load_xml(xml_path):
    anno_file = os.path.join(xml_path)

    with open(anno_file) as f:
        tree = ET.parse(f)

    r = {
        "height": int(tree.findall("./size/height")[0].text),
        "width": int(tree.findall("./size/width")[0].text),
    }
    instances = []
    for obj in tree.findall("object"):
        cls = obj.find("name").text
        bbox = obj.find("robndbox")
        bbox = [float(bbox.find(x).text) for x in ["cx", "cy", "w", "h", "angle"]]
        instances.append(
            {"category": cls, "bbox": bbox}
        )
    r["annotations"] = instances
    return r

def test_one_img(img_path, xml_path):
    img = cv2.imread(img_path)
    dicts = load_xml(xml_path)
    rbbox = []
    anno = dicts['annotations']
    for a in anno:
        rbbox.append(a['bbox'])
    bbox = xywht_to_xyxyxyxy(rbbox)

    for b in bbox:
        p1 = (b[0], b[1])
        p2 = (b[2], b[3])
        p3 = (b[4], b[5])
        p4 = (b[6], b[7])
        cv2.line(img, p1, p2, (0, 255, 0), 3)
        cv2.line(img, p2, p3, (0, 0, 255), 3) # The red line indicates the heading direction
        cv2.line(img, p3, p4, (0, 255, 0), 3)
        cv2.line(img, p4, p1, (0, 255, 0), 3)
    cv2.namedWindow("test", cv2.WINDOW_NORMAL)
    cv2.imshow('test', img)
    cv2.waitKey(0)

if __name__ == '__main__':
    img_dir = 'D:\\video\\UAV-ROD\\train\\images'
    xml_dir = 'D:\\video\\UAV-ROD\\train\\annotations'
    img_lists = os.listdir(img_dir)
    for l in img_lists:
        img_path = os.path.join(img_dir, l)
        xml_path = os.path.join(xml_dir, l.split('.')[0] + '.xml')
        test_one_img(img_path, xml_path)