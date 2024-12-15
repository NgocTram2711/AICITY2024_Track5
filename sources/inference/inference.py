# CUDA_VISIBLE_DEVICES=1 python3 inference.py --test_path /mlcv1/WorkingSpace/Personal/haov/aicity2023/Track5_2024/aicity2024_track5_train/test-set/aicity2024_track5_test/videos/
from mmdet.apis import init_detector, inference_detector
from mmdet.core import DatasetEnum
import cv2
import numpy as np
import os
import argparse

from ensemble_boxes import *
from tqdm import tqdm

from utils.filter2 import Filter
from utils.detection_object import Human, Motor

#Xử lý 1 frame trong video, đưa qua bộ filter để tạo virtual boundingbox và remove overlap giữa các object 
#Trả về chuỗi gồm thông tin virtual boundingbox chứa vị trí của người và xe máy trong 1 frame
def process_objects(vid, fid, human_list, motor_list):
    filter = Filter(motor_list, human_list)
    result = ''
    all_class = filter.create_virtual()
    for obj in all_class:
        left, top, right, bottom, class_id, conf, _ = obj.get_box_info()
        result += ','.join(map(str, [vid, fid, left, top, right - left, bottom - top, class_id, conf])) + '\n'
    return result

#Xử lý từng video trong dataset, lấy ra thông tin virtual boundingbox của đối tượng người và xe máy của toàn bộ các frames trong video đó
def process_video(dataset, vid):
    result = ''
    for fid in dataset[vid].keys():
        if 'human' not in dataset[vid][fid].keys():
            dataset[vid][fid]['human'] = []
        if 'motor' not in dataset[vid][fid].keys():
            dataset[vid][fid]['motor'] = []
        result += process_objects(vid, fid, dataset[vid][fid]['human'], dataset[vid][fid]['motor']) 
    return result

#Phân loại đối tượng human và motor trong video
#Trả về thông tin các virtual boundingbox 
def Virtural_Expander(data: list):
    dataset = {}
    for line in data:
        vid, fid, left, top, width, height, cls, conf = line
        if int(float(cls)) != 1:
            
            if vid not in dataset.keys():
                dataset[vid] = {}
            if fid not in dataset[vid].keys():
                dataset[vid][fid] = {}
            if 'human' not in dataset[vid][fid].keys():
                dataset[vid][fid]['human'] = []
            dataset[vid][fid]['human'].append(Human(bbox=[float(left), float(top), float(width), float(height),float(cls), float(conf)]))
       
        else:
            if vid not in dataset.keys():
                dataset[vid] = {}
            if fid not in dataset[vid].keys():
                dataset[vid][fid] = {}
            if 'motor' not in dataset[vid][fid].keys():
                dataset[vid][fid]['motor'] = []
            dataset[vid][fid]['motor'].append(Motor(bbox=[float(left), float(top), float(width), float(height),float(cls), float(conf)]))
       
            # if 'human' not in dataset[vid][fid].keys():
            #     dataset[vid][fid]['human'] = []
            # dataset[vid][fid]['human'].append(Human(bbox=[float(left), float(top), float(width), float(height),float(cls), float(conf)]))
    # Create ouput
    results = ''
    for vid in tqdm(dataset.keys()):
        results += process_video(dataset, vid)
    return results

#Đếm số lượng mẫu trong mỗi lớp (từ 1 đến 9) trong tập dữ liệu
#Trả về một danh sách class_counts, trong đó mỗi phần tử tương ứng với số lượng mẫu của từng lớp
def count_samples_per_class(data):
    class_counts = [0,0,0,0,0,0,0,0,0] 
    for line in data:
        class_id = int(line[-2]) 
        class_counts[class_id-1] += 1
    return class_counts

#Tìm lớp có số lượng mẫu nhiều nhất 
#Trả về lớp đó cùng với danh sách số lượng mẫu cho từng lớp
def find_max(classes):
    classes_count = count_samples_per_class(classes)
    max_class = max(classes_count)
    return max_class, classes_count

#Xác định ngưỡng cho các lớp hiếm
def minority(p, classes): #p là ngưỡng tối thiểu cho trước, classes là danh sách gồm thông tin các lớp và số lượng mẫu
    #n_maxclass: lớp có số lượng mẫu lớn nhất
    #classes_count số lượng mẫu cho từng lớp
    n_maxclass, classes_count = find_max(classes)
    mean_samples = float(len(classes)/9)
    #alpha xác định mức độ hiếm của các lớp
    alpha = mean_samples/n_maxclass
    #Tìm lớp hiếm (rare classes) bằng cách so sánh số lượng mẫu của chúng với lớp có số lượng mẫu lớn nhất
    rare_classes = []
    for index, each_class in enumerate(classes_count):
        n_class = each_class
        if n_class < (n_maxclass * alpha):
            rare_classes.append(index)
    min_thresh = float('inf')
    #Tìm ngưỡng tối thiểu cho lớp hiếm
    for each_class_index in rare_classes:
        for each_sample in classes:
            if each_class_index != int(each_sample[-2]-1):
                continue
            if each_sample[-1] < min_thresh:
                min_thresh = each_sample[-1]
    return max(min_thresh, p)

#Lấy ra danh sách các đối tượng được phát hiện và chuẩn hóa theo kích thước video
def read_detections(lines: list):
    detections_dict = {}
    w, h = 1920, 1080 # NOTE: Change this to the actual width and height of the video
    for line in lines:
        #Mỗi line gồm một chuỗi chứa thông tin bounding box của đối tượng, cách nhau bởi dấu phẩy.
        video_id, frame, bb_left, bb_top, bb_width, bb_height, class_id, score = line.strip().split(',')
        frame = int(float(frame))
        video_id = int(video_id)
        if video_id not in detections_dict:
            detections_dict[video_id] = {}
        if frame not in detections_dict[video_id]:
            detections_dict[video_id][frame] = []
        #Tính lại tọa độ của bounding box theo tỷ lệ của kích thước video
        detections_dict[video_id][frame].append([float(bb_left) / w, float(bb_top) / h, (float(bb_width) + float(bb_left)) / w, 
        (float(bb_height) + float(bb_top)) / h,  float(score), int(float(class_id)), ])
    return detections_dict

#Phát hiện đối tượng dựa trên trọng số pre-trained và checkpoint Co-DERT 
def detect_video(
    test_path: str,
    config_path: str,
    checkpoint_files: list,
    batch_size: int,
) -> list:
    process_video_results = []
    configs_weights = [
        ('co_dino_5scale_swin_large_16e_o365tococo.py','epoch_10.pth'),
        ('640x640co_dino_5scale_swin_large_16e_o365tococo.py','epoch_10.pth'),
        ('1280x1280co_dino_5scale_swin_large_16e_o365tococo.py','epoch_10.pth'),
        ('640x640co_dino_5scale_swin_large_16e_o365tococo.py','epoch_15.pth'),
        ('1280x1280co_dino_5scale_swin_large_16e_o365tococo.py','epoch_15.pth'),
    ]
    for config_name, checkpoint_file in configs_weights:
        config_f_name = config_name.split(".")[0]
        checkpoint_file = os.path.join(checkpoint_files, checkpoint_file)

        lines = []
        config_file = os.path.join(config_path, config_name)
        #Sử dụng init_detector của mmdet để khởi tạo detector từ config file
        model = init_detector(config_file, checkpoint_file, DatasetEnum.COCO, device='cuda:0')

        #Xử lý lần lượt các video trong thư mục test
        for video_name in tqdm(os.listdir(test_path)):
            video_id = video_name.split(".")[0]
            video_path = os.path.join(test_path, video_name)
            frame_id = 0
            cap = cv2.VideoCapture(video_path)
            batch = []
            is_break = False
            while True:
                while len(batch) < batch_size:
                    ret, img = cap.read()
                    if not ret:
                        is_break = True
                        break
                    batch.append(img)
                if is_break:
                    break
                print(f"[INFO] Current frame_id: {frame_id}")
                #Sử dụng inference_detector của mmdet để dự đoán hình ảnh trên mô hình có sẵn
                results = inference_detector(model, batch)
                for idx, result in enumerate(results):
                    #bbox_result lưu thông tin boundingbox
                    bbox_result, segm_result = result, None

                    #Kết hợp các bounding box thành mảng
                    bboxes = np.vstack(bbox_result)

                    #Tạo nhãn cho đối tượng trong từng boundingbox
                    labels = [
                        np.full(bbox.shape[0], i, dtype=np.int32)
                        for i, bbox in enumerate(bbox_result)
                    ]
                    labels = np.concatenate(labels)
                    score_thr = 0.01
                    scores = None
                    if score_thr > 0:
                        scores = bboxes[:, -1]
                        #Lấy ra các chỉ số của các bounding box có confidence lớn hơn threshold
                        inds = scores > score_thr 
                        scores = scores[inds]
                        bboxes = bboxes[inds, :]
                        #Lấy ra các label tương ứng với các bounding box được chọn
                        labels = labels[inds]
                    width, height = img.shape[1], img.shape[0]
                    for label, score, bbox in zip(labels, scores, bboxes):
                        bbox = list(map(int, bbox))
                        label = int(label) + 1
                        #Tính toán chiều rộng và chiều cao của bounding box
                        w,h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                        #Mỗi dòng result chứa videoid, frameid, boundingbox, chiều rộng, cao, label và confidence score
                        lines.append(
                            f"{int(video_id)},{frame_id + idx + 1},{bbox[0]},{bbox[1]},{w},{h},{label},{score}\n"
                        )
                frame_id += len(batch)
                batch = []
            process_video_results.append(lines)
        
    return process_video_results


def fuse(
    process_video_results: list,
    video_path: str,
    iou_thr: float = 0.7,  # Tăng từ 0.5 lên 0.7
    skip_box_thr: float = 0.01,  # Tăng từ 0.0001 lên 0.01
) -> list:
    datas = [read_detections(item) for item in process_video_results]
    results = []
    w, h = 1920, 1080

    for video_name in tqdm(os.listdir(video_path)):
        video_id = int(video_name.split(".")[0])
        for frame_idx in range(1, 201):
            frame_idx = str(frame_idx)
            weights = [1] * len(datas)
            weights[0] = 3

            # Khởi tạo dict để lưu trữ box, scores và labels theo từng lớp
            class_boxes_dict = {}
            class_scores_dict = {}
            class_labels_dict = {}

            # Thu thập boxes, scores và labels từ tất cả các mô hình
            for idx, data in enumerate(datas):
                if video_id in data and int(frame_idx) in data[video_id]:
                    for box in data[video_id][int(frame_idx)]:
                        x1, y1, x2, y2 = box[:4]
                        score = box[4]
                        label = int(box[5])  # Đảm bảo label là số nguyên
                        # Khởi tạo list cho lớp này nếu chưa có
                        if label not in class_boxes_dict:
                            class_boxes_dict[label] = [[] for _ in range(len(datas))]
                            class_scores_dict[label] = [[] for _ in range(len(datas))]
                            class_labels_dict[label] = [[] for _ in range(len(datas))]
                        # Thêm dữ liệu vào lớp tương ứng
                        class_boxes_dict[label][idx].append([x1, y1, x2, y2])
                        class_scores_dict[label][idx].append(score)
                        class_labels_dict[label][idx].append(label)

            # Áp dụng WBF riêng cho từng lớp
            for label in class_boxes_dict.keys():
                boxes_list = class_boxes_dict[label]
                scores_list = class_scores_dict[label]
                labels_list = class_labels_dict[label]
                # Kiểm tra xem có bounding box cho lớp này không
                if any(len(boxes) > 0 for boxes in boxes_list):
                    # Áp dụng WBF
                    fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
                        boxes_list, scores_list, labels_list,
                        weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr
                    )
                    # Thêm kết quả vào danh sách kết quả chung
                    for i in range(len(fused_boxes)):
                        x1, y1, x2, y2 = fused_boxes[i]
                        results.append([
                            video_id,
                            frame_idx,
                            x1 * w,
                            y1 * h,
                            (x2 - x1) * w,
                            (y2 - y1) * h,
                            fused_labels[i],
                            fused_scores[i]
                        ])

    return results

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Inference')
    args.add_argument('--batch_size', type=int, default=1)
    args.add_argument('--checkpoint_path', type=str, default='weights')
    args.add_argument('--config_path', type=str, default='configs')
    args.add_argument('--p', type=float, default=0.0001)
    data_dir = os.path.abspath('../../data/aicity2024_track5_test/videos')
    args.add_argument('--test_path', type=str, default=data_dir)
    args = args.parse_args()

    p = args.p
    batch_size = args.batch_size
    test_path = args.test_path
    config_path = args.config_path
    checkpoint_files = args.checkpoint_path
    print("Start inference")
    process_video_results = detect_video(test_path, config_path, checkpoint_files, batch_size)

    print("Start Fuse")
    results = fuse(process_video_results, test_path)

    print("Start Minority")
    minority_score = minority(p, results)

    # Remove boxes with score less than minority_score
    new_results = []
    for result in results:
        if result[-1] >= minority_score:
            new_results.append(result)
    results = new_results   

    print("Start Virtural Expander")
    results = Virtural_Expander(results)
    
    with open("results.txt", "w") as f:
        f.write(results)