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

def process_objects(vid, fid, human_list, motor_list):
    filter = Filter(motor_list, human_list)
    result = ''
    all_class = filter.create_virtual()
    for obj in all_class:
        left, top, right, bottom, class_id, conf, _ = obj.get_box_info()
        result += ','.join(map(str, [vid, fid, left, top, right - left, bottom - top, class_id, conf])) + '\n'
    return result

def process_video(dataset, vid):
    result = ''
    for fid in dataset[vid].keys():
        if 'human' not in dataset[vid][fid].keys():
            dataset[vid][fid]['human'] = []
        if 'motor' not in dataset[vid][fid].keys():
            dataset[vid][fid]['motor'] = []
        result += process_objects(vid, fid, dataset[vid][fid]['human'], dataset[vid][fid]['motor'])
    return result

def calculate_iou(box1, box2):
    # box format: [x1, y1, x2, y2]
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def apply_nms(objects, iou_thresh):
    """
    Perform Non-Maximum Suppression (NMS) on a list of objects.
    
    Args:
        objects: List of objects with format [[x1, y1, x2, y2, class_id, conf], ...]
        iou_thresh: Intersection-over-Union threshold for suppression.
    
    Returns:
        A list of filtered objects after NMS.
    """
    if len(objects) == 0:
        return []

    # Sort objects by confidence in descending order
    objects = sorted(objects, key=lambda x: x.bbox[5], reverse=True)
    keep = []

    while objects:
        # Pick the object with the highest confidence
        current = objects.pop(0)
        keep.append(current)

        # Remove objects with IoU > iou_thresh
        objects = [
            obj for obj in objects
            if calculate_iou(current.bbox[:4], obj.bbox[:4]) < iou_thresh
        ]

    return keep


def count_samples_per_class(data):
    class_counts = [0,0,0,0,0,0,0,0,0] 
    for line in data:
        class_id = int(line[-2]) 
        class_counts[class_id-1] += 1
    return class_counts

def find_max(classes):
    classes_count = count_samples_per_class(classes)
    max_class = max(classes_count)
    return max_class, classes_count


def minority(p, classes):
    n_maxclass, classes_count = find_max(classes)
    mean_samples = float(len(classes)/9)
    alpha = mean_samples/n_maxclass
    rare_classes = []

    for index, each_class in enumerate(classes_count):
        n_class = each_class
        if n_class < (n_maxclass * alpha):
            rare_classes.append(index)


    min_thresh = float('inf') # min_thresh = 1
    for each_class_index in rare_classes:
        for each_sample in classes:
            if each_class_index != int(each_sample[-2]-1):
                continue
            if each_sample[-1] < min_thresh:
                min_thresh = each_sample[-1]

    return max(min_thresh, p)

def read_detections(lines: list):
    detections_dict = {}
    w, h = 1920, 1080 # NOTE: Change this to the actual width and height of the video
    for line in lines:
        video_id, frame, bb_left, bb_top, bb_width, bb_height, class_id, score = line.strip().split(',')
        frame = int(float(frame))
        video_id = int(video_id)
        if video_id not in detections_dict:
            detections_dict[video_id] = {}
        if frame not in detections_dict[video_id]:
            detections_dict[video_id][frame] = []
        detections_dict[video_id][frame].append([float(bb_left) / w, float(bb_top) / h, (float(bb_width) + float(bb_left)) / w, 
        (float(bb_height) + float(bb_top)) / h,  float(score), int(float(class_id)), ])
    return detections_dict

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
        model = init_detector(config_file, checkpoint_file, DatasetEnum.COCO, device='cuda:0')
        print(test_path)
        # print(os.listdir('/target/path/data'))
       
        for video_name in tqdm(os.listdir(test_path)):
            print(video_name)
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
                results = inference_detector(model, batch)
                for idx, result in enumerate(results):
                    bbox_result, segm_result = result, None
                    bboxes = np.vstack(bbox_result)
                    labels = [
                        np.full(bbox.shape[0], i, dtype=np.int32)
                        for i, bbox in enumerate(bbox_result)
                    ]
                    labels = np.concatenate(labels)
                    score_thr = 0.01 # Các giá trị ngưỡng thử nghiệm [0.01, 0.3, 0.5]
                    scores = None
                    if score_thr > 0:
                        scores = bboxes[:, -1]
                        inds = scores > score_thr
                        scores = scores[inds]
                        bboxes = bboxes[inds, :]
                        labels = labels[inds]
                    width, height = img.shape[1], img.shape[0]
                    for label, score, bbox in zip(labels, scores, bboxes):
                        bbox = list(map(int, bbox))
                        label = int(label) + 1
                        w,h = bbox[2] - bbox[0], bbox[3] - bbox[1]
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


def Virtural_Expander(data: list, conf_thresh_human=0.3, conf_thresh_motor=0.5, iou_thresh=0.5):
    from concurrent.futures import ThreadPoolExecutor
    from itertools import chain

    def compute_iou_batch(bboxes1, bboxes2):
        """ Compute IoU for two batches of boxes """
        inter_x1 = np.maximum(bboxes1[:, 0], bboxes2[:, 0])
        inter_y1 = np.maximum(bboxes1[:, 1], bboxes2[:, 1])
        inter_x2 = np.minimum(bboxes1[:, 2], bboxes2[:, 2])
        inter_y2 = np.minimum(bboxes1[:, 3], bboxes2[:, 3])

        inter_area = np.maximum(inter_x2 - inter_x1, 0) * np.maximum(inter_y2 - inter_y1, 0)
        area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
        area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])

        union_area = area1 + area2 - inter_area
        return inter_area / np.maximum(union_area, 1e-6)

    def apply_nms(objects, iou_thresh):
        if not objects:
            return []

        bboxes = np.array([obj.bbox[:4] for obj in objects])
        scores = np.array([obj.bbox[5] for obj in objects])
        indices = np.argsort(scores)[::-1]

        keep = []
        while indices.size > 0:
            current = indices[0]
            keep.append(objects[current])

            if indices.size == 1:
                break

            rest = indices[1:]
            ious = compute_iou_batch(bboxes[current:current + 1], bboxes[rest])
            indices = rest[ious < iou_thresh]

        return keep

    dataset = {}
    for line in data:
        vid, fid, left, top, width, height, cls, conf = line
        conf = float(conf)
        cls = int(cls)

        if conf < (conf_thresh_human if cls != 1 else conf_thresh_motor):
            continue

        if vid not in dataset:
            dataset[vid] = {}
        if fid not in dataset[vid]:
            dataset[vid][fid] = {'human': [], 'motor': []}

        obj_class = 'human' if cls != 1 else 'motor'
        obj_instance = Human(
            bbox=[float(left), float(top), float(width), float(height), float(cls), conf]) if cls != 1 else Motor(
            bbox=[float(left), float(top), float(width), float(height), float(cls), conf])
        dataset[vid][fid][obj_class].append(obj_instance)

    def process_single_video(vid):
        results = []
        for fid in dataset[vid]:
            dataset[vid][fid]['human'] = apply_nms(dataset[vid][fid]['human'], iou_thresh)
            dataset[vid][fid]['motor'] = apply_nms(dataset[vid][fid]['motor'], iou_thresh)
            results.append(process_objects(vid, fid, dataset[vid][fid]['human'], dataset[vid][fid]['motor']))
        return results

    with ThreadPoolExecutor() as executor:
        all_results = list(chain(*executor.map(process_single_video, dataset.keys())))

    return ''.join(all_results)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Inference')
    args.add_argument('--batch_size', type=int, default=1)
    args.add_argument('--checkpoint_path', type=str, default='weights')
    args.add_argument('--config_path', type=str, default='configs')
    args.add_argument('--p', type=float, default=0.0001)
    args.add_argument('--test_path', type=str)
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

    new_results = [result for result in results if result[-1] >= minority_score]

    print("Start Virtural Expander")
    results = Virtural_Expander(new_results)

    with open("../../results.txt", "w") as f:
        f.write(results)
