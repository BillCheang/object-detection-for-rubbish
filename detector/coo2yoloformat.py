import json
import os

def load_coco_annotations(coco_annotation_file):
    with open(coco_annotation_file, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    return coco_data

def convert_coco_to_yolo(coco_data, output_dir):
    images = {image['id']: image for image in coco_data['images']}
    categories = {category['id']: category for category in coco_data['categories']}
    
    for annotation in coco_data['annotations']:
        image = images[annotation['image_id']]
        category = categories[annotation['category_id']]
        
        x_center = (annotation['bbox'][0] + annotation['bbox'][2] / 2) / image['width']
        y_center = (annotation['bbox'][1] + annotation['bbox'][3] / 2) / image['height']
        width = annotation['bbox'][2] / image['width']
        height = annotation['bbox'][3] / image['height']
        
        yolo_label = f"{category['id']} {x_center} {y_center} {width} {height}\n"
        label_file = os.path.join(output_dir, f"{os.path.splitext(image['file_name'])[0]}.txt")
        
        os.makedirs(os.path.dirname(label_file), exist_ok=True)
        with open(label_file, 'a') as f:
            f.write(yolo_label)

def filter_images_by_batch(coco_data, batch_name, start_index, end_index):
    batch_images = [image for image in coco_data['images'] if image['file_name'].startswith(f"{batch_name}/")]
    filtered_images = batch_images[start_index:end_index+1]
    filtered_image_ids = {image['id'] for image in filtered_images}
    
    filtered_annotations = [annotation for annotation in coco_data['annotations'] if annotation['image_id'] in filtered_image_ids]
    
    filtered_coco_data = {
        "info": coco_data["info"],
        "images": filtered_images,
        "annotations": filtered_annotations,
        "categories": coco_data["categories"],
        "licenses": coco_data.get("licenses", [])
    }
    
    return filtered_coco_data

def main():
    coco_annotation_file = './data/annotations.json'  # 替換為您的 COCO 標註文件路徑
    output_dir = './yolo'  # 替換為您希望保存 YOLO 標籤文件的目錄
    batch_name = 'batch_15'  # 替換為您的批次名稱
    start_index = 0  # 替換為您的起始索引
    end_index = 9  # 替換為您的結束索引
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    coco_data = load_coco_annotations(coco_annotation_file)
    filtered_coco_data = filter_images_by_batch(coco_data, batch_name, start_index, end_index)
    convert_coco_to_yolo(filtered_coco_data, output_dir)

if __name__ == "__main__":
    main()
