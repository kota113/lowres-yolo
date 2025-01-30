import os
import zipfile

import yaml
import shutil
import random  # train/val 分割用に追加
from PIL import Image, ImageFilter
from ultralytics import YOLO
import kagglehub


# 1. 初期設定
# モデルのロード
model = YOLO('yolo11m.pt')
# Download latest version
dataset_dl_path = kagglehub.dataset_download("mdfahimbinamin/100k-vehicle-dashcam-image-dataset")
# copy directory
shutil.move(dataset_dl_path, 'original_images')


# ベースディレクトリとデータセットルートディレクトリの設定
base_dir = os.getcwd() # ベースディレクトリ (スクリプトが存在するディレクトリ)
dataset_root_dir = os.path.join(base_dir, 'dataset') # データセットのルートディレクトリ

# 入出力ディレクトリの設定
original_images_dir = os.path.join(base_dir, 'original_images') # オリジナル画像ディレクトリ (入力)

train_images_dir = os.path.join(dataset_root_dir, 'train', 'images') # 学習用劣化画像ディレクトリ
val_images_dir = os.path.join(dataset_root_dir, 'val', 'images')     # 検証用劣化画像ディレクトリ
test_images_dir = os.path.join(dataset_root_dir, 'test', 'images')   # テスト用劣化画像ディレクトリ
train_labels_dir = os.path.join(dataset_root_dir, 'train', 'labels')   # 学習用ラベルディレクトリ
val_labels_dir = os.path.join(dataset_root_dir, 'val', 'labels')       # 検証用ラベルディレクトリ
test_labels_dir = os.path.join(dataset_root_dir, 'test', 'labels')     # テスト用ラベルディレクトリ

dataset_config_path = os.path.join(dataset_root_dir, 'dataset_degraded.yaml') # データセット設定ファイルパス

# ディレクトリが存在しない場合は作成
os.makedirs(original_images_dir, exist_ok=True) # オリジナル画像ディレクトリ (入力用)
os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(test_images_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)
os.makedirs(test_labels_dir, exist_ok=True)


# 3. オリジナル画像での物体検出とバウンディングボックスの保存 (Train/Val 分割対応)
def detect_and_save_bboxes(input_dir, output_labels_dir):
    image_files = os.listdir(input_dir)
    for image_file in image_files:
        image_path = os.path.join(input_dir, image_file)
        results = model.predict(str(image_path))

        # バウンディングボックスとラベルを保存 (YOLO format)
        label_file_path = os.path.join(output_labels_dir, os.path.splitext(image_file)[0] + '.txt')
        with open(label_file_path, 'w') as f:
            for result in results:
                if result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        xywhn = box.xywhn[0]  # normalized xywh
                        class_id = int(box.cls[0])  # class id
                        line = f'{class_id} {" ".join(map(str, xywhn.tolist()))}\n'
                        f.write(line)
                else:
                    pass # No detections, create empty label file


# 4. 画像の劣化
def degrade_image(input_dir, output_dir):
    image_files = os.listdir(input_dir)
    for image_file in image_files:
        image_path = os.path.join(input_dir, image_file)
        output_path = os.path.join(output_dir, image_file)

        image = Image.open(image_path)
        # 画像劣化処理 (例: Gaussian blur)
        degraded_image = image.filter(ImageFilter.GaussianBlur(radius=2)) # blur radius を調整
        # ノイズを追加
        for _ in range(10):
            x = random.randint(0, image.width - 1)
            y = random.randint(0, image.height - 1)
            degraded_image.putpixel((x, y), (255, 255, 255))

        degraded_image.save(output_path)


# 5. ラベルディレクトリを劣化画像ディレクトリ内にコピーする関数 (Train/Val 分割対応)
def copy_labels_to_degraded_dir(labels_dir, degraded_labels_dir, image_files):
    for image_file in image_files:
        label_file = os.path.splitext(image_file)[0] + '.txt'
        original_label_path = os.path.join(labels_dir, label_file)
        degraded_label_path = os.path.join(degraded_labels_dir, label_file)
        if os.path.exists(original_label_path): # ラベルファイルが存在する場合のみコピー
            shutil.copy2(original_label_path, degraded_label_path)


# 6. データセット設定ファイルの作成 (Train/Val パス修正)
def create_dataset_yaml():
    data = {
        'path': dataset_root_dir,  # データセットのルートディレクトリ
        'nc': len(model.names),
        'names': model.names,
        "train": "../train/images",
        "val": "../valid/images",
        "test": "../test/images"
    }
    with open(dataset_config_path, 'w') as f:
        yaml.dump(data, f)


# 7. メイン処理
if __name__ == '__main__':
    # 7.2. Train データセットの処理
    print("Train データセット処理開始")
    source_images = os.path.join(original_images_dir, 'train')
    detect_and_save_bboxes(source_images, train_labels_dir) # Train ラベル生成
    degrade_image(source_images, train_images_dir) # Train 劣化画像生成
    print("Train データセット処理完了")

    # 7.3. Val データセットの処理
    print("Val データセット処理開始")
    source_images = os.path.join(original_images_dir, 'val')
    detect_and_save_bboxes(source_images, val_labels_dir) # Val ラベル生成
    degrade_image(source_images, val_images_dir) # Val 劣化画像生成
    print("Val データセット処理完了")

    # 7.4. Test データセットの処理
    print("Test データセット処理開始")
    source_images = os.path.join(original_images_dir, 'test')
    detect_and_save_bboxes(source_images, test_labels_dir) # Test ラベル生成
    degrade_image(source_images, test_images_dir) # Test 劣化画像生成
    print("Test データセット処理完了")

    # 7.4. データセット設定ファイルの作成
    create_dataset_yaml()
    print("データセット設定ファイル作成完了")

    # 7.5. 学習の実行
    print("学習開始")
    results = model.train(
        data=dataset_config_path,
        epochs=1,
        imgsz=640,
        batch=16,
        name='yolov11m_degraded_trained_split' # runs/detect/yolov11m_degraded_trained_split に保存
    )
    print("学習完了")
