import os
import yaml
import shutil
import random  # train/val 分割用に追加
from PIL import Image, ImageFilter
from ultralytics import YOLO

# 1. 初期設定
# モデルのロード
model = YOLO('yolo11m.pt')

# ベースディレクトリとデータセットルートディレクトリの設定
base_dir = 'I:/PycharmProjects/rg-wip' # ベースディレクトリ (スクリプトが存在するディレクトリ)
dataset_root_dir = os.path.join(base_dir, 'dataset') # データセットのルートディレクトリ

# 入出力ディレクトリの設定
original_images_dir = os.path.join(base_dir, 'original_images') # オリジナル画像ディレクトリ (入力)
degraded_images_dir = os.path.join(dataset_root_dir, 'images') # 劣化画像ディレクトリ (出力先ルート)
labels_output_dir = os.path.join(dataset_root_dir, 'labels') # ラベルディレクトリ (出力先ルート)

train_images_dir = os.path.join(degraded_images_dir, 'train') # 学習用劣化画像ディレクトリ
val_images_dir = os.path.join(degraded_images_dir, 'val')     # 検証用劣化画像ディレクトリ
train_labels_dir = os.path.join(labels_output_dir, 'train')   # 学習用ラベルディレクトリ
val_labels_dir = os.path.join(labels_output_dir, 'val')       # 検証用ラベルディレクトリ

dataset_config_path = os.path.join(dataset_root_dir, 'dataset_degraded.yaml') # データセット設定ファイルパス

# ディレクトリが存在しない場合は作成
os.makedirs(original_images_dir, exist_ok=True) # オリジナル画像ディレクトリ (入力用)
os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)

# Train/Val 分割比率
VAL_RATIO = 0.2 # 検証データの割合 (例: 20%)


# 2. Train/Val 分割関数
def split_train_val(input_images_dir, val_ratio=VAL_RATIO):
    image_files = [f for f in os.listdir(input_images_dir) if os.path.isfile(os.path.join(input_images_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(image_files) # ファイルリストをシャッフル
    val_size = int(len(image_files) * val_ratio)
    val_files = image_files[:val_size]
    train_files = image_files[val_size:]
    return train_files, val_files

# 3. オリジナル画像での物体検出とバウンディングボックスの保存 (Train/Val 分割対応)
def detect_and_save_bboxes(input_dir, output_labels_dir, image_files):
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
def degrade_image(input_dir, output_dir, image_files):
    for image_file in image_files:
        image_path = os.path.join(input_dir, image_file)
        output_path = os.path.join(output_dir, image_file)

        image = Image.open(image_path)
        # 画像劣化処理 (例: Gaussian blur)
        degraded_image = image.filter(ImageFilter.GaussianBlur(radius=5)) # blur radius を調整

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
def create_dataset_yaml(dataset_root_dir, output_yaml_path):
    data = {
        'path': dataset_root_dir,  # データセットのルートディレクトリ
        'train': 'images/train', # train ディレクトリへの相対パス
        'val': 'images/val',   # val ディレクトリへの相対パス
        'nc': len(model.names),
        'names': model.names
    }
    with open(output_yaml_path, 'w') as f:
        yaml.dump(data, f)


# 7. メイン処理 (Train/Val 分割実行と各処理呼び出し)
if __name__ == '__main__':
    # 7.1. Train/Val 分割
    train_files, val_files = split_train_val(original_images_dir)
    print(f"Train data size: {len(train_files)}, Val data size: {len(val_files)}")

    # 7.2. Train データセットの処理
    print("Train データセット処理開始")
    detect_and_save_bboxes(original_images_dir, train_labels_dir, train_files) # Train ラベル生成
    degrade_image(original_images_dir, train_images_dir, train_files) # Train 劣化画像生成
    # copy_labels_to_degraded_dir(train_labels_dir, train_labels_dir, train_files) # Train ラベルコピー (同じディレクトリなので不要だが、関数を共通化のため)
    print("Train データセット処理完了")

    # 7.3. Val データセットの処理
    print("Val データセット処理開始")
    detect_and_save_bboxes(original_images_dir, val_labels_dir, val_files) # Val ラベル生成
    degrade_image(original_images_dir, val_images_dir, val_files) # Val 劣化画像生成
    # copy_labels_to_degraded_dir(val_labels_dir, val_labels_dir, val_files) # Val ラベルコピー (同じディレクトリなので不要だが、関数を共通化のため)
    print("Val データセット処理完了")

    # 7.4. データセット設定ファイルの作成
    create_dataset_yaml(dataset_root_dir, dataset_config_path)
    print("データセット設定ファイル作成完了")

    # 7.5. 学習の実行
    print("学習開始")
    results = model.train(
        data=dataset_config_path,
        epochs=1, #todo: 100に増やして実行
        imgsz=640,
        batch=16,
        name='yolov11m_degraded_trained_split' # runs/detect/yolov11m_degraded_trained_split に保存
    )
    print("学習完了")
