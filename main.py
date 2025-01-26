import os
import yaml
import shutil
import random  # train/val 分割用に追加
from PIL import Image, ImageFilter
from ultralytics import YOLO
from roboflow import Roboflow
from dotenv import load_dotenv


# 0. 環境変数の読み込み
load_dotenv()

# 1. 初期設定
# モデルのロード
model = YOLO('yolo11m.pt')

# ベースディレクトリとデータセットルートディレクトリの設定
base_dir = os.getcwd() # ベースディレクトリ (スクリプトが存在するディレクトリ)
dataset_root_dir = os.path.join(base_dir, 'dataset') # データセットのルートディレクトリ
dataset_test_dir = os.path.join(dataset_root_dir, 'test') # テストデータセットのルートディレクトリ
dataset_train_dir = os.path.join(dataset_root_dir, 'train') # 学習データセットのルートディレクトリ
dataset_val_dir = os.path.join(dataset_root_dir, 'valid') # 検証データセットのルートディレクトリ

# 入出力ディレクトリの設定
original_dataset_dir = os.path.join(base_dir, 'Citypersons-11') # オリジナル画像ディレクトリ (入力)

train_images_dir = os.path.join(dataset_train_dir, 'images') # 学習用劣化画像ディレクトリ
val_images_dir =  os.path.join(dataset_val_dir, 'images')
test_images_dir = os.path.join(dataset_test_dir, 'images')     # 検証用劣化画像ディレクトリ
train_labels_dir = os.path.join(dataset_train_dir, 'labels')   # 学習用ラベルディレクトリ
val_labels_dir =  os.path.join(dataset_val_dir, 'labels')
test_labels_dir = os.path.join(dataset_test_dir, 'labels')       # 検証用ラベルディレクトリ

dataset_config_path = os.path.join(dataset_root_dir, 'dataset_degraded.yaml') # データセット設定ファイルパス

# ディレクトリが存在しない場合は作成
os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(test_images_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)
os.makedirs(test_labels_dir, exist_ok=True)

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
def degrade_image(input_dir, output_dir):
    image_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for image_file in image_files:
        image_path = os.path.join(input_dir, image_file)
        output_path = os.path.join(output_dir, image_file)

        image = Image.open(image_path)
        # 画像劣化処理
        degraded_image = image.filter(ImageFilter.GaussianBlur(radius=10)) # blur radius を調整

        degraded_image.save(output_path)


# 5. ラベルディレクトリを劣化画像ディレクトリ内にコピーする関数
def copy_labels_to_degraded_dir(labels_dir, degraded_labels_dir):
    files = [f for f in os.listdir(labels_dir) if os.path.isfile(os.path.join(labels_dir, f)) and f.lower().endswith('.txt')]
    for file in files:
        src_path = os.path.join(labels_dir, file)
        dst_path = os.path.join(degraded_labels_dir, file)
        shutil.copyfile(src_path, dst_path)


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


# 7. メイン処理 (Train/Val 分割実行と各処理呼び出し)
if __name__ == '__main__':
    # 0. データセットのダウンロード
    rf = Roboflow(api_key="tUKOA4vr8IajOjGWYTC0")
    project = rf.workspace("citypersons-conversion").project("citypersons-woqjq")
    version = project.version(11)
    dataset = version.download("yolov11")


    # 7.1. Train データセットの処理
    print("Train データセット処理開始")
    original_train_images = os.path.join(original_dataset_dir, 'train', 'images')
    original_train_labels = os.path.join(original_dataset_dir, 'train', 'labels')
    degrade_image(original_train_images, train_images_dir) # Train用 劣化画像生成
    copy_labels_to_degraded_dir(original_train_labels, train_labels_dir) # Train用 ラベルコピー
    print("Train データセット処理完了")

    # 7.2. Val データセットの処理
    print("Val データセット処理開始")
    original_val_images = os.path.join(original_dataset_dir, 'valid', 'images')
    original_val_labels = os.path.join(original_dataset_dir, 'valid', 'labels')
    degrade_image(original_val_images, val_images_dir) # Val用 劣化画像生成
    copy_labels_to_degraded_dir(original_val_labels, val_labels_dir) # Val用 ラベルコピー
    print("Val データセット処理完了")

    # 7.3. Test データセットの処理
    print("Test データセット処理開始")
    original_test_images = os.path.join(original_dataset_dir, 'test', 'images')
    original_test_labels = os.path.join(original_dataset_dir, 'test', 'labels')
    degrade_image(original_test_images, test_images_dir) # Test用 劣化画像生成
    copy_labels_to_degraded_dir(original_test_labels, test_labels_dir) # Test用 ラベルコピー
    print("Test データセット処理完了")

    # 7.4. データセット設定ファイルの作成
    create_dataset_yaml()
    print("データセット設定ファイル作成完了")

    # 7.5. 学習の実行
    print("学習開始")
    results = model.train(
        data=dataset_config_path,
        epochs=100, #todo: 100に増やして実行
        imgsz=640,
        batch=16,
        name='yolov11m_degraded_trained_split' # runs/detect/yolov11m_degraded_trained_split に保存
    )
    print("学習完了")
