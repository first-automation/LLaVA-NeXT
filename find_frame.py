import cv2
import torch
from PIL import Image
import open_clip
import numpy as np

# 動画パスとフレーム抽出間隔を設定
video_path = "playground/demo/pc_assembly_youtube.mp4"
interval = 5  # 秒

# OpenCLIPモデルと前処理の設定
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
tokenizer = open_clip.get_tokenizer('ViT-B-32')
model.to(device)

def extract_frames(video_path, interval=1):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % (interval * int(cap.get(cv2.CAP_PROP_FPS))) == 0:
            frames.append(frame)
        frame_count += 1
        
    cap.release()
    return frames

def encode_frames(frames):
    frame_features = []
    for frame in frames:
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            frame_feature = model.encode_image(image)
        frame_features.append(frame_feature)
    return frame_features

def encode_texts(descriptions):
    text_features = []
    for description in descriptions:
        text = tokenizer(description).to(device)
        with torch.no_grad():
            text_feature = model.encode_text(text)
        text_features.append(text_feature)
    return text_features

def calculate_similarity(frame_features, text_features):
    similarities = []
    for text_feature in text_features:
        similarity = []
        for frame_feature in frame_features:
            sim = torch.cosine_similarity(text_feature, frame_feature).item()
            similarity.append(sim)
        similarities.append(similarity)
    return np.array(similarities)

def select_frames_with_dp(similarities):
    num_descriptions = similarities.shape[0]
    num_frames = similarities.shape[1]
    
    dp = np.full((num_descriptions, num_frames), -np.inf)
    path = np.full((num_descriptions, num_frames), -1, dtype=int)

    dp[0] = similarities[0]
    
    for i in range(1, num_descriptions):
        for j in range(i, num_frames):
            for k in range(i - 1, j):
                if dp[i - 1][k] + similarities[i][j] > dp[i][j]:
                    dp[i][j] = dp[i - 1][k] + similarities[i][j]
                    path[i][j] = k

    selected_frames = []
    max_index = np.argmax(dp[-1])
    for i in range(num_descriptions - 1, -1, -1):
        selected_frames.append(max_index)
        max_index = path[i][max_index]
    
    selected_frames.reverse()
    return selected_frames

# 解説文のリスト
descriptions = [
    "Intel Core i5-12600K CPUの箱を開ける",
    "ASUS B600M PRO Gaming Motherboardの箱を開ける",
    "モザイド封入のMotherboardを取り出す",
    "CPUをMotherboardに取り付ける",
    "RAMモジュールをMotherboardに取り付ける",
    "CPU風扇を取り付け、電線を接続する",
    "PCケースを開ける",
    "CPU風扇を取り付け、電線を接続する",
    "パワーサプライPSUを取り付け、電線を接続する",
    "GPUカード (GeForce RTX 3060)を取り付け、電線を接続する",
    "PCケースの側面パネルを閉じる",
    "Windows 10 Pro 21H2をインストールする",
    "OneBenchMarkを実行して、PCの性能を測定する",
    "Apex Legendsゲームをプレイして、PCのグラフィックス性能を測定する"
]

# 各ステップを実行
frames = extract_frames(video_path, interval)
frame_features = encode_frames(frames)
text_features = encode_texts(descriptions)
similarities = calculate_similarity(frame_features, text_features)
selected_frames_indices = select_frames_with_dp(similarities)

# 選択されたフレームを取得
selected_frames = [frames[idx] for idx in selected_frames_indices]

# 選択されたフレームを保存または表示する例
for i, frame in enumerate(selected_frames):
    cv2.imwrite(f"selected_frame_{i}.png", frame)

print("フレーム抽出と選択が完了しました。")
