import os

base = r"C:\Users\avash\OneDrive\Desktop\projects\ad_ablation_study\Alzheimer (Preprocessed Data)\Alzheimer (Preprocessed Data)"

for cls in os.listdir(base):
    cls_path = os.path.join(base, cls)
    if os.path.isdir(cls_path):
        count = len(os.listdir(cls_path))
        print(f"{cls}: {count}")