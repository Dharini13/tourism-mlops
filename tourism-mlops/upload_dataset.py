from huggingface_hub import login, upload_file
import os

login()

upload_file(
    path_or_fileobj="/content/data/train_data.csv",
    path_in_repo="train_data.csv",
    repo_id="Dharini95/tourism-package-dataset",
    repo_type="dataset"
)

upload_file(
    path_or_fileobj="/content/data/test_data.csv",
    path_in_repo="test_data.csv",
    repo_id="Dharini95/tourism-package-dataset",
    repo_type="dataset"
)

print("Dataset uploaded")
