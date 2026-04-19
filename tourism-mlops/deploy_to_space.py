from huggingface_hub import login, create_repo, upload_folder

login()

upload_folder(
    folder_path="tourism-app",
    repo_id="Dharini95/tourism-app",
    repo_type="space"
)
