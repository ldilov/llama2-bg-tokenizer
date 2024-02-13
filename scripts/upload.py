from pathlib import Path

from huggingface_hub import HfApi, HfFolder, Repository


def upload_tokenizer_to_huggingface(repo_name, organization=None, private=False):
    """
    Uploads a tokenizer to the Hugging Face Hub.

    Parameters:
    - tokenizer_directory: The directory where the tokenizer files are saved.
    - repo_name: The repository name for your tokenizer on the Hub.
    - organization: The organization under which to upload the tokenizer. If None, uploads under your user.
    - private: Whether the repository should be private.
    """

    # Authenticate with Hugging Face
    token = HfFolder.get_token()
    if token is None:
        raise ValueError("Hugging Face authentication token not found. Please log in using `huggingface-cli login`.")

    api = HfApi()

    # Define repository name and organization
    repo_id = f"{organization}/{repo_name}" if organization else repo_name

    # Create or get repository
    repo_url = api.create_repo(
        repo_id=repo_id,
        token=token,
        private=private,
        exist_ok=True
    )

    repo = Repository(local_dir=repo_name, clone_from=repo_url, use_auth_token=True)
    repo.git_pull()
    repo.push_to_hub()

    # Push to the Hub
    repo.git_add(auto_lfs_track=True)
    repo.git_commit("Add tokenizer files")
    repo.git_push()

    print(f"Tokenizer successfully uploaded to {repo_url}")


tokenizer_directory = str(Path(__file__).parent / 'saved_models' / 'llama')
repo_name = 'llama2-bg-tokenizer'
organization = 'ldilov'
private = False

upload_tokenizer_to_huggingface(tokenizer_directory, repo_name, organization, private)