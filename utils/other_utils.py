from git import Repo

def get_commit_message ():

    commit_hash_message = ""

    try:
        repo = Repo("./")
        commit_hash = repo.git.rev_parse("HEAD")
        commit_hash_message = f"This script is based on commit hash = {commit_hash}"
    finally:
        return commit_hash_message