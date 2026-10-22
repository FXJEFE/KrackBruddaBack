"""Upload HF-ready folders. Token comes from the `hf auth login` cache (huggingface_hub.get_token()).
Usage: python models/upload_to_hf.py [hf_user=mrlarp] [--private] [--only name-substring]"""
import sys, pathlib
from huggingface_hub import HfApi, get_token
HF = pathlib.Path(__file__).resolve().parent / "hf"
def main():
    a = [x for x in sys.argv[1:] if not x.startswith("--")]; user = a[0] if a else "mrlarp"
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
    tok = get_token()
    if not tok: sys.exit("No HF token cached. Run:  hf auth login   (write token) then retry.")
    api = HfApi(token=tok); print("logged in as", api.whoami()["name"])
    for d in sorted(p for p in HF.iterdir() if p.is_dir()):
        if not d.name.startswith("larrylarp3-btc-") or (only and only not in d.name): continue
        rid = f"{user}/{d.name}"; api.create_repo(rid, private="--private" in sys.argv, exist_ok=True)
        if "--private" not in sys.argv: api.update_repo_settings(rid, private=False)
        api.upload_folder(repo_id=rid, folder_path=str(d), commit_message="upload model + card"); print("uploaded https://huggingface.co/" + rid)
if __name__ == "__main__": main()
