from pathlib import Path
def main():
 from huggingface_hub import snapshot_download
 out=Path('models/trocr-handwritten');out.mkdir(parents=True,exist_ok=True)
 snapshot_download('microsoft/trocr-small-handwritten',local_dir=str(out),local_dir_use_symlinks=False)
 print('Modèle local:',out.resolve())
if __name__=='__main__':main()
