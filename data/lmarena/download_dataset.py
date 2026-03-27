from datasets import load_dataset
import os

def export_hf_home():
    # make sure that the HF_HOME dir is the custom one
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    os.environ["HF_HOME"] = os.path.join(cur_dir, "../../", ".cache/huggingface")
    print(f"HF_HOME: {os.environ['HF_HOME']}")

if __name__ == "__main__":
    export_hf_home()
    # take the first 10 rows
    ds = load_dataset("lmarena-ai/arena-human-preference-140k", split="train[:2]")
    # save this example to a csv file
    ds.to_csv("data/lmarena/sample-lmarena-data.csv")