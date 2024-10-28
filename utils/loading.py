import os
import yaml
from .paths import HERE, CONFIG
from .directory import createDir


def LoadConfig(dir_name=CONFIG, args_name="base"):
    # Look for name of the yaml file in the directory recursively
    for root, dirs, files in os.walk(dir_name):
        if args_name + ".yaml" in files:
            break
    else:
        raise ValueError(f"File {args_name}.yaml not found in {dir_name}")
    with open(dir_name / (args_name + ".yaml")) as f:
        opt = yaml.load(f, Loader=yaml.FullLoader)

    ROOT = HERE / "runs" / args_name
    # ROOT = Path('/home/nbiescas/Desktop/CVC/CVC_internship/runs') / test_name
    opt["run_name"] = args_name

    opt["root_dir"] = ROOT
    opt["weights_dir"] = ROOT / "weights"
    opt["output_dir"] = ROOT / "images"
    opt["output_svm"] = ROOT / "images" / "svm"
    opt["output_kmeans"] = ROOT / "images" / "kmeans"
    opt["json_kmeans"] = ROOT / "images" / "kmeans"
    opt["json_svm"] = ROOT / "images" / "svm"

    createDir(opt["root_dir"])
    createDir(opt["weights_dir"])
    createDir(opt["output_dir"])
    createDir(opt["output_svm"])
    createDir(opt["output_kmeans"])

    opt["network"]["checkpoint"] = opt["network"].get("checkpoint", None)

    return opt


# NOTE: debug
print(f"DEDUG - #print loaded config: {LoadConfig()}")
