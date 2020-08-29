import json, sys, importlib, os, time
import time


def run():
    with open("Z:\\job\\admin\\3\\3\\info\\ca9818dbd7544e49951988a6c402a0cb\\input\\system_1_1\\input.txt", "r") as f:
        input = f.read()
        f.close()

    input = json.loads(input)
    sys.path.insert(0, "Z:\\algorithm\\admin\\1\\1\\app\\72f03225cbaa44828b0c1e2e9c189a0e\\un_compress")
    main = importlib.import_module("main")
    output = main.main(**input)

    for i in range(10):
        print(i)
        time.sleep(1)
    if output == None:
        output = {}
    base_path = "Z:\job\admin\3\3\info\ca9818dbd7544e49951988a6c402a0cb\\algorithm/output/"
    with open(base_path + "output.txt", "w") as f:
        f.write(json.dumps(output))
        f.close()


if __name__ == '__main__':
    run()

