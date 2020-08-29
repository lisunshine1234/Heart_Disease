import json, sys, importlib, os, time
import time


def run():
    a = {"aaa": "111", "bbb": "222", "ccc": "333", "ddd": "444"}
    print("方式1：for i in a:")
    for i in a:
        print(i)
    print("方式2：for i in a.keys():")
    for i in a.keys():
        print(i)
    print("方式3：for i in a.values():")
    for i in a.values():
        print(i)
    print("方式4：for i in a.items():")
    for i in a.items():
        print(i)


if __name__ == '__main__':
    run()
