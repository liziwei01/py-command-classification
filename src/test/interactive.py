'''
Author: liziwei01
Date: 2022-12-03 00:46:19
LastEditors: liziwei01
LastEditTime: 2022-12-03 02:36:02
Description: file content
'''
"""Simple script that allows direct interaction with the model"""

import base

if __name__ == "__main__":
    while True:
        inp = input("Enter string: ")
        is_ci = base.predict_is_ci_str(inp)
        out = "Prediction: '{}' Is {}Command Injection!".format(inp, "" if is_ci else "Not ")
        print(out)
    