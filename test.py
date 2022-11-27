'''
Author: liziwei01
Date: 2022-11-27 07:05:01
LastEditors: liziwei01
LastEditTime: 2022-11-27 07:30:25
Description: file content
'''
import numpy as np

str = "&lt;!--#exec%20cmd=&quot;/bin/cat%20/etc/passwd&quot;--&gt;"

def command2Matrix(command):
	line = []
	lines = []
	mid = []
	# divide line to words
	for i in range(len(command)):
		if command[i] == " ":
			lines.append(line)
			line = []
		elif command[i] == ";":
			lines.append(line)
			line = []
		else:
			char = ord(str[i])
			line.append(char)

ss = [0, 1, 2]

print(ss[:2])