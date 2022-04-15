#!/bin/bash

echo -e "\n >>> Logged User: \n"
whoami

echo -e "\n\n >>> HOME folder: \n"
env | grep -w "HOME"

echo -e "\n\n >>> OS Info: \n"
uname -a

echo -e "\n\n >>> Working Directory: \n"
pwd

echo -e "\n\n >>> List Working Directory Folder: \n"
ls -l

echo -e "\n\n >>> List Root Folder: \n"
ls -l /

echo -e "\n\n >>> List Resources Folder: \n"
find /home/lenzi/resources | sed -e "s/[^-][^\/]*\//   |/g" -e "s/|\([^ ]\)/|-\1/"

echo -e "\n\n >>> Environment Variables: \n"
env

echo -e "\n\n >>> Python3 Libraries: \n"
pip3 freeze --all

echo -e "\n\n >>> RAM: \n"
free -h

echo -e "\n\n >>> Disk: \n"
df -h /

echo -e "\n\n >>> CPU: \n"
cat /proc/cpuinfo | egrep "$model name"

echo -e "\n\n >>> GPU: \n"
nvidia-smi | cat
