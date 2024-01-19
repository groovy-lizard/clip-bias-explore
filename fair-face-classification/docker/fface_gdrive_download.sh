#!/bin/bash

fileid="1g7qNOZz9wC7OfOhcPqH1EZ5bk1UFGmlL"
filename="fairface-img-margin125-trainval.zip"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o ${filename}