#-*- coding:utf-8 -*-
import os
category_dirs = os.listdir("data")
categories_data = {}
# foreach all product review dir
vocab_list = []
for category_dir in category_dirs:
    with open("data/{}".format(category_dir), "r", encoding="utf-8") as file:
        for line in file.readlines():
            vocab_list.extend(list(line.strip()))
