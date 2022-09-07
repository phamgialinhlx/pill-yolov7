python build_dict.py --data_dir /home/pill/competition/dataset/public_train/prescription/label
python ocr.py --extract_ocr --data_dir /home/pill/competition/dataset/public_test/prescription/image
python preprocessing.py --origin_path /home/pill/competition/dataset/public_test/pill/images --overwrite