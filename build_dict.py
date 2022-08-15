import pandas as pd
import os
import json
import tqdm
import argparse
def read_json_files_from_directory(directory):
  """
  Reads all json files from a directory and returns a list of dictionaries.
  """
  json_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.json')]
  json_data = []

  pbar = tqdm.tqdm(total=len(json_files))
  for i, json_file in enumerate(json_files):
    with open(json_file) as f:
      data =  json.load(f)
      file_name = json_file.split('/',10)[-1]
      data.append({'text':file_name, 'label':"filename"})
      json_data.append(data)
    pbar.update(1)#update progress bar
  pbar.close()
  return json_data

def build_dict(json_data):
  """
  Builds a dictionary of all pil in the json_data.
  """
  mapping_dict = []
  for pres in json_data:  
      for text in pres:
        if text['label'] == 'drugname':
          drug_name = text['text'][3:]
          id = text['mapping']
          mapping_dict.append((id, drug_name.lower()))
  df = pd.DataFrame(mapping_dict,columns = ['id','drugname'])
  res = df.drop_duplicates(subset=['id', 'drugname'], keep='last',ignore_index =  True)
  return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='ocr', help='path to save result')
    parser.add_argument('--save_name', type=str, default='drug_name_dict', help='name of the file to save')
    parser.add_argument('--data_dir', type=str, help='path to train prescription label directory. Eg: public_train/prescription/label')
    args = parser.parse_args()
    save_dir = args.save_dir
    save_name = args.save_name
    data_dir = args.data_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    json_data = read_json_files_from_directory(data_dir)
    drug_name_dict = build_dict(json_data)
    drug_name_dict.to_csv(os.path.join(save_dir, save_name+'.csv'), index=False)
    drug_name_dict.to_json(os.path.join(save_dir, save_name+'.json'), orient='records')
    print('save dict to {}'.format(os.path.join(save_dir, save_name+'.csv')))
    print('save dict to {}'.format(os.path.join(save_dir, save_name+'.json')))
