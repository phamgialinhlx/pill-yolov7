import os
import tqdm
import pandas as pd
import easyocr
import fuzzywuzzy.fuzz as fuzz
import argparse
import difflib

def get_drug_name_dict(path):
  df = pd.read_json(path)
  # print(df)
  return df 

def OCR_drugname(img_path, reader):
  # OCR
  """
  Params used

  allowlist (string) - Force EasyOCR to recognize only subset of characters.(not yet)
  text_threshold (float) - Text confidence threshold
  decoder (string, default = 'greedy') - options are 'greedy', 'beamsearch' and 'wordbeamsearch'.
  height_ths (float, default = 0.5) - Maximum different in box height. Boxes with very different text size should not be merged.
  width_ths (float, default = 0.5) - Maximum horizontal distance to merge boxes.
  """
  result = reader.readtext(img_path, width_ths=20, height_ths=0.5, decoder='beamsearch',  text_threshold=0.5)
  drug_name_recg=[]
  for detect in result:
    text = str(detect[1]).lower()
    if text is not None and text !='':
      if text[0].isdigit() and (text.find(')')<=2 and text.find(')')>=0):
        sl_idx = text.rfind('sl:', 0, len(text))
        pill_no_idx = text.rfind('viên', 0, len(text))
        if (pill_no_idx - sl_idx in range(6,9)):
          # print(True)
          text = text[:sl_idx].rstrip()
        drug_name_recg.append(text[4:])
  return drug_name_recg
def match_text(drug_name, drug_name_dict):
    score_max = 0
    targ = drug_name
    for drugname in drug_name_dict['drugname']:
      score = difflib.SequenceMatcher(None, drug_name, drugname).ratio()
      # score = fuzz.token_set_ratio(drug_name.replace(" ",''), drugname.replace(" ",''))
      if score > score_max:
        score_max = score
        targ = drugname
    # print(drug_name, score_max)
    return (targ,score_max)

def extract_all(directory):
  """
  Extract all prescription files from a directory and returns a list of dictionaries {drugname, match, score}.
  """

  reader = easyocr.Reader(['vi', 'en'], gpu=True) # run once to initialize
  pres_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.png')]
  results = []
  pbar = tqdm.tqdm(total=len(pres_files))
  for i, pres_file in enumerate(pres_files):
    file_name = pres_file.split('/',10)[-1]
    drug_name_recg = OCR_drugname(pres_file, reader)
    result= {'file_name':file_name,'drugname':drug_name_recg}
    results.append(result)
    pbar.update(1)#update progress bar
  pbar.close()
  return results

def match_all_labels(path, drug_name_dict): 
  df=pd.read_json(path)
  result= pd.DataFrame(columns=['filename','drugname','match','score'])
  for i in range(df.shape[0]):
    for drugname in df.iloc[i]['drugname']:
      match, score = match_text(drugname,drug_name_dict)
      result.loc[result.shape[0]] = [df.iloc[i]['file_name'], drugname,match,score]
  return result

if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--save_dir', type=str, default='ocr', help='path to save result')
  parser.add_argument('--save_name', type=str, default='ocr_test_res', help='name of the file to save')
  parser.add_argument('--data_dir', type=str, help='path to data directory. Eg: public_test/prescription/image')
  parser.add_argument('--drug_name_dict', type=str, default='./ocr/drug_name_dict.json', help='path to drug name dictionary')
  parser.add_argument('--extract_ocr', action='store_true', help='extract ocr from images. If not specified, only match labels. Should only be used for first time')
  args = parser.parse_args()
  save_dir = args.save_dir
  save_name = args.save_name
  data_dir = args.data_dir
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)
  drug_name_dict = get_drug_name_dict(args.drug_name_dict)
  json_path = os.path.join(save_dir, f'extracted_{save_name}.json')
  if args.extract_ocr:
    extracted = pd.DataFrame(extract_all(data_dir))
    extracted.to_json(json_path, orient='records')
  matched = match_all_labels(json_path, drug_name_dict)
  save_path = os.path.join(save_dir, f'{save_name}.csv')
  matched.to_csv(save_path,index=False)
  print(f'OCR results are saved in {save_path}')