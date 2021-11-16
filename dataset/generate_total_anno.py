import os
import json

detect_jsonfile_path = r'D:\Dataset\6000detection\annotations'
detect_imagefile_path=r'D:\Dataset\6000detection\images'
detect_targetfile_path=r'.\helmet_detect_total_anno.txt'
detect_target_file=open(detect_targetfile_path,'w')

embed_jsonfile_path = r'D:\Dataset\9000track\annotations'
embed_imagefile_path=r'D:\Dataset\9000track\images'
embed_targetfile_path=r'.\helmet_embed_total_anno.txt'
embed_target_file=open(embed_targetfile_path,'w')

cfg=json.load(open('../cfg/jde_cfg.json'))
class_names=cfg['class_names']
class_label={}
for i,name in enumerate(class_names):
    class_label[i]=name


def get_key(dict, value):
    for k,v in dict.items():
        if v==value:
            return k
    return None


def generate_total_anno(jsonfile_path,imagefile_path,target_file):

    for jsonfile_dir in os.listdir(jsonfile_path):
        file_list=os.listdir(os.path.join(jsonfile_path,jsonfile_dir))

        file_list.sort(key=lambda x: int(x[:-5])) # 按数字大小排序
        for jsonfile_name in file_list:
            jsonfile=os.path.join(jsonfile_path,jsonfile_dir,jsonfile_name)

            if jsonfile_dir=='download':
                imagepath = os.path.join(imagefile_path,jsonfile_dir,jsonfile_name.replace('.json','.png'))
            else:
                imagepath = os.path.join(imagefile_path,jsonfile_dir,jsonfile_name.replace('.json','.jpg'))

            target_file.write(imagepath)
            print(imagepath)
            data = json.load(open(jsonfile, 'r'))
            for object in data['shapes']:
                object_name = object['label']
                xmin, ymin, xmax, ymax = object['ltrb']
                if 'track_id'  not in object: # 表示不计算
                    track_id = -1#object['track_id']
                else:
                    track_id=object['track_id']
                target_file.write(" "+str(xmin)+',' +str(ymin)+','+str(xmax)+','+str(ymax)+','+str(get_key(class_label,object_name))+','+str(track_id))

            target_file.write('\n')
    target_file.close()

generate_total_anno(detect_jsonfile_path,detect_imagefile_path,detect_target_file)
generate_total_anno(embed_jsonfile_path,embed_imagefile_path,embed_target_file)


