import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from PIL import Image
from torchvision import transforms
from .utils import merge_bboxes
import random
from .utils import image2Modelinput

class EmbedDataset(Dataset):
    def __init__(self,train_lines,extractor_image_size):
        super(EmbedDataset,self).__init__()
        self.num_lines=len(train_lines)
        self.train_lines=train_lines
        self.extractor_image_size=extractor_image_size

    def __len__(self):
        return self.num_lines

    def box_filter(self,ltrb_box,model_w,model_h,min_limit=10):
        if len(ltrb_box)>0:
            ltrb_box[:, 0:2][ltrb_box[:, 0:2] < 0] = 0  # 越界
            ltrb_box[:, 2][ltrb_box[:, 2] > model_w] = model_w
            ltrb_box[:, 3][ltrb_box[:, 3] > model_h] = model_h
            ltrb_box_w = ltrb_box[:, 2] - ltrb_box[:, 0]
            ltrb_box_h = ltrb_box[:, 3] - ltrb_box[:, 1]

            ltrb_box = ltrb_box[(ltrb_box_w > min_limit) & (ltrb_box_h > min_limit)]
            return ltrb_box
        return ltrb_box

    def __getitem__(self, index):
        n = self.num_lines
        index = index % n
        line = self.train_lines[index].split()
        origin_image = Image.open(line[0])
        origin_w, origin_h = origin_image.size
        origin_image = np.array(origin_image)
        objs = [list(map(int, obj.split(','))) for obj in line[1:]]
        objs = np.array(objs,dtype=np.int32)
        objs = self.box_filter(objs, origin_w, origin_h)  # 输入大小为原图片大小
        image_crops=[]
        tids=[]
        for obj in objs:
            if obj[5]==-1:
                continue
            x1,y1,x2,y2 = obj[0:4]
            crop = origin_image[y1:y2,x1:x2,:]
            crop,_,_ = image2Modelinput(crop,self.extractor_image_size, is_letterbox_image=True, type_modelinput='tensor')
            image_crops.append(crop)
            tids.append(obj[5])
        if len(image_crops)==0:
            ew,eh=self.extractor_image_size
            return torch.empty((0,3,eh,ew),dtype=torch.float32),torch.empty((0,1),dtype=torch.int32)
        image_crops = torch.cat(image_crops,dim=0) #N,C,H,W tensor
        tids=torch.tensor(tids,dtype=torch.int32,requires_grad=False)[:,None] # N,1
        return image_crops,tids

def embed_dataset_collate(batch):
    imagess = []
    tidss = []
    for imgs, tids in batch:
        imagess.append(imgs)
        tidss.append(tids)

    imagess = torch.cat(imagess,dim=0) # B,C,H,W
    tidss=torch.cat(tidss,dim=0)
    return imagess, tidss # 输出images为tensor(B,C,H,W),tidss为tensor(B,1),int,requires_grad=False
    # 有可能为empty

class TrackerDataset(Dataset):
    def __init__(self, train_lines):
        super(TrackerDataset, self).__init__()

        self.train_lines = train_lines
        self.num_lines = len(train_lines)

    def __len__(self):
        return self.num_lines

    def box_filter(self, ltrb_box, model_w, model_h, min_limit=10):
        if len(ltrb_box) > 0:
            ltrb_box[:, 0:2][ltrb_box[:, 0:2] < 0] = 0  # 越界
            ltrb_box[:, 2][ltrb_box[:, 2] > model_w] = model_w
            ltrb_box[:, 3][ltrb_box[:, 3] > model_h] = model_h
            ltrb_box_w = ltrb_box[:, 2] - ltrb_box[:, 0]
            ltrb_box_h = ltrb_box[:, 3] - ltrb_box[:, 1]

            ltrb_box = ltrb_box[(ltrb_box_w > min_limit) & (ltrb_box_h > min_limit)]
            return ltrb_box
        return ltrb_box


    def __getitem__(self, index):
        # 返回img为tensor(C,H,W),范围0到1,float32
        # 返回target为tensor(-1,5),xywh,对应在输入图片上,float32
        lines = self.train_lines
        n = self.num_lines
        index = index % n

        line = lines[index].split()
        origin_image = Image.open(line[0])
        origin_w, origin_h = origin_image.size
        origin_image=np.array(origin_image)
        objs = [list(map(int, obj.split(','))) for obj in line[1:]]
        objs = np.array(objs, dtype=np.int32)
        objs = self.box_filter(objs, origin_w, origin_h)  # 输入大小为原图片大小

        if len(objs) != 0:
            box = objs[:, :4]
            # 前面已经判断过是否越界了
            # 转换为ltwh!!!
            box[:, 2] = box[:, 2] - box[:, 0]
            box[:, 3] = box[:, 3] - box[:, 1]

            objs = np.concatenate([box, objs[:, 4:6]], axis=1)
        return origin_image, objs

def tracker_dataset_collate(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append(img)
        bboxes.append(box)

    return images, bboxes # 输出images为tensor(B,C,H,W),bboxes为list[B*tensor(-1,5 or 6)],都是float32,requires_grad=False

class YoloDataset(Dataset):
    def __init__(self, train_lines, model_image_size, mosaic=True, is_random=True,output_origin=False):
        super(YoloDataset, self).__init__()

        self.train_lines = train_lines
        self.num_lines = len(train_lines)
        self.model_image_size = model_image_size
        self.mosaic =  mosaic
        self.flag = True
        self.is_random =  is_random # 是否要实时数据增强
        self.output_origin=output_origin # 直接返回PIL.Image

        self.totensor=transforms.ToTensor()
        self.colorjitter=transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2)

    def __len__(self):
        return self.num_lines
    def shuffle(self):
        random.shuffle(self.train_lines)
    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def box_filter(self, ltrb_box, model_w, model_h, min_limit=10):
        if len(ltrb_box) > 0:
            ltrb_box[:, 0:2][ltrb_box[:, 0:2] < 0] = 0  # 越界
            ltrb_box[:, 2][ltrb_box[:, 2] > model_w] = model_w
            ltrb_box[:, 3][ltrb_box[:, 3] > model_h] = model_h
            ltrb_box_w = ltrb_box[:, 2] - ltrb_box[:, 0]
            ltrb_box_h = ltrb_box[:, 3] - ltrb_box[:, 1]

            ltrb_box = ltrb_box[(ltrb_box_w > min_limit) & (ltrb_box_h > min_limit)]
            return ltrb_box
        return ltrb_box

    def box_process(self,ltrb_box,model_w,model_h,new_w,new_h,origin_w,origin_h,dx,dy,min_limit=10):
        # 调整目标框坐标
        if len(ltrb_box) > 0:
            ltrb_box[:, [0, 2]] = ltrb_box[:, [0, 2]] * new_w / origin_w + dx
            ltrb_box[:, [1, 3]] = ltrb_box[:, [1, 3]] * new_h / origin_h + dy

            return self.box_filter(ltrb_box,model_w,model_h,min_limit)

        return ltrb_box

    def get_random_data(self, annotation_line, model_image_size, jitter=0.3, random=True):
        """实时数据增强"""
        line = annotation_line.split()
        origin_image = Image.open(line[0])
        origin_w, origin_h = origin_image.size
        model_w, model_h = model_image_size
        objs= [list(map(int, obj.split(','))) for obj in line[1:]]

        objs = torch.tensor(objs,dtype=torch.float,requires_grad=False)

        if not random:
            scale = min(model_w/origin_w, model_h/origin_h)
            new_w = int(origin_w*scale)
            new_h = int(origin_h*scale)
            dx = (model_w-new_w)//2 # 居中后，两边的留白距离
            dy = (model_h-new_h)//2

            new_image = origin_image.resize((new_w,new_h), Image.BICUBIC) # 双三次插值法
            image = Image.new('RGB', (model_w,model_h), (128,128,128)) # (128,128,128)为灰色填充
            image.paste(new_image, (dx, dy))

            image_data=self.totensor(image).float()
            objs=self.box_process(objs,model_w,model_h,new_w,new_h,origin_w,origin_h,dx,dy)
            return image_data, objs  # new_image为tensor，box为tensor float32 左上角右下角格式

        # 是否翻转图片
        if  self.rand() < 0.5 and len(objs) > 0:
            origin_image = origin_image.transpose(Image.FLIP_LEFT_RIGHT)  # 左右翻转
            objs[:, [0,2]] = origin_w - objs[:, [2,0]]

        # 调整图片大小
        new_ar = model_w / model_h * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        # 确保nh<h,nw<w
        scale = self.rand(0.7, 1)
        if new_ar < model_w/model_w:
            new_h = int(scale * model_h)
            new_w = int(new_h * new_ar)
        else:
            new_w = int(scale * model_w)
            new_h = int(new_w / new_ar)

        new_image = origin_image.resize((new_w, new_h), Image.BICUBIC)

        # 放置图片
        dx = int(self.rand(0, model_w - new_w))
        dy = int(self.rand(0, model_h - new_h))
        image = Image.new('RGB', (model_w, model_h),(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)))
        # 选用不同颜色填充
        image.paste(new_image, (dx, dy))

        # 色域变换
        image=self.colorjitter(image)

        image_data=self.totensor(image).float()
        box=self.box_process(objs,model_w,model_h,new_w,new_h,origin_w,origin_h,dx,dy)

        return image_data, box  # image_data为tensor，box为tensor float32 左上角右下角格式

    def get_random_data_with_Mosaic(self, annotation_line, model_image_size):
        model_w,model_h = model_image_size
        min_offset_x = 0.3
        min_offset_y = 0.3
        scale_low = 1 - min(min_offset_x, min_offset_y)
        scale_high = scale_low + 0.2

        image_datas = []
        box_datas = []
        index = 0

        place_x = [0, 0, int(model_w * min_offset_x), int(model_w * min_offset_x)]
        place_y = [0, int(model_h * min_offset_y), int(model_h * min_offset_y), 0]
        for line in annotation_line: # 共4行
            # 每一行进行分割
            line_content = line.split()
            # 打开图片
            origin_image = Image.open(line_content[0])
            # 图片的大小
            origin_w, origin_h = origin_image.size
            # 保存框的位置
            objs = [list(map(int, obj.split(','))) for obj in line_content[1:]]
            objs = torch.tensor(objs, dtype=torch.float, requires_grad=False)
            # 是否翻转图片
            if self.rand() < 0.5 and len(objs) > 0:
                origin_image = origin_image.transpose(Image.FLIP_LEFT_RIGHT)
                objs[:, [0, 2]] = origin_w - objs[:, [2, 0]]
            # 对输入进来的图片进行缩放
            new_ar = model_w / model_h  # 长宽比不变
            scale = self.rand(scale_low,scale_high)
            new_h = int(scale * model_h)
            new_w = int(new_h * new_ar)

            new_image = origin_image.resize((new_w, new_h), Image.BICUBIC) #插值法

            # 进行色域变换
            new_image=self.colorjitter(new_image)

            # 将图片进行放置，分别对应四张分割图片的位置
            dx = place_x[index]
            dy = place_y[index]
            image = Image.new('RGB', (model_w, model_h),(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)))
            # 随机颜色填充空白
            image.paste(new_image, (dx, dy))

            image_data = self.totensor(image)

            index = index + 1

            objs=self.box_process(objs, model_w, model_h, new_w, new_h, origin_w, origin_h, dx, dy)

            image_datas.append(image_data)
            box_datas.append(objs)

        # 将图片分割，放在一起
        cutx = np.random.randint(int(model_w * min_offset_x), int(model_w * (1 - min_offset_x)))
        cuty = np.random.randint(int(model_h * min_offset_y), int(model_h * (1 - min_offset_y)))

        new_image = torch.zeros((3,model_h,model_w),dtype=torch.float)
        new_image[:,:cuty,:cutx] = image_datas[0][:,:cuty,:cutx]
        new_image[:,cuty:,:cutx] = image_datas[1][:,cuty:,:cutx]
        new_image[:,cuty:,cutx:] = image_datas[2][:,cuty:,cutx:]
        new_image[:,:cuty,cutx:] = image_datas[3][:,:cuty,cutx:]

        # 对框进行进一步的处理
        new_boxes = torch.tensor(merge_bboxes(box_datas, cutx, cuty)).float()

        return new_image, new_boxes # new_image为tensor float32，box为tensor float32 左上角右下角格式

    def __getitem__(self, index):
        # 返回img为tensor(C,H,W),范围0到1,float32
        # 返回target为tensor(-1,5),xywh,对应在输入图片上,float32
        lines = self.train_lines
        n = self.num_lines
        index = index % n

        if self.mosaic:
            if self.flag and (index + 4) < n:
                img, y = self.get_random_data_with_Mosaic(lines[index:index + 4], self.model_image_size)
            else:
                img, y = self.get_random_data(lines[index], self.model_image_size, random=self.is_random)
            self.flag = bool(1-self.flag) # mosaic和实时数据增强轮着来
        else:
            img, y = self.get_random_data(lines[index], self.model_image_size, random=self.is_random)

        if len(y) != 0:
            box = y[:,:4]

            # 前面已经判断过是否越界了
            # 转换为x,y,w,h
            box[:, 2] = box[:, 2] - box[:, 0]
            box[:, 3] = box[:, 3] - box[:, 1]
            box[:, 0] = box[:, 0] + box[:, 2] / 2
            box[:, 1] = box[:, 1] + box[:, 3] / 2
            if y.shape[1]==5:
                y = torch.cat([box, y[:, 4:5]], dim=1) # y[:,-1:]是label
            elif y.shape[1]==6:
                y = torch.cat([box, y[:, 4:6]], dim=1)

        return img, y

# 用于DataLoader中的collate_fn
def yolo_dataset_collate(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append(img)
        bboxes.append(box)
    images = torch.stack(images,dim=0) # B,C,H,W
    return images, bboxes # 输出images为tensor(B,C,H,W),bboxes为list[B*tensor(-1,5 or 6)],都是float32,requires_grad=False


class MutiltaskDataset(YoloDataset):
    def __init__(self,detect_lines,embed_lines, image_size, mosaic=True, is_random=True):

        self.detect_lines = detect_lines
        self.embed_lines=embed_lines
        self.total_num = max(len(embed_lines),len(detect_lines))*2
        self.image_size = image_size
        self.mosaic = mosaic
        self.flag = True
        self.is_random = is_random  # 是否要实时数据增强

        self.totensor = transforms.ToTensor()
        self.colorjitter = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2)

        self.nID=0
        for line in self.embed_lines:
            line = line.split()
            if len(line) <=1 :
                continue
            target_id = [list(map(int, box.split(',')))[-1] for box in line[1:]]
            self.nID=max(max(target_id),self.nID)

        self.nID += 1
        print('nID:{}'.format(self.nID))
    def shuffle(self):
        random.shuffle(self.detect_lines)
        random.shuffle(self.embed_lines)
    def __len__(self):
        return self.total_num

    def __getitem__(self, index):
        # 返回img为tensor(C,H,W),范围0到1,float32
        # 返回target为tensor(-1,6),xywh,对应在输入图片上,float32
        index = index % self.total_num

        if index % 2==0:
            lines = self.detect_lines

        else:
            lines = self.embed_lines


        index = (index + 1) // 2
        if index >= len(lines):
            index = random.randint(0,len(lines)-1)

        if self.mosaic:
            if self.flag and (index + 4) < len(lines):
                img, y = self.get_random_data_with_Mosaic(lines[index:index + 4], self.image_size)
            else:
                img, y = self.get_random_data(lines[index], self.image_size, random=self.is_random)
            self.flag = bool(1-self.flag) # mosaic和实时数据增强轮着来
        else:
            img, y = self.get_random_data(lines[index], self.image_size, random=self.is_random)
        if len(y) != 0:
            box = y[:,:4]
            # 前面已经判断过是否越界了
            # 转换为x,y,w,h
            box[:, 2] = box[:, 2] - box[:, 0]
            box[:, 3] = box[:, 3] - box[:, 1]
            box[:, 0] = box[:, 0] + box[:, 2] / 2
            box[:, 1] = box[:, 1] + box[:, 3] / 2

            y = torch.cat([box, y[:, -2:]], dim=1) # y[:,-2:]是cls_label和trackid
        return img, y
'''

if __name__=='__main__':
    #调式用
    from torch.utils.data import DataLoader
    from PIL import ImageFont,ImageDraw
    def toPIL(tensor):
        image = tensor.cpu().clone()
        image = image.squeeze(0)
        image = transforms.ToPILImage()(image)
        return image


    def save_result_img(image,top_bboxes):
        # 把有bbox的检测图片也保存下来
        for i in range(len(top_bboxes)):

            top, left, bottom, right = top_bboxes[i, 1].cpu().numpy(),\
                                       top_bboxes[i, 0].cpu().numpy(), top_bboxes[i, 3].cpu().numpy(), top_bboxes[i, 2].cpu().numpy()

            top = top - 5
            left = left - 5
            bottom = bottom + 5
            right = right + 5

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(np.shape(image)[1], np.floor(right + 0.5).astype('int32'))

            font = ImageFont.truetype(font='model_data/simhei.ttf',
                                      size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype(
                                          'int32'))
            label = '{}'.format(int(top_bboxes[i,-1]))
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')


            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])
            thickness = 2

            draw = ImageDraw.Draw(image)

            # 画框框
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],)
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            del draw
        return image

    annotation_path = './dataset/helmet_detect_test_anno.txt'
    with open(annotation_path) as f:
        lines = f.readlines()
    dataset=YoloDataset(lines,(1088,608),mosaic=True,is_random=True)
    gen=DataLoader(dataset,batch_size=1,collate_fn=yolo_dataset_collate,shuffle=True)

    i=0
    for img,target in gen:
        img=toPIL(img)
        target=target[0]
        if(len(target)==0):
            img.save('./results/'+str(i)+'.jpg')
            i+=1
            continue
        box=target
        box[:,0],box[:,2]=box[:,0]-box[:,2]/2,box[:,0]+box[:,2]/2
        box[:,1],box[:,3]=box[:,1]-box[:,3]/2,box[:,1]+box[:,3]/2

        result=save_result_img(img,box)
        result.save('./results/'+str(i)+'.jpg')
        i+=1
'''



