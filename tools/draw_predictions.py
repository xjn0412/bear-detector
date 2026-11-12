"""Draw actual predict.py JSON boxes; never invent or adjust model coordinates."""
import argparse
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image',required=True)
    p.add_argument('--predictions',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    data=json.loads(Path(a.predictions).read_text())
    image=Image.open(a.image).convert('RGB')
    if list(image.size)!=[data['width'],data['height']]:
        raise ValueError('Predictions and image size differ')
    draw=ImageDraw.Draw(image)
    size=max(18,round(image.width/48))
    try:font=ImageFont.truetype('DejaVuSans-Bold.ttf',size)
    except OSError:font=ImageFont.load_default()
    for d in data['detections']:
        box=d['xyxy']
        text=f"{d.get('name', 'class '+str(d['class_index']))} {d['score']:.3f}"
        draw.rectangle(box,outline=(255,193,72),width=max(3,size//5))
        b=draw.textbbox((0,0),text,font=font)
        w,h=b[2]-b[0]+20,b[3]-b[1]+16
        x,y=box[0],max(0,box[1]-h)
        draw.rectangle((x,y,x+w,y+h),fill=(255,193,72))
        draw.text((x+10,y+8-b[1]),text,font=font,fill=(35,26,16))
    out=Path(a.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    image.save(out,quality=94)
    print(out)


if __name__=='__main__':main()
