"""Exportable bilinear, zero-padded grid sampling without ONNX GridSample.

Fixed spatial shapes are intentional for the detector's batch1x640 export.
Training continues to use native PyTorch grid_sample.
"""
import torch

def grid_sample(input,grid,mode='bilinear',padding_mode='zeros',align_corners=False):
 if mode!='bilinear' or padding_mode!='zeros' or align_corners not in [False,None]:
  raise ValueError('Portable sampling supports bilinear/zeros/align_corners=False only')
 n,c,h,w=input.shape;ho,wo=grid.shape[1:3]
 # Keep coordinate arithmetic FP32, matching the protected decoder.
 gx=grid[...,0].float();gy=grid[...,1].float()
 x=((gx+1)*w-1)*.5;y=((gy+1)*h-1)*.5
 x0=torch.floor(x);y0=torch.floor(y);x1=x0+1;y1=y0+1
 flat=input.reshape(n,c,h*w)
 def sample(xx,yy):
  valid=(xx>=0)&(xx<w)&(yy>=0)&(yy<h)
  index=(yy.clamp(0,h-1).to(torch.int64)*w+xx.clamp(0,w-1).to(torch.int64)).reshape(n,1,ho*wo).expand(n,c,ho*wo)
  value=torch.gather(flat,2,index).reshape(n,c,ho,wo)
  return value*valid.unsqueeze(1).to(value.dtype)
 a=sample(x0,y0);b=sample(x1,y0);c0=sample(x0,y1);d=sample(x1,y1)
 return a*((x1-x)*(y1-y)).unsqueeze(1)+b*((x-x0)*(y1-y)).unsqueeze(1)+c0*((x1-x)*(y-y0)).unsqueeze(1)+d*((x-x0)*(y-y0)).unsqueeze(1)
