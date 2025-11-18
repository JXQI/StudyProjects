# png2coco
## 功能：  
1.将png实例的标注转换为coco format  
2.从nii.gz格式的CT图像生成切片数据，并且生成对应的mask  

## 文件结构：  
generate-slice.py 生成切片数据，保存在axial_slice文件夹下，注意生成的mask包含多个实例，mask不同的值代表不同的类别  
label2mask.py 将一张包含多个实例的mask转换成多个只有一个实例的二进制mask  
  
generate-slice2.py  生成切片数据，保存在axial_slice文件夹下，注意这个生成的是mask是每个实例的mask，二进制的图像  
png2coco.py 将生成的切片和每个实例的二进制mask数据转换成coco格式的数据集，并且保存在指定的位置  
  
visualization.py 查看生成的coco格式数据集，加载mask到原图上  

## 使用方法：  
生成切片数据并且保存对应的mask(mask包含所有的实例)：python generate-slice.py  
将多个实例的mask转换成多个单个实例的二进制mask：python label2mask.py  
  
生成切片数据和对应的实例的二进制mask：python generate-slice2.py  
将数据集转换成coco格式：python png2coco.py  

  
具体参数看每个文件内部配置，注意一下几点：  
1.generate-slice2.py 中窗口的值 windows = 1400  leval = 600  
2.generate-slice2.py 中area的值，决定保留实例标注的像素值  




