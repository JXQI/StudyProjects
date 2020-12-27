# png2coco
功能：  
1.将png实例的标注转换为coco format  
2.从nii.gz格式的CT图像生成切片数据，并且生成对应的mask  

文件结构：  
generate-slice.py 生成切片数据，保存在axial_slice文件夹下，注意生成的mask包含多个实例，mask不同的值代表不同的类别  
generate-slice2.py  

