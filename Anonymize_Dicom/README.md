# Anonymize_Dicom  
将dicom格式文件转换成nii格式：  
1. 两者的区别

 diff | dicom  |  nii  
 :---- | :-----:  |  :---:  
头信息 | 有 | 无  
数据类型 | 2D | 3D  
是否匿名 | 否 | 是  

2. 程序使用说明
  
>所需要的环境：  
>>python 3.7.9  
>>dicom2nifti 2.3.0  
>>tqdm  
>>argparse  
3. 环境配置：
  
>pip install dicom2nifti  
>pip install tqdm  
4. 使用方式:    
>>python main.py src_path dst_path  

>参数说明：  
>>src_path:包括多个病人的dicom文件夹所在的路径([目录格式参考](src_path.png) )   
>>dst_path:需要保存nii文件的目录  

>运行结果说明：  
>>在dst_path下生成相同数量的.nii文件  
  
5. 参考资料：
  
>[dicom头信息](https://www.cnblogs.com/XDU-Lakers/p/9863114.html)  
>[dicom匿名化](https://pydicom.github.io/pydicom/dev/auto_examples/metadata_processing/plot_anonymize.html#anonymize-dicom-data)  



