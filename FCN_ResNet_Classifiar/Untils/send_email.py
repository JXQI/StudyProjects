# #!/usr/bin/python
# # coding:utf-8
# """
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
# author：helun@baidu.com
# usge：python performance_statistics.py
# date:2019-03-04
# Catalog:./src_py/
# """
# import os
# import re
# import time
# import subprocess
# import getpass
# from email.mime.base import MIMEBase
# from email.mime.text import MIMEText
# from email.mime.application import MIMEApplication
# from email.mime.multipart import MIMEMultipart
# from email.utils import parseaddr, formataddr
# from email.header import Header
# import socket
# import smtplib
#
# #源配置文件
# source_conf = "../conf/performance/performance_task.conf"
# #参数配置文件
# performance_path = "../conf/performance"
#
# def get_conf_value(conf_file, name):
#     """
#     :brief:获取配置文件中相关key的值
#     :param conf_file :配置文件
#     :param name:配置的key
#     :return 配置的值
#     """
#     pattern = re.compile(r'\s*:?\s*')
#     with open(conf_file, 'r+') as f:
#         lines = f.readlines()
#         for line in lines:
#             list = pattern.split(line)
#             if list[0] == name:
#                 return list[1]
#
#
# def get_data(path):
#     """
#     获取结果文件数据
#     :param path: 结果文件路径(result.txt)
#     :return: 数据
#     """
#     result = []
#     with open(path, 'r') as f:
#         while True:
#             lines = f.readline().strip('\n')
#             if not lines:
#                 break
#             data = lines.split('\t')
#             result.append(data)
#     return result
#
#
# def make_table(result):
#     """
#     制作表格
#     :param job_num: 任务数
#     :return:html格式内容
#     """
#     table_head = '<table border="1"> \
#         <tr>\
#         <th>并发数</th>\
#         <th>aue</th>\
#         <th>ctp</th>\
#         <th>per</th>\
#         <th>QPS</th>\
#         <th>平均首包响应时间</th>\
#         <th>成功率</th>\
#         <th>dealtime</th>\
#         <th>in_queue_time</th>\
#         <th>total_time</th>\
#         <th>audio_tm</th>\
#     	<th>real_tm_factor</th>\
#         <th>cal_real_tm_factor</th>\
#         </tr>'
#     table_content = ''
#     for i in range(0, len(result)):
#         table_content = table_content + '\
#                                         <tr>\
#                                         <td>%s</td>\
#                                         <td>%s</td>\
#                                         <td>%s</td>\
#                                         <td>%s</td>\
#                                         <td>%s</td>\
#                                         <td>%s</td>\
#                                         <td>%s</td>\
#                                         <td>%s</td>\
#                                         <td>%s</td>\
#                                         <td>%s</td>\
#                                         <td>%s</td>\
#                                         <td>%s</td>\
#                                         <td>%s</td>\
#                                         </tr>' %(result[i][0],\
#                                                   result[i][1], \
#                                                   result[i][2], \
#                                                   result[i][3], \
#                                                   result[i][4], \
#                                                   result[i][5], \
#                                                   result[i][6], \
#                                                   result[i][7], \
#                                                   result[i][8], \
#                                                   result[i][9], \
#                                                   result[i][10],\
#                                                   result[i][11],
#                                                   result[i][12])
#     return table_head + table_content + '</table>' + '</br>'
#
# def sendEmail(context):
#     """
#     发送邮件
#     :param context:  html格式内容
#     :return: 无
#     """
#     msg = MIMEMultipart()
#     print(msg)
#     part = MIMEText(context, 'html', 'utf8')
#     msg.attach(part)
#     msg["Accept-Language"] = "zh-CN"
#     msg['Subject'] = '新性能测试数据--tts'
#     user_name = getpass.getuser()
#     myname = socket.getfqdn(socket.gethostname())
#     mailFrom = user_name + '@' + myname
#     print(mailFrom)
#     mailTo = ['18811610296@163.com']
#     def _format_addr(s):
#         name, addr = parseaddr(s)
#         #print name
#         #print addr
#         #将邮件的name转换成utf-8格式，addr如果是unicode，则转换utf-8输出，否则直接输出addr
#         return formataddr((\
#             Header(name, 'utf-8').encode(),\
#             addr.encode("utf-8") if isinstance(addr,str) else addr))
#     #msg['From'] = _format_addr(u'<%s>' % mailFrom)
#     msg['From'] = mailFrom
#     #msg['To'] = _format_addr(u'<%s>' % mailTo)
#     msg['To'] = mailTo
#     #send_smtp = smtplib.SMTP('localhost')
#     send_smtp = smtplib.SMTP()
#     ans = send_smtp.sendmail(mailFrom, mailTo, msg.as_string())
#     send_smtp.close()
#     return ans
#
# if __name__ == '__main__':
#     print('main is runing')
#
#     #获取结果文件
#     # f = open('temp', 'r');
#     # dir = f.readline().strip('\n')
#     # file = "./" + str(dir) + '/final_result'
#     # result = get_data(file)
#     # print(result)
#     # #整合表格内容
#     # data = make_table(result)
#     data="test"
#     sendEmail(data)
#     print('runing is ok')


#新的方法
#!/usr/bin/python3
'''
import smtplib
from email.mime.text import MIMEText
from email.header import Header

# 第三方 SMTP 服务
mail_host="smtp.163.com"  #设置服务器
mail_user="18811610296@163.com"    #用户名
mail_pass="VWTEZUGJIHVODSAS"   #口令


sender = '18811610296@163.com'
receivers = ['2390139915@qq.com']  # 接收邮件，可设置为你的QQ邮箱或者其他邮箱

message = MIMEText('Python 邮件发送测试...', 'plain', 'utf-8')
message['From'] = Header("W3Cschool教程", 'utf-8')
message['To'] =  Header("测试", 'utf-8')

subject = 'Python SMTP 邮件测试'
message['Subject'] = Header(subject, 'utf-8')


try:
    smtpObj = smtplib.SMTP()
    smtpObj.connect(mail_host, 465)    # 25 为 SMTP 端口号
    smtpObj.login(mail_user,mail_pass)
    smtpObj.sendmail(sender, receivers, message.as_string())
    print ("邮件发送成功")
except smtplib.SMTPException:
    print ("Error: 无法发送邮件")
'''
import smtplib
from email.mime.text import MIMEText
# 引入smtplib和MIMEText
from time import sleep


def sentemail():
    host = 'smtp.163.com'
    # 设置发件服务器地址
    port = 465
    # 设置发件服务器端口号。注意，这里有SSL和非SSL两种形式，现在一般是SSL方式
    sender = '18811610296@163.com'
    # 设置发件邮箱，一定要自己注册的邮箱
    pwd = 'VWTEZUGJIHVODSAS'
    # 设置发件邮箱的授权码密码，根据163邮箱提示，登录第三方邮件客户端需要授权码
    mailto = ['2390139915@qq.com','18811610296@163.com','2403454992@qq.com','2361753941@qq.com']
    # 设置邮件接收人，可以是QQ邮箱
    body = '<h1>模型训练完成</h1><p>你是个大傻子,能不能猜到我是谁?</p>'
    # 设置邮件正文，这里是支持HTML的
    msg = MIMEText(body, 'html')
    # 设置正文为符合邮件格式的HTML内容
    msg['subject'] = '打卡通知'
    # 设置邮件标题
    msg['from'] = sender
    # 设置发送人
    for receiver in mailto:
        msg['to'] = receiver
        # 设置接收人
        try:
            s = smtplib.SMTP_SSL(host, port)
            # 注意！如果是使用SSL端口，这里就要改为SMTP_SSL
            s.login(sender, pwd)
            # 登陆邮箱
            s.sendmail(sender, receiver, msg.as_string())
            # 发送邮件！
            print('Done.sent email success')
        except smtplib.SMTPException:
            print('Error.sent email fail')


if __name__ == '__main__':
    sentemail()