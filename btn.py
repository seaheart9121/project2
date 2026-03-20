#coding:utf-8
import pygame.font


class Button:
    def __init__(self,screen,centenxy,width,height,button_color,text_color,msg,size):
        # 初始化按钮的属性
        self.screen=screen
        # 设置按钮的宽和高
        self.width,self.height=width,height
        # 设置按钮的背景色
        self.button_color=button_color
        self.text_color=text_color
        # 设置按钮上文字的字体和大小
        self.font=pygame.font.SysFont('SimHei',size)
        # 设置按钮的大小
        self.rect=pygame.Rect(0,0,self.width,self.height)
        # 设置按钮的中心位置
        self.rect.centerx=centenxy[0]-self.width/2+2
        self.rect.centery=centenxy[1]-self.height/2+2

        self.deal_msg(msg) # msg按钮上的文本

    def deal_msg(self,msg):
         # 将文字转成图片，放在按钮
        self.msg_img=self.font.render(msg,True,self.text_color,self.button_color)

        self.msg_img_rect=self.msg_img.get_rect() # 将文字转成图片之后的矩形

        # 矩形的中心点  self.rect 按钮的矩形
        self.msg_img_rect.center=self.rect.center

    # 绘制按钮
    def draw_button(self):
        self.screen.fill(self.button_color,self.rect)
        self.screen.blit(self.msg_img,self.msg_img_rect)



