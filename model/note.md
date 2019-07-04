ssd_aspp_b_1中，inter_planes 大小考虑微调 当前是input_plane//8 ,之后考虑换为24
可以考虑讲extral layer中的后半部分结构也改为aspp试试看
之后frcnn以及其他结构中的FPN结构用aspp替代下看看
之后考虑加入bn与relu做微调


考虑将extra layer最后两层也替换为aspp

考虑为aspp添加shotcut


more priors 目前直接加到相应的model中去了，之后可以考虑单独拆出来看看


aspp_a 多个branch的aspp，每个branch前要先进行一个conv1x1操作      0.7885
aspp_a_1 先进行conv1x1操作，再进行branch  当前a_1效果较好         0.7905
aspp_b_1 分branch，每个branch中，先conv1x1，后串联 r->2*r -> 3*r  0.7910
aspp_b_2 与b_1有点类似，但是每层的ft都要输出                      0.7933
aspp_b_3 与b_2类似，先进行conv1x1                                0.7921
aspp_b_2_left_last                                              0.7941 


a 与 a_1 的反应与 b_2 与 b_3 的反应不太一样，这里考虑下元婴！！！！！！！！！！！！！


注意conig中的aspect_ratios要和ssd中的mbox对应起来，尤其是数量！！！！！！！！！！！！



考虑大幅添加bn！！！！！！！！！！！！！！




之后基本确定增值之后，可以复现RFBNet中的aspp-s，然后改进，看在其上面的增值



最后尝试在vgg中添加backbone！！！！！！！！！！！！！！！！！！！



注意，当前b_2_relu还是跟在extra中rate为6,3,2,1的对比
b_2_mid_concat_relu 是和b_2_relu对比



left last 有效，最后一层不改



concat 可以考虑attention机制


aspp_relu当前输出的是未relu后的结果，可以尝试直接输出relu结果

当前看extra rate 貌似不如之前直接6,3,2,1 来的好，这块可以再考虑下原因，是否1259的确不太好
当前left_last 貌似提高一些,0.7941
mid_relu  提到0.7955
relu 当前看不出来

关于rate！！！ 好好再提提