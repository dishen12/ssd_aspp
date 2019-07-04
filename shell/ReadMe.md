i extra block中将前两个s=2的卷积替换为dilate con
a 将extra block所有s=2替换为dilate conv，即保持ft尺寸大小不变（19x19）
h 前两个s=2替换为dilate con（19x19），后两个下采样（10,5）

注意：目前ration啥的，都没改