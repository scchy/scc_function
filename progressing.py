# python 3.6
# author: Scc_hy 
# create date: 2019-08-20
# Function： 列表遍历进度条


class My_Progress():
    """
    对列表遍历，返回对应遍历进度  
    param lsit_i: list 
    param width: int 进度条长度 

    将列表读入，每次循环会自动遍历
    返回进度条  形状如下:  
    [###################                          ] 42.9%

    例子：
        m = ['asdas', 'asdas','qwe124', 'asd112', '12asdx', '12asdx', '12asdx']
        p = My_Progress(m, width = 45)
        for i in m:
            msg = p.progress()
            print('目前打印: {}, 进度: {}'.format(i, msg))
    """
    def __init__(self, list_i, width = 25):
        self.list_i = list_i
        self.list_long = len(list_i)
        self.width = width
        self.start = 0

    def get_progress(self, percent = 0):
        """
        百分比显示方式 
        """
        left = percent * self.width // self.list_long
        pct = percent * self.width / self.list_long
        right = self.width - left
        left_now = '#' * left
        right_now = ' ' * right
        mult = 100 / self.width
        return "[{}{}] {:.1f}%".format(left_now, right_now, pct * mult)

    def flush_percent(self):
        """
        刷新开始位置
        """
        self.start += 1
    
    def progress(self):
        """
        输出进度条
        """
        if self.start <= self.list_long:
            self.flush_percent()
            msg = self.get_progress(self.start)
            return msg


if __name__ == '__main__':
    m = ['asdas', 'asdas','qwe124', 'asd112', '12asdx', '12asdx', '12asdx']
    p = My_Progress(m, width = 45)
    for i in m:
        msg = p.progress()
        print('目前打印: {}, 进度: {}'.format(i, msg))
